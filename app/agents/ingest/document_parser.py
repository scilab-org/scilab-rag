"""Document parsing service using Docling."""

import datetime
import logging
import time
import json
import base64
from io import BytesIO
from typing import Optional, Tuple

import requests
from PIL import Image
from pydantic import AnyUrl

from docling.datamodel.base_models import InputFormat, OpenAiApiResponse, VlmStopReason
from docling.datamodel.pipeline_options import (
    PictureDescriptionApiOptions,
    PdfPipelineOptions,
    CodeFormulaVlmOptions,
)
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.datamodel.stage_model_specs import VlmModelSpec
from docling.datamodel.pipeline_options_vlm_model import ResponseFormat
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from app.core.config import settings
from app.agents.ingest.prompts import IMAGE_DESCRIPTION_PROMPT, FORMULA_DESCRIPTION_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_PAGES_PER_BATCH = 5
DELAY_BETWEEN_BATCHES_SECONDS = 0

# ── Retry-enabled replacement for docling's api_image_request ─────────────────
# Docling's built-in api_image_request has no retry logic; transient 5xx errors
# from OpenRouter cause it to silently return empty strings for every image.
# This wrapper adds exponential-backoff retries.

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 2  # seconds: 2, 4, 8 …


def _api_image_request_with_retry(
    image: Image.Image,
    prompt: str,
    url: AnyUrl,
    timeout: float = 20,
    headers: Optional[dict[str, str]] = None,
    **params,
) -> Tuple[str, Optional[int], VlmStopReason]:
    """Drop-in replacement for docling.utils.api_image_request.api_image_request
    that retries on transient HTTP errors (5xx, timeouts)."""

    img_io = BytesIO()
    image = image.copy()
    image = image.convert("RGBA")
    try:
        image.save(img_io, "PNG")
    except Exception as e:
        logger.error(f"Corrupt PNG of size {image.size}: {e}")
        return "", 0, VlmStopReason.UNSPECIFIED

    image_base64 = base64.b64encode(img_io.getvalue()).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    payload = {"messages": messages, **params}
    headers = headers or {}

    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            r = requests.post(
                str(url), headers=headers, json=payload, timeout=timeout
            )

            if r.status_code >= 500:
                last_error = f"HTTP {r.status_code}: {r.text[:200]}"
                wait = _RETRY_BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    f"API returned {r.status_code} (attempt {attempt + 1}/{_MAX_RETRIES}), "
                    f"retrying in {wait}s …"
                )
                time.sleep(wait)
                continue

            if not r.ok:
                logger.error(f"API error (non-retryable): {r.status_code} — {r.text[:300]}")
                return "", 0, VlmStopReason.UNSPECIFIED

            api_resp = OpenAiApiResponse.model_validate_json(r.text)
            generated_text = api_resp.choices[0].message.content.strip()
            num_tokens = api_resp.usage.total_tokens
            stop_reason = (
                VlmStopReason.LENGTH
                if api_resp.choices[0].finish_reason == "length"
                else VlmStopReason.END_OF_SEQUENCE
            )
            return generated_text, num_tokens, stop_reason

        except requests.exceptions.Timeout:
            last_error = "Request timed out"
            wait = _RETRY_BACKOFF_BASE ** (attempt + 1)
            logger.warning(
                f"API timeout (attempt {attempt + 1}/{_MAX_RETRIES}), "
                f"retrying in {wait}s …"
            )
            time.sleep(wait)
        except Exception as e:
            logger.error(f"Unexpected error in image API request: {e}")
            return "", 0, VlmStopReason.UNSPECIFIED

    logger.error(
        f"All {_MAX_RETRIES} retries exhausted for image API request. "
        f"Last error: {last_error}"
    )
    return "", 0, VlmStopReason.UNSPECIFIED


def _patch_docling_api_image_request():
    """Monkey-patch docling's api_image_request with our retry-enabled version.

    Because several docling modules use ``from docling.utils.api_image_request
    import api_image_request`` (creating a local name binding), we must patch
    every module that holds a reference — not just the source module.
    """
    import docling.utils.api_image_request as _src_mod

    _src_mod.api_image_request = _api_image_request_with_retry

    # Patch all known consumers that import the function by name
    _consumer_modules = [
        "docling.models.stages.picture_description.picture_description_api_model",
        "docling.models.inference_engines.vlm.api_openai_compatible_engine",
        "docling.models.vlm_pipeline_models.api_vlm_model",
    ]
    import importlib, sys

    for mod_name in _consumer_modules:
        mod = sys.modules.get(mod_name)
        if mod is None:
            try:
                mod = importlib.import_module(mod_name)
            except ImportError:
                continue
        if hasattr(mod, "api_image_request"):
            mod.api_image_request = _api_image_request_with_retry

    logger.info("Patched docling api_image_request with retry logic (max %d retries)", _MAX_RETRIES)


# Apply the patch at module load time
_patch_docling_api_image_request()

def run_hybrid(doc, max_tokens: int = 6000):
    import tiktoken
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.encoding_for_model("gpt-4o-mini"),
        max_tokens=max_tokens,
    )

    chunker = HybridChunker(tokenizer=tokenizer)
    return list(chunker.chunk(doc))


def _get_page_count(input_doc_path: str) -> int:
    """Get total page count of a PDF."""
    import fitz
    with fitz.open(input_doc_path) as doc:
        return len(doc)


def _create_converter() -> DocumentConverter:
    """Create a configured DocumentConverter instance."""
    # Use image-specific model for picture descriptions
    image_model = settings.OPENROUTER_IMAGE_MODEL
    accelerator_options = AcceleratorOptions(
        num_threads=4,
        device=AcceleratorDevice.CPU,  
    )
    
    picture_desc_api_option = PictureDescriptionApiOptions(
        url=AnyUrl(settings.OPENROUTER_API_URL_CHAT),
        prompt=IMAGE_DESCRIPTION_PROMPT,
        params=dict(
            model=image_model,
            max_tokens=1024,    
        ),
        headers={
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "X-Title": "docling-pdf-parser",
        },
        timeout=120,
        concurrency=4,
    )

    code_formula_model_spec = VlmModelSpec(
        name="Qwen VL for Code and Formula",
        default_repo_id="Qwen/Qwen2-VL-7B-Instruct",  
        prompt=FORMULA_DESCRIPTION_PROMPT,
        response_format=ResponseFormat.MARKDOWN,
    )

    code_formula_api_options = ApiVlmEngineOptions(
        url=AnyUrl(settings.OPENROUTER_API_URL_CHAT),
        headers={
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "X-Title": "docling-pdf-parser",
        },
        params={
            "model": settings.OPENROUTER_IMAGE_MODEL,
            "max_tokens": 4096,
        },
        concurrency=4,
    )

    code_formula_options = CodeFormulaVlmOptions(
        model_spec=code_formula_model_spec,
        engine_options=code_formula_api_options,
        scale=1.0,
        extract_code=True,
        extract_formulas=True,
    )

    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_picture_description=True,
        picture_description_options=picture_desc_api_option,
        do_code_enrichment=True,
        do_formula_enrichment=True,
        code_formula_options=code_formula_options,
        enable_remote_services=True,
        generate_picture_images=True,
        images_scale=1,
        accelerator_options=accelerator_options,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
def parse_document(
    input_doc_path: str,
    hybrid_max_tokens: int = settings.HYBRID_MAX_TOKENS,
) -> dict:

    start_time = datetime.datetime.now()
    total_pages = _get_page_count(input_doc_path)
    converter = _create_converter()

    logger.info(f"Converting {total_pages} pages in a single pass")

    conv_res = converter.convert(
        source=input_doc_path,
        raises_on_error=False,
    )

    chunks = run_hybrid(conv_res.document, max_tokens=hybrid_max_tokens)

    duration = (datetime.datetime.now() - start_time).total_seconds()
    logger.info(f"PDF conversion completed in {duration:.2f} seconds — {len(chunks)} chunks")

    return chunks_to_minimal_json(chunks)

def chunks_to_minimal_json(chunks):
    parsed_chunks = []

    for chunk in chunks:
        meta = getattr(chunk, "meta", None)

        headings = None
        captions = None

        if meta:
            headings = getattr(meta, "headings", None)
            captions = getattr(meta, "captions", None)
        
        obj = {
            "text": chunk.text,
            "headings": headings,
        }

        if captions: 
            obj["captions"] = captions

        parsed_chunks.append(obj)
        
    return {"chunks": parsed_chunks}

def parse_document_per_batch(
    input_doc_path: str,
    pages_per_batch: int = DEFAULT_PAGES_PER_BATCH,
    hybrid_max_tokens: int = settings.HYBRID_MAX_TOKENS,
) -> dict:
    start_time = datetime.datetime.now()
    total_pages = _get_page_count(input_doc_path)
    converter = _create_converter()

    logger.info(f"Phase 1: converting {total_pages} pages in batches of {pages_per_batch}")

    markdown_parts = []

    for batch_start in range(1, total_pages + 1, pages_per_batch):
        batch_end = min(batch_start + pages_per_batch - 1, total_pages)
        logger.info(f"  Layout/table/VLM pass: pages {batch_start}-{batch_end}/{total_pages}")

        conv_res = converter.convert(
            source=input_doc_path,
            page_range=(batch_start, batch_end),
            raises_on_error=False,
        )

        markdown_parts.append(conv_res.document.export_to_markdown())

        if batch_end < total_pages:
            time.sleep(DELAY_BETWEEN_BATCHES_SECONDS)

    logger.info("Phase 2: re-parsing combined Markdown for structure-aware chunking")

    combined_markdown = "\n\n".join(markdown_parts)

    import tempfile, os
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(combined_markdown)
        tmp_md_path = tmp.name

    try:
        md_converter = DocumentConverter()   
        md_result    = md_converter.convert(tmp_md_path)
        full_doc     = md_result.document
    finally:
        os.unlink(tmp_md_path)

    # ── Chunk once on the full document ───────────────────────────────────────
    chunks = run_hybrid(full_doc, max_tokens=hybrid_max_tokens)

    duration = (datetime.datetime.now() - start_time).total_seconds()
    logger.info(
        f"Conversion complete in {duration:.2f}s — "
        f"{len(chunks)} chunks across {total_pages} pages"
    )

    return chunks_to_minimal_json(chunks)


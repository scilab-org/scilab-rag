"""Document parsing service using Docling."""

import datetime
import logging
import time

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PictureDescriptionApiOptions,
    PdfPipelineOptions,
    CodeFormulaVlmOptions,
)
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.datamodel.stage_model_specs import VlmModelSpec
from docling.datamodel.pipeline_options_vlm_model import ResponseFormat

from docling.document_converter import DocumentConverter, PdfFormatOption
from pydantic import AnyUrl

from app.core.config import settings
from app.core.prompts import IMAGE_DESCRIPTION_PROMPT, FORMULA_DESCRIPTION_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_PAGES_PER_BATCH = 5
DELAY_BETWEEN_BATCHES_SECONDS = 2


def _get_page_count(input_doc_path: str) -> int:
    """Get total page count of a PDF."""
    import fitz
    with fitz.open(input_doc_path) as doc:
        return len(doc)


def _create_converter() -> DocumentConverter:
    """Create a configured DocumentConverter instance."""
    model = settings.VLM_MODEL

    picture_desc_api_option = PictureDescriptionApiOptions(
        url=AnyUrl(settings.OPENROUTER_API_URL_CHAT),
        prompt=IMAGE_DESCRIPTION_PROMPT,
        params=dict(model=model),
        headers={
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "X-Title": "docling-pdf-parser",
        },
        timeout=120,
        batch_size=1,
    )

    code_formula_model_spec = VlmModelSpec(
        name="Qwen VL for Code and Formula",
        default_repo_id=settings.VLM_MODEL,
        prompt=FORMULA_DESCRIPTION_PROMPT,
        response_format=ResponseFormat.MARKDOWN,
    )

    code_formula_api_options = ApiVlmEngineOptions(
        url=AnyUrl(settings.OPENROUTER_API_URL_CHAT),
        headers={
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "X-Title": "docling-pdf-parser",
        },
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
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def parse_document(
    input_doc_path: str,
    pages_per_batch: int = DEFAULT_PAGES_PER_BATCH,
) -> str:
    """
    Parse a PDF document using Docling with chunked processing.
    
    Args:
        input_doc_path: Path to the PDF file
        pages_per_batch: Number of pages per batch
    
    Returns:
        Full extracted text from the document
    """
    start_time = datetime.datetime.now()
    total_pages = _get_page_count(input_doc_path)
    converter = _create_converter()
    
    logger.info(f"Processing {total_pages} pages in batches of {pages_per_batch}")
    
    text_parts = []
    for batch_start in range(1, total_pages + 1, pages_per_batch):
        batch_end = min(batch_start + pages_per_batch - 1, total_pages)
        
        logger.info(f"Processing pages {batch_start}-{batch_end}/{total_pages}")
        
        conv_res = converter.convert(
            source=input_doc_path,
            page_range=(batch_start, batch_end),
            raises_on_error=False,
        )
        text_parts.append(conv_res.document.export_to_text())
        
        if batch_end < total_pages:
            time.sleep(DELAY_BETWEEN_BATCHES_SECONDS)
    
    duration = (datetime.datetime.now() - start_time).total_seconds()
    logger.info(f"PDF conversion completed in {duration:.2f} seconds")
    
    return "\n\n".join(text_parts)



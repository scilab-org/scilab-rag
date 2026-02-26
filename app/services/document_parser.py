"""Document parsing service using Docling."""

import datetime
import logging

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

def parse_document(
    input_doc_path: str,
) -> str:
    """
    Parse a PDF document using Docling.
    
    Args:
        input_doc_path: Path to the PDF file
        do_picture_description: Enable AI-generated descriptions for images
        do_formula_enrichment: Enable formula enrichment
    
    Returns:
        ConversionResult from Docling
    """
    model = settings.VLM_MODEL


    picture_desc_api_option = PictureDescriptionApiOptions(
        url=AnyUrl(settings.OPENROUTER_API_URL_CHAT),
        prompt=IMAGE_DESCRIPTION_PROMPT,
        params=dict(model=model),
        headers={
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "X-Title": "docling-pdf-parser",
        },
        timeout=60,
    )
    
    # Configure code and formula extraction using OpenRouter API
    code_formula_model_spec = VlmModelSpec(
        name="Qwen VL for Code and Formula", 
        default_repo_id=settings.VLM_MODEL,
        prompt=FORMULA_DESCRIPTION_PROMPT,
        response_format=ResponseFormat.MARKDOWN,
    )
    
    code_formula_api_options = ApiVlmEngineOptions(
        url=AnyUrl(settings.OPENROUTER_API_URL_BASE),
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
        do_picture_description=True,
        picture_description_options=picture_desc_api_option,
        do_code_enrichment=True,
        do_formula_enrichment=True,
        code_formula_options=code_formula_options,
        enable_remote_services=True,
        generate_picture_images=True,
        images_scale=1,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    start_time = datetime.datetime.now()
    
    conv_res = converter.convert(source=input_doc_path)    
    
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info("PDF conversion completed in %.2f seconds", duration)

    # save_markdown_to_text_file(conv_res, input_doc_path)
    return conv_res.document.export_to_text()



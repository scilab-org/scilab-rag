"""
Router for reformatting paper LaTeX content to match a conference/journal template style.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.agents.formatter import FormatterAgent
from app.api.api_models.request import FormatPaperStyleRequest
from app.api.api_models.response import FormatPaperStyleResponse
from app.auth import CurrentUser
from app.core.dependencies import get_chat_llm

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/papers",
    tags=["Papers"],
)


@router.post("/format-style", response_model=FormatPaperStyleResponse)
async def format_paper_style(
    request: FormatPaperStyleRequest,
    user: CurrentUser,
):
    """
    Reformat a paper's LaTeX content to match a conference/journal template style.

    Receives the full LaTeX content of the paper and the conference/journal template,
    then uses an LLM agent to restructure the paper following the template's formatting
    without modifying any written content.
    """
    try:
        agent = FormatterAgent(llm=get_chat_llm())
        formatted_content = await agent.format_paper_to_style(
            paper_content=request.paper_content,
            template_content=request.template_content,
        )
        return FormatPaperStyleResponse(formatted_content=formatted_content)

    except Exception as e:
        logger.exception("Failed to format paper style: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to format paper to template style: {str(e)}",
        )

import logging

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM

from app.agents.formatter.prompts import FORMAT_PAPER_STYLE_PROMPT, FORMAT_PAPER_STYLE_USER_TEMPLATE

logger = logging.getLogger(__name__)


class FormatterAgent:
    """
    Single-pass LLM agent that reformats a paper's LaTeX content to match
    a conference/journal template style.

    Preserves all written content verbatim — only changes structural LaTeX
    formatting such as document class, packages, author blocks, section
    commands, and bibliography style.
    """

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    async def format_paper_to_style(
        self,
        paper_content: str,
        template_content: str,
    ) -> str:
        """
        Reformat ``paper_content`` to follow the style of ``template_content``.

        Returns the complete, compilable reformatted LaTeX string.
        """
        user_prompt = FORMAT_PAPER_STYLE_USER_TEMPLATE.format(
            template_content=template_content,
            paper_content=paper_content,
        )

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=FORMAT_PAPER_STYLE_PROMPT),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ]

        logger.info("Sending paper formatting request to LLM")
        response = await self.llm.achat(messages)
        formatted_content = response.message.content.strip()

        # Strip markdown code fences if the LLM wraps the output despite instructions
        lines = formatted_content.split("\n")
        if (
            len(lines) >= 2
            and lines[0].startswith("```")
            and lines[-1].strip() == "```"
        ):
            # Drop the opening fence line (```latex or ```) and the closing ``` line
            formatted_content = "\n".join(lines[1:-1])

        logger.info("Paper formatting completed successfully")
        return formatted_content

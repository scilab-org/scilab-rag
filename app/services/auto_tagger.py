import asyncio
import json
import logging
from typing import Any, Callable, List, Optional, Union

import nest_asyncio
nest_asyncio.apply()

from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import TransformComponent, BaseNode, MetadataMode
from llama_index.core.llms.llm import LLM
from llama_index.core import Settings
from llama_index.core.async_utils import run_jobs
from llama_index.core.prompts import PromptTemplate
from app.core.prompts import CHUNK_SUMMARY_PROMPT, GLOBAL_SUMMARY_PROMPT, TAG_FROM_SUMMARY_PROMPT

logger = logging.getLogger(__name__)


class AutoTagger(TransformComponent):
    llm: LLM
    num_workers: int
    existing_tags: List[str]

    def __init__(
        self,
        llm: Optional[LLM] = None,
        num_workers: int = 2,
        existing_tags: Optional[List[str]] = None,
    ) -> None:

        super().__init__(
            llm=llm or Settings.llm,
            num_workers=num_workers,
            existing_tags=existing_tags or [],
        )

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            return loop.run_until_complete(self.acall(nodes, show_progress))
        else:
            return asyncio.run(self.acall(nodes, show_progress))

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """
        Hierarchical tagging:
        1. Summarize each big chunk
        2. Merge summaries → global summary
        3. Generate 5-10 tags from global summary
        4. Remove duplicates and normalize
        """
        if not nodes:
            logger.warning("No nodes provided for tagging")
            return nodes

        try:
            # Step 1: Summarize each chunk in parallel
            summary_jobs = [self._summarize_chunk(node) for node in nodes]
            chunk_summaries = await run_jobs(
                summary_jobs, show_progress, 
                desc="Summarizing chunks",
                workers=self.num_workers,
            )

            # Step 2: Create global summary
            global_summary = await self._create_global_summary(chunk_summaries)
            logger.info(f"Generated global summary: {global_summary[:100]}...")

            # Step 3: Generate final tags
            tags = await self._generate_tags(global_summary)
            logger.info(f"Generated {len(tags)} tags before deduplication")

            # Step 4: Remove duplicates
            tags = self._remove_duplicate_tags(tags)
            logger.info(f"Final tag count after deduplication: {len(tags)}")

            # Attach to first node (document-level tags)
            if nodes:
                nodes[0].metadata["summary"] = global_summary
                nodes[0].metadata["tags"] = tags

            return nodes
        
        except Exception as e:
            logger.error(f"Error during tagging: {str(e)}", exc_info=True)
            # Return nodes with empty tags on error
            if nodes:
                nodes[0].metadata["summary"] = ""
                nodes[0].metadata["tags"] = []
            return nodes
        
    async def _summarize_chunk(self, node: BaseNode) -> str:
        """Summarize a single chunk of text."""
        text = node.get_content(metadata_mode=MetadataMode.LLM)
        prompt = PromptTemplate(CHUNK_SUMMARY_PROMPT)

        try:
            response = await self.llm.apredict(
                prompt,
                context=text,
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error summarizing chunk: {str(e)}")
            return ""
   
    async def _create_global_summary(self, chunk_summaries: List[str]) -> str:
        """Create a global summary from chunk summaries."""
        # Filter out empty summaries
        valid_summaries = [s for s in chunk_summaries if s.strip()]
        
        if not valid_summaries:
            logger.warning("No valid chunk summaries found")
            return ""
        
        combined = "\n\n".join(valid_summaries)

        try:
            prompt = PromptTemplate(GLOBAL_SUMMARY_PROMPT)
            response = await self.llm.apredict(
                prompt,
                context=combined,
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error creating global summary: {str(e)}")
            return ""
    
    async def _generate_tags(self, summary: str) -> List[str]:
        """Generate tags from the global summary."""
        if not summary.strip():
            logger.warning("Empty summary provided for tag generation")
            return []
        
        # Format existing tags as a readable list
        existing_tags_str = (
            ", ".join(self.existing_tags)
            if self.existing_tags
            else "None"
        )        
        
        try:
            prompt = PromptTemplate(TAG_FROM_SUMMARY_PROMPT)
            response = await self.llm.apredict(
                prompt,
                context=summary,
                existing_tags=existing_tags_str,
            )

            return self._default_parse_tags(response)
        except Exception as e:
            logger.error(f"Error generating tags: {str(e)}")
            return []
    
    @staticmethod
    def _default_parse_tags(response: str) -> List[str]:
        """
        Parse tags from LLM JSON response.
        Expected format: {"tags": ["Tag One", "Tag Two"]}
        Fallback to comma-separated parsing if JSON fails.
        """
        if not response or not response.strip():
            return []
        
        # Try JSON parsing first (expected format from TAG_FROM_SUMMARY_PROMPT)
        try:
            # Remove markdown code fences if present
            clean_response = response.strip()
            if clean_response.startswith("```"):
                lines = clean_response.split("\n")
                clean_response = "\n".join(lines[1:-1]) if len(lines) > 2 else clean_response
                clean_response = clean_response.replace("```json", "").replace("```", "")
            
            data = json.loads(clean_response.strip())
            
            if isinstance(data, dict) and "tags" in data:
                tags = data["tags"]
                if isinstance(tags, list):
                    # Filter out empty or invalid tags
                    return [str(tag).strip() for tag in tags if tag and str(tag).strip()]
            
            logger.warning(f"Unexpected JSON structure: {data}")
            return []
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return []
        
    def _remove_duplicate_tags(self, tags: List[str]) -> List[str]:
        """
        Remove duplicate tags using case-insensitive comparison and normalization.
        Also checks against existing_tags to avoid duplicates.
        """
        if not tags:
            return []
        
        # Normalize tags: strip, title case for comparison
        seen = set()
        unique_tags = []
        
        # Add existing tags to seen set (normalized)
        for tag in self.existing_tags:
            normalized = tag.strip().lower()
            seen.add(normalized)
        
        for tag in tags:
            tag = tag.strip()
            if not tag:
                continue
            
            normalized = tag.lower()
            
            # Check for exact duplicates and semantic similarity
            if normalized not in seen:
                # Also check for plural vs singular (basic check)
                if not self._is_semantic_duplicate(normalized, seen):
                    seen.add(normalized)
                    unique_tags.append(tag)
                else:
                    logger.debug(f"Skipping semantic duplicate: {tag}")
        
        return unique_tags
    
    @staticmethod
    def _is_semantic_duplicate(tag: str, seen_tags: set) -> bool:
        """
        Check if a tag is a semantic duplicate of existing tags.
        Handles plural/singular forms and minor variations.
        """
        # Check for plural variations
        if tag.endswith('s') and tag[:-1] in seen_tags:
            return True
        if tag + 's' in seen_tags:
            return True
        
        # Check for minor variations (e.g., hyphen vs space)
        tag_variants = [
            tag.replace('-', ' '),
            tag.replace(' ', '-'),
            tag.replace('_', ' '),
            tag.replace(' ', '_'),
        ]
        
        return any(variant in seen_tags for variant in tag_variants)
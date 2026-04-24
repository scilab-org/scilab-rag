import asyncio
import concurrent.futures
import json
import logging
import re
from typing import Any, List, Optional

from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import TransformComponent, BaseNode, MetadataMode
from llama_index.core.llms.llm import LLM
from llama_index.core import Settings
from llama_index.core.async_utils import run_jobs
from llama_index.core.prompts import PromptTemplate
from app.agents.tagger.prompts import CHUNK_SUMMARY_PROMPT, GLOBAL_SUMMARY_PROMPT, TAG_FROM_SUMMARY_PROMPT

logger = logging.getLogger(__name__)


class AutoTagger(TransformComponent):
    llm: LLM
    num_workers: int
    existing_tags: List[str]

    def __init__(
        self,
        llm: Optional[LLM] = None,
        num_workers: int = 6,
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
            # Running inside an event loop (e.g., FastAPI with uvloop)
            # Use a thread pool to run async code in a separate thread
            def run_in_new_loop():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self.acall(nodes, show_progress))
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_new_loop)
                return future.result()
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
            official_keywords_line = self._extract_keywords_line(nodes)

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
            tags = await self._generate_tags(global_summary, official_keywords_line)
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
    
    async def _generate_tags(self, summary: str, official_keywords_line: Optional[str]) -> List[dict]:          
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
                official_keywords=official_keywords_line or "None",
            )

            return self._default_parse_tags(response)
        except Exception as e:
            logger.error(f"Error generating tags: {str(e)}")
            return []
    
    @staticmethod
    def _default_parse_tags(response: str) -> List[dict]:
        """
        Parse tags from LLM JSON response.
        Expected format: {"tags": [{"name": "Tag One", "isFromPaper": true}, ...]}
        Fallback to old format: {"tags": ["Tag One", "Tag Two"]} for backward compatibility.
        """
        if not response or not response.strip():
            return []
        
        # Try JSON parsing first (expected format from TAG_FROM_SUMMARY_PROMPT)
        try:
            clean_response = response.strip()
            
            # Remove BOM if present
            if clean_response.startswith('\ufeff'):
                clean_response = clean_response[1:]
            
            # Remove markdown code fences if present
            if clean_response.startswith("```"):
                lines = clean_response.split("\n")
                # Remove first line (```json or ```) and last line (```)
                if len(lines) > 2:
                    clean_response = "\n".join(lines[1:-1])
                else:
                    clean_response = "\n".join(lines)
                # Strip remaining fence markers
                clean_response = clean_response.replace("```json", "").replace("```", "")
            
            clean_response = clean_response.strip()
            
            # Find the first { and last } to extract JSON
            json_start = clean_response.find('{')
            json_end = clean_response.rfind('}')
            
            if json_start == -1 or json_end == -1 or json_start >= json_end:
                logger.warning(f"Could not find valid JSON delimiters in response: {response[:100]}")
                return []
            
            clean_response = clean_response[json_start:json_end+1]
            
            data = json.loads(clean_response)
            
            if isinstance(data, dict) and "tags" in data:
                tags = data["tags"]
                if isinstance(tags, list):
                    result = []
                    for tag in tags:
                        if not tag:
                            continue
                        
                        # New format: object with name and isFromPaper
                        if isinstance(tag, dict):
                            name = tag.get("name", "").strip()
                            is_from_paper = tag.get("isFromPaper", False)
                            if name:
                                result.append({
                                    "name": name,
                                    "isFromPaper": bool(is_from_paper)
                                })
                        # Old format: string (backward compatibility)
                        elif isinstance(tag, str):
                            name = tag.strip()
                            if name:
                                result.append({
                                    "name": name,
                                    "isFromPaper": False
                                })
                    
                    return result
            
            logger.warning(f"Unexpected JSON structure: {data}")
            return []
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}. Original response: {response[:200]}")
            return []
        
    def _remove_duplicate_tags(self, tags: List[dict]) -> List[dict]:
        """
        Remove duplicate tags using case-insensitive comparison and normalization.
        Also checks against existing_tags to avoid duplicates.
        Tags are dictionaries with "name" and "isFromPaper" keys.
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
        
        for tag_obj in tags:
            if not isinstance(tag_obj, dict) or "name" not in tag_obj:
                continue
                
            tag_name = tag_obj["name"].strip()
            if not tag_name:
                continue
            
            normalized = tag_name.lower()
            
            # Check for exact duplicates and semantic similarity
            if normalized not in seen:
                # Also check for plural vs singular (basic check)
                if not self._is_semantic_duplicate(normalized, seen):
                    seen.add(normalized)
                    unique_tags.append({
                        "name": tag_name,
                        "isFromPaper": tag_obj.get("isFromPaper", False)
                    })
                else:
                    logger.debug(f"Skipping semantic duplicate: {tag_name}")
        
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
    
    @staticmethod
    def _extract_keywords_line(nodes: List[BaseNode]) -> Optional[str]:
        """
        Extract the official Keywords/Index Terms line directly from raw text.
        Matches either:
          - "Keywords: ..." (colon form, any content after)
          - "Keywords foo, bar, baz" (no colon, but must be comma-separated list)
        Returns the line as-is, or None if not found.
        """
        # Colon form: "Keywords: anything"
        colon_pattern = re.compile(
            r'^\s*(keywords|index\s+terms)\s*:\s*(.+)$',
            re.IGNORECASE | re.MULTILINE,
        )
        # No-colon form: "Keywords foo, bar" — requires at least one comma to
        # distinguish from prose sentences that start with "Keywords ..."
        no_colon_pattern = re.compile(
            r'^\s*(keywords|index\s+terms)[ \t]+([^,\n]+(?:,[ \t]*[^,\n]+)+)$',
            re.IGNORECASE | re.MULTILINE,
        )
        for node in nodes:
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            for pattern in (colon_pattern, no_colon_pattern):
                match = pattern.search(text)
                if match:
                    label = match.group(1).strip()
                    terms = match.group(2).strip().rstrip('.')
                    return f"{label.title()}: {terms}"
        return None
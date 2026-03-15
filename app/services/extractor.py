import asyncio
import concurrent.futures
import logging
from typing import Any, Callable, Optional, Union, List

import tiktoken

from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.llms.llm import LLM

from llama_index.core.indices.property_graph.utils import default_parse_triplets_fn
from llama_index.core import Settings
from llama_index.core.async_utils import run_jobs

from llama_index.core.prompts.default_prompts import (
    DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
)

from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)

from app.domain.models import PaperInfo
from app.core.config import settings
from app.helpers.utils import normalize_entity_key

logger = logging.getLogger(__name__)

class GraphRAGExtractor(TransformComponent):
    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_path_per_chunks: int
    paper_info: PaperInfo  
        
    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = default_parse_triplets_fn,
        num_workers: int = 2,
        max_path_per_chunks: int = settings.MAX_TRIPLETS_PER_CHUNK,
        paper_info: Optional[PaperInfo] = None,
    ) -> None:
        if paper_info is None:
            raise ValueError("paper_info is required for GraphRAGExtractor")

        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_path_per_chunks=max_path_per_chunks,
            paper_info=paper_info,
        )
        

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """
        Sync entrypoint required by LlamaIndex.
        Wraps async implementation safely.
        """
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
        self, nodes: List[BaseNode], show_progress=False, **kwargs: Any
    ) -> List[BaseNode]:
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(jobs, show_progress, desc="Extracting paths from text")

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """
        ENTITY
        Name  Label Description
        (Apple, Company, A technology company)

        ENTITY_RELATIONSHIPS
        Sub, OBJ, Rel, Description
        (Apple, California, headquartered_in, Apple is based in California)

        """
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")

        # ── Dynamic max_triplets based on chunk token count ───────────
        # Formula: base=5 + 5 per 1000 tokens, capped by MAX_TRIPLETS_PER_CHUNK
        enc = tiktoken.encoding_for_model("gpt-4o-mini")  # cached internally
        token_count = len(enc.encode(text))
        max_triplets = min(5 + (token_count // 1000) * 5, self.max_path_per_chunks)
        logger.debug(
            "Chunk has %d tokens → max_triplets=%d (ceiling=%d)",
            token_count, max_triplets, self.max_path_per_chunks,
        )

        # Build section headings string from chunk metadata
        headings = node.metadata.get("headings") or []
        if headings:
            section_headings = " > ".join(headings)
        else:
            section_headings = "(no section heading available)"

        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=max_triplets,
                paper_title=self.paper_info.paper_name,
                section_headings=section_headings,
            )
            entities, entities_relationship = self.parse_fn(llm_response)

        except ValueError as exc:
            logger.warning("parse_fn failed for chunk: %s", exc)
            entities = []
            entities_relationship = []

        logger.debug(
            "Extracted %d entities and %d relationships", 
            len(entities),
            len(entities_relationship),
        )
                
        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        entity_metadata = node.metadata.copy()
        entity_metadata["paper_id"] = self.paper_info.paper_id
        entity_metadata["paper_name"] = self.paper_info.paper_name

        for entity, entity_type, description in entities:
            entity_key = entity
            entity_uid = f"{self.paper_info.paper_id}::{entity_key}"
            
            if not entity_type:
                logger.warning("Missing entity_type for entity=%r", entity)
            
            new_entity = EntityNode(
                name=entity_uid,
                label=entity_type,
                properties={
                    **entity_metadata,
                    "entity_name": entity,  
                    "entity_description": description,
                    "entity_key": entity_key,
                    "entity_key_normalized": normalize_entity_key(entity_key),
                }
            )
            existing_nodes.append(new_entity)

        relation_metadata = node.metadata.copy()
        relation_metadata["paper_id"] = self.paper_info.paper_id
        for sub, obj, rel, description in entities_relationship:

            source_id = f"{self.paper_info.paper_id}::{sub}"
            target_id = f"{self.paper_info.paper_id}::{obj}"

            relation = Relation(
                source_id=source_id,
                target_id=target_id,
                label=rel,
                properties=  {
                    **relation_metadata,
                    "relation_description": description,
                },
            )

            existing_relations.append(relation)

        node.metadata["paper_id"] = self.paper_info.paper_id        
        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        
        return node

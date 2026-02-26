import asyncio
import logging
from typing import Any, Callable, Optional, Union, List

import nest_asyncio

nest_asyncio.apply()

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


logger = logging.getLogger(__name__)


class GraphRAGExtractor(TransformComponent):
    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_path_per_chunks: int

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = default_parse_triplets_fn,
        num_workers: int = 2,
        max_path_per_chunks: int = 10,
    ) -> None:

        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_path_per_chunks=max_path_per_chunks,
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
            # FastAPI / async context
            return loop.run_until_complete(self.acall(nodes, show_progress))
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

        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_path_per_chunks,
            )
            entities, entities_relationship = self.parse_fn(llm_response)

        except ValueError:
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

        for entity, entity_type, description in entities:
            if not entity_type:
                logger.warning("Missing entity_type for entity=%r", entity)
            entity_metadata["entity_description"] = description
            new_entity = EntityNode(
                name=entity, label=entity_type, properties=entity_metadata
            )
            existing_nodes.append(new_entity)

        relation_metadata = node.metadata.copy()

        for triplet in entities_relationship:
            sub, obj, rel, description = triplet
            relation_metadata["relation_description"] = description
            relation = Relation(
                source_id=sub, target_id=obj, label=rel, properties=relation_metadata
            )
            existing_relations.append(relation)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

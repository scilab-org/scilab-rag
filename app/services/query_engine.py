"""GraphRAG Query Engine for answering questions using community-based retrieval."""

import logging
import re
from typing import List

from llama_index.core import PropertyGraphIndex
from llama_index.core.llms import ChatMessage, LLM

from app.core.prompts import QUERY_ANSWER_PROMPT, AGGREGATE_ANSWERS_PROMPT
from app.services.store import GraphRAGStore


logger = logging.getLogger(__name__)


class GraphRAGQueryEngine:
    """
    Query engine that uses Graph RAG for answering questions.

    Process:
    1. Retrieve entities similar to the query
    2. Find communities those entities belong to
    3. Generate answers from each community's summary
    4. Aggregate answers into a final response
    """

    def __init__(
        self,
        graph_store: GraphRAGStore,
        index: PropertyGraphIndex,
        llm: LLM,
        similarity_top_k: int = 20,
    ):
        self.graph_store = graph_store
        self.index = index
        self.llm = llm
        self.similarity_top_k = similarity_top_k

    async def acustom_query(self, query_str: str) -> str:
        """Process query using community-based retrieval."""
        logger.info("Starting acustom_query")
        
        # Get similar entities
        logger.debug("Step 1: Getting entities")
        entities = await self.get_entities(query_str, self.similarity_top_k)
        logger.debug("Found %d entities (sample: %s)", len(entities), entities[:5])

        # Get community IDs for those entities
        logger.debug("Step 2: Retrieving entity communities")
        community_ids = await self.retrieve_entity_communities(
            self.graph_store.entity_info, entities
        )
        logger.debug("Found %d community IDs", len(community_ids))
        
        logger.debug("Step 3: Getting community summaries")
        community_summaries = await self.graph_store.get_community_summaries()
        logger.debug("Total community summaries: %d", len(community_summaries))

        # Get community summaries
        community_answers = []

        logger.debug("Step 4: Generating answers from communities")
        for id, summary in community_summaries.items():
            if id in community_ids:
                answer = await self.generate_answer_from_summary(
                    summary, query_str
                )
                community_answers.append(answer)
        
        logger.debug("Generated %d community answers", len(community_answers))

        # Aggregate into final answer
        if not community_answers:
            logger.info("No community answers found for query")
            return "I don't have enough information to answer this question."

        logger.debug("Step 5: Aggregating final answer")
        final_answer = await self.aggregate_answers(community_answers)
        logger.debug("Final answer length: %d", len(final_answer))
        
        return final_answer

    async def get_entities(self, query_str: str, similarity_top_k: int) -> List[str]:
        """Retrieve entities similar to the query."""
        retriever = self.index.as_retriever(
            similarity_top_k=similarity_top_k
        )

        nodes_retrieved = await retriever.aretrieve(query_str) 

        entities = set()

        # Pattern: entity -> rel -> entity
        pattern = r"^(\w+(?:\s+\w+)*)\s*->\s*([a-zA-Z\s]+?)\s*->\s*(\w+(?:\s+\w+)*)$"

        for node in nodes_retrieved:
            matches = re.findall(pattern, node.text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                subject = match[0]
                obj = match[2]
                entities.add(subject)
                entities.add(obj)

        return list(entities)

    async def retrieve_entity_communities(
        self, entity_info: dict, entities: List[str]
    ) -> List[int]:
        """Get community IDs for a list of entities."""
        community_ids = []

        for entity in entities:
            if entity in entity_info:
                community_ids.extend(entity_info[entity])

        return list(set(community_ids))

    async def generate_answer_from_summary(
        self, community_summary: str, query: str
    ) -> str:
        """Generate an answer from a community summary."""
        prompt = QUERY_ANSWER_PROMPT.format(query=query)

        messages = [
            ChatMessage(role="system", content=community_summary),
            ChatMessage(role="user", content=prompt),
        ]

        response = await self.llm.achat(messages)
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return cleaned_response

    async def aggregate_answers(self, community_answers: List[str]) -> str:
        """Aggregate individual answers into a final response."""
        messages = [
            ChatMessage(role="system", content=AGGREGATE_ANSWERS_PROMPT),
            ChatMessage(
                role="user",
                content=f"Intermediate answers: {community_answers}",
            ),
        ]

        final_response = await self.llm.achat(messages)
        cleaned_final_response = re.sub(
            r"^assistant:\s*", "", str(final_response)
        ).strip()

        return cleaned_final_response
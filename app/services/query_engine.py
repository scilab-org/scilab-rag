"""GraphRAG Query Engine — scoped 2-hop retrieval + chunk text + single LLM call."""

import logging
import re
from typing import Dict, List

from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.embeddings import BaseEmbedding

from app.core.prompts import GRAPH_QA_SYSTEM_PROMPT, GRAPH_QA_USER_PROMPT
from app.services.store import GraphRAGStore
from app.core.config import settings

logger = logging.getLogger(__name__)


class GraphRAGQueryEngine:
    """
    Query engine that retrieves paper-scoped context from the knowledge
    graph and answers in a single LLM call.

    Process:
    1. Embed the user query
    2. Vector search scoped to paper_ids + 2-hop neighbor traversal
    3. Retrieve original chunk text via MENTIONS relationships
    4. Format everything as natural "paper notes"
    5. Single LLM call (system: HyperDataLab Assistant, user: context + question)
    """

    def __init__(
        self,
        graph_store: GraphRAGStore,
        embed_model: BaseEmbedding,
        llm: LLM,
        similarity_top_k: int = settings.SIMILARITY_TOP_K,
    ):
        self.graph_store = graph_store
        self.embed_model = embed_model
        self.llm = llm
        self.similarity_top_k = similarity_top_k

    async def acustom_query(self, query_str: str, paper_ids: List[str]) -> str:
        """Answer a user question scoped to the given paper_ids."""
        logger.info("Starting query: %s", query_str[:80])

        # Step 1: Embed the query
        query_embedding = await self.embed_model.aget_query_embedding(query_str)

        # Step 2 & 3: Retrieve graph context + chunk text (hybrid)
        context = self.graph_store.retrieve_scoped_context(
            query_embedding=query_embedding,
            paper_ids=paper_ids,
            top_k=self.similarity_top_k,
        )

        graph_records = context.get("graph", [])
        chunk_records = context.get("chunks", [])

        # if not graph_records and not chunk_records:
        #     logger.info("No context retrieved for query")
        #     return (
        #         "I couldn't find relevant information in your papers for this question. "
        #         "Could you try rephrasing, or make sure the relevant papers have been uploaded?"
        #     )

        # Step 4: Format as natural paper notes
        context_text = self._format_context(graph_records, chunk_records)
        logger.debug("Formatted context length: %d chars", len(context_text))

        # Step 5: Single LLM call
        user_message = GRAPH_QA_USER_PROMPT.format(
            context=context_text,
            question=query_str,
        )

        messages = [
            ChatMessage(role="system", content=GRAPH_QA_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_message),
        ]

        response = await self.llm.achat(messages)
        answer = re.sub(r"^assistant:\s*", "", str(response)).strip()
        logger.debug("Answer length: %d chars", len(answer))

        return answer

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _format_context(
        graph_records: List[Dict],
        chunk_records: List[Dict],
    ) -> str:
        """Format retrieved graph records + chunk text as natural-language
        paper notes. No graph syntax (arrows, entity types, etc.) is exposed.

        Structure:
        1. "Key findings from papers" — structured facts from graph
        2. "Relevant excerpts" — original paper text from chunks
        """
        sections = []

        # ── Part 1: Graph-derived facts ──────────────────────────────────────
        if graph_records:
            graph_text = GraphRAGQueryEngine._format_graph_records(graph_records)
            if graph_text:
                sections.append("### Key findings from papers\n" + graph_text)

        # ── Part 2: Original chunk text ──────────────────────────────────────
        if chunk_records:
            chunk_text = GraphRAGQueryEngine._format_chunks(chunk_records)
            if chunk_text:
                sections.append("### Relevant excerpts from papers\n" + chunk_text)

        return "\n\n".join(sections) if sections else "(No relevant notes found.)"

    @staticmethod
    def _format_graph_records(records: List[Dict]) -> str:
        """Format graph records as grouped natural-language notes.
        Filters out SAME_AS (structural link, not a user-facing fact).
        """
        seen = set()
        grouped: dict[str, list[str]] = {}

        for r in records:
            source = r.get("source_name") or ""
            target = r.get("target_name") or ""
            relation = r.get("relation") or ""
            rel_desc = r.get("relation_description") or ""
            src_desc = r.get("source_description") or ""
            src_paper = r.get("source_paper_id") or ""

            if not source:
                continue

            # Skip SAME_AS — it's a structural link, not a fact
            if relation == "SAME_AS":
                continue

            # Build a natural note for this fact
            parts = []

            # Always include source description (it's the key info about this entity)
            if src_desc:
                parts.append(src_desc)

            # Add the relationship as a readable sentence
            if target and relation:
                readable_rel = relation.replace("_", " ").lower()
                if rel_desc:
                    parts.append(rel_desc)
                else:
                    parts.append(f"{source} {readable_rel} {target}")

            if not parts:
                continue

            # Deduplicate
            dedup_key = (source, target, relation)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            note = ". ".join(parts)
            group_key = f"{source} (paper {src_paper})" if src_paper else source
            grouped.setdefault(group_key, []).append(note)

        lines = []
        for group, notes in grouped.items():
            lines.append(f"- {group}:")
            for note in notes:
                lines.append(f"  \u2022 {note}")

        return "\n".join(lines)

    @staticmethod
    def _format_chunks(chunks: List[Dict]) -> str:
        """Format original paper text chunks."""
        lines = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            paper_id = chunk.get("paper_id", "")
            header = f"[Excerpt {i}, paper {paper_id}]" if paper_id else f"[Excerpt {i}]"
            lines.append(f"{header}\n{text}")

        return "\n\n".join(lines)

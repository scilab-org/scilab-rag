"""GraphRAG Query Engine — scoped 2-hop retrieval + chunk text + single LLM call."""

import logging
import re
from typing import AsyncGenerator, Dict, List, Tuple

from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.embeddings import BaseEmbedding

from app.agents.chat.prompts import GRAPH_QA_SYSTEM_PROMPT, GRAPH_QA_USER_PROMPT
from app.domain.models import ChatQuery
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
    4. Resolve paper_id → paper_name for human-readable attribution
    5. Format everything as natural "paper notes"
    6. Single LLM call (system: HyperDataLab Assistant, user: context + question)
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

    async def _build_messages(
        self, chat_query: ChatQuery,
    ) -> Tuple[List[ChatMessage], Dict[str, str]]:
        """Shared: embed query, retrieve context, resolve paper names,
        build LLM message list.

        Returns:
            (messages, paper_names) where paper_names is {paper_id: paper_name}.
        """
        logger.info("Starting query: %s", chat_query.query_str[:80])

        # Step 1: Embed the query
        query_embedding = await self.embed_model.aget_query_embedding(chat_query.query_str)

        # Step 2 & 3: Retrieve graph context + chunk text (hybrid)
        context = self.graph_store.retrieve_scoped_context(
            query_embedding=query_embedding,
            paper_ids=chat_query.paper_ids,
            top_k=self.similarity_top_k,
        )

        graph_records = context.get("graph", [])
        chunk_records = context.get("chunks", [])

        # Step 4: Resolve paper_id → paper_name
        paper_names = self.graph_store.resolve_paper_names(chat_query.paper_ids)

        # Step 5: Format as natural paper notes
        context_text = self._format_context(graph_records, chunk_records, paper_names)
        logger.debug("Formatted context length: %d chars", len(context_text))

        # Build papers-in-scope section
        if paper_names:
            papers_in_scope = "\n".join(
                f"- {name}" for name in paper_names.values()
            )
        else:
            papers_in_scope = "(No papers resolved for this conversation.)"

        # Step 6: Build message list
        messages: List[ChatMessage] = []

        messages.append(
            ChatMessage(role="system", content=GRAPH_QA_SYSTEM_PROMPT)
        )

        if chat_query.summary:
            messages.append(
                ChatMessage(
                    role="system",
                    content=f"## Summary of earlier conversation\n{chat_query.summary}",
                )
            )

        for msg in chat_query.history:
            messages.append(ChatMessage(role=msg.role, content=msg.content))

        user_message = GRAPH_QA_USER_PROMPT.format(
            papers_in_scope=papers_in_scope,
            context=context_text,
            question=chat_query.query_str,
        )
        messages.append(ChatMessage(role="user", content=user_message))

        return messages, paper_names

    async def acustom_query(
        self, chat_query: ChatQuery,
    ) -> Tuple[str, Dict[str, str]]:
        """Answer a user question scoped to the given paper_ids.

        Returns:
            (answer, paper_names) where paper_names is {paper_id: paper_name}.
        """
        messages, paper_names = await self._build_messages(chat_query)

        response = await self.llm.achat(messages)
        answer = re.sub(r"^assistant:\s*", "", str(response)).strip()
        logger.debug("Answer length: %d chars", len(answer))

        return answer, paper_names

    async def astream_query(
        self, chat_query: ChatQuery,
    ) -> Tuple[AsyncGenerator[str, None], Dict[str, str]]:
        """Stream answer tokens for a user question scoped to the given paper_ids.

        Returns:
            (token_generator, paper_names) — the generator yields token delta
            strings as they arrive from the LLM.
        """
        messages, paper_names = await self._build_messages(chat_query)

        async def _generate() -> AsyncGenerator[str, None]:
            response = await self.llm.astream_chat(messages)
            async for delta in response:
                token = delta.delta or ""
                if token:
                    yield token

        return _generate(), paper_names

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _format_context(
        graph_records: List[Dict],
        chunk_records: List[Dict],
        paper_names: Dict[str, str],
    ) -> str:
        """Format retrieved graph records + chunk text as natural-language
        paper notes. No graph syntax (arrows, entity types, etc.) is exposed.
        Structure:
        1. "Key findings from papers" — structured facts from graph
        """
        sections = []

        # ── Part 1: Graph-derived facts ──────────────────────────────────────
        if graph_records:
            graph_text = GraphRAGQueryEngine._format_graph_records(
                graph_records, paper_names,
            )
            if graph_text:
                sections.append("### Key findings from papers\n" + graph_text)

        # ── Part 2: Original chunk text ──────────────────────────────────────
        if chunk_records:
            chunk_text = GraphRAGQueryEngine._format_chunks(
                chunk_records, paper_names,
            )
            if chunk_text:
                sections.append("### Relevant original text from papers\n" + chunk_text)

        return "\n\n".join(sections) if sections else "(No relevant notes found.)"

    @staticmethod
    def _resolve_paper_label(
        paper_id: str,
        paper_name_from_record: str,
        paper_names: Dict[str, str],
    ) -> str:
        """Return a human-readable paper label. Never returns a UUID.
        Returns empty string if the paper name cannot be resolved.
        """
        return (
            paper_name_from_record
            or paper_names.get(paper_id, "")
        )

    @staticmethod
    def _build_attribution_suffix(
        authors: str,
        publication_month_year: str,
    ) -> str:
        """Build a short '(Author et al., Year)' attribution string.

        Only included when at least one of authors/year is available.
        Never exposes cite_key or paper_id.
        """
        # Extract year from e.g. "May 2015" or "2015"
        year = ""
        if publication_month_year:
            parts = publication_month_year.strip().split()
            for part in reversed(parts):
                if part.isdigit() and len(part) == 4:
                    year = part
                    break

        # Build short author string
        author_short = ""
        if authors:
            # Split on semicolon or comma to get individual author names
            author_list = [a.strip() for a in authors.replace(";", ",").split(",") if a.strip()]
            if author_list:
                # Use first author's last name (last token before any comma within a name)
                first_author = author_list[0].strip()
                # Handle "Last, First" format
                name_parts = first_author.split()
                last_name = name_parts[0] if name_parts else first_author
                if len(author_list) > 1:
                    author_short = f"{last_name} et al."
                else:
                    author_short = last_name

        if author_short and year:
            return f"{author_short}, {year}"
        if author_short:
            return author_short
        if year:
            return year
        return ""

    @staticmethod
    def _format_graph_records(
        records: List[Dict],
        paper_names: Dict[str, str],
    ) -> str:
        """Format graph records as grouped natural-language notes."""
        seen = set()
        grouped: dict[str, list[str]] = {}

        for r in records:
            source = r.get("source_name") or ""
            target = r.get("target_name") or ""
            relation = r.get("relation") or ""
            rel_desc = r.get("relation_description") or ""
            src_desc = r.get("source_description") or ""
            src_paper_id = r.get("source_paper_id") or ""
            src_paper_name = r.get("source_paper_name") or ""
            src_authors = r.get("source_authors") or ""
            src_pub_year = r.get("source_publication_month_year") or ""

            if not source:
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
            paper_label = GraphRAGQueryEngine._resolve_paper_label(
                src_paper_id, src_paper_name, paper_names,
            )
            # Enrich label with attribution suffix (authors + year)
            attribution = GraphRAGQueryEngine._build_attribution_suffix(
                src_authors, src_pub_year,
            )
            if paper_label and attribution:
                group_key = f"{source} ({paper_label} — {attribution})"
            elif paper_label:
                group_key = f"{source} ({paper_label})"
            else:
                group_key = source
            grouped.setdefault(group_key, []).append(note)

        lines = []
        for group, notes in grouped.items():
            lines.append(f"- {group}:")
            for note in notes:
                lines.append(f"  \u2022 {note}")

        return "\n".join(lines)

    @staticmethod
    def _format_chunks(
        chunks: List[Dict],
        paper_names: Dict[str, str],
    ) -> str:
        """Format original paper text chunks.
        Uses paper name + authors + year for attribution; never exposes UUIDs or cite_keys.
        """
        lines = []
        for chunk in chunks:
            text = chunk.get("text", "")
            paper_id = chunk.get("paper_id", "")
            chunk_paper_name = chunk.get("paper_name", "")
            authors = chunk.get("authors", "")
            pub_year = chunk.get("publication_month_year", "")

            paper_label = GraphRAGQueryEngine._resolve_paper_label(
                paper_id, chunk_paper_name, paper_names,
            )
            attribution = GraphRAGQueryEngine._build_attribution_suffix(authors, pub_year)
            if paper_label and attribution:
                header = f"[{paper_label} — {attribution}]"
            elif paper_label:
                header = f"[{paper_label}]"
            else:
                header = ""

            formatted_line = f"{header}\n{text}" if header else text
            lines.append(formatted_line)

        return "\n\n".join(lines)

"""GraphRAG Store with 2-hop scoped retrieval and hybrid context."""

import logging
from typing import Dict, List, Optional

from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

logger = logging.getLogger(__name__)


class GraphRAGStore(Neo4jPropertyGraphStore):
    """
    Extended Neo4j Property Graph Store.

    Features:
    - Shared global entity graph (entities merge by name across papers)
    - 2-hop scoped retrieval filtered by paper_ids
    - Hybrid retrieval: graph context + original chunk text
    """

    def __init__(self, *args, **kwargs):
        # Pop 'llm' if callers still pass it (e.g. dependencies.py)
        # so we don't break the constructor contract during migration.
        kwargs.pop("llm", None)
        super().__init__(*args, **kwargs)
        try:
            self.structured_query(
                "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS "
                "FOR (c:Chunk) ON c.embedding"
            )
        except Exception as e:
            logger.warning("Could not create chunk vector index: %s", e)

    # ── Query: paper name resolution ────────────────────────────────────────

    def resolve_paper_names(self, paper_ids: List[str]) -> Dict[str, str]:
        """Resolve paper_id → paper_name from existing entity nodes.

        Returns a dict like ``{"uuid-1": "My Paper Title", ...}``.
        Papers whose entities lack ``paper_name`` are silently omitted.
        """
        if not paper_ids:
            return {}

        cypher = """
        MATCH (n:__Entity__)
        WHERE n.paper_id IN $paper_ids
          AND n.paper_name IS NOT NULL
          AND n.paper_name <> ''
        RETURN DISTINCT n.paper_id AS paper_id, n.paper_name AS paper_name
        """
        data = self.structured_query(cypher, param_map={"paper_ids": paper_ids})
        if not data:
            return {}

        return {
            row["paper_id"]: row["paper_name"]
            for row in data
            if row.get("paper_id") and row.get("paper_name")
        }

    def resolve_cite_keys(self, paper_ids: List[str]) -> Dict[str, str]:
        """Resolve paper_id → cite_key from existing entity/chunk nodes.

        Returns a dict like ``{"uuid-1": "LeCun2015", ...}``.
        Papers without a cite_key are silently omitted.
        """
        if not paper_ids:
            return {}

        cypher = """
        MATCH (n:__Entity__)
        WHERE n.paper_id IN $paper_ids
          AND n.cite_key IS NOT NULL
          AND n.cite_key <> ''
        RETURN DISTINCT n.paper_id AS paper_id, n.cite_key AS cite_key
        """
        data = self.structured_query(cypher, param_map={"paper_ids": paper_ids})
        if not data:
            return {}

        return {
            row["paper_id"]: row["cite_key"]
            for row in data
            if row.get("paper_id") and row.get("cite_key")
        }

    def resolve_paper_info(self, paper_ids: List[str]) -> Dict[str, Dict]:
        """Resolve full bibliographic info for a list of paper_ids.

        Returns a dict like::

            {
                "uuid-1": {
                    "paper_name": "Deep Learning",
                    "cite_key": "LeCun2015",
                    "authors": "LeCun, Yann; Bengio, Yoshua; Hinton, Geoffrey",
                    "journal_name": "Nature",
                    "publication_month_year": "May 2015",
                    "doi": "10.1038/nature14539",
                },
                ...
            }

        Only fields that are non-null/non-empty are included in each entry.
        Papers with no matching nodes are silently omitted.
        """
        if not paper_ids:
            return {}

        cypher = """
        MATCH (n:__Entity__)
        WHERE n.paper_id IN $paper_ids
          AND n.paper_name IS NOT NULL
          AND n.paper_name <> ''
        RETURN DISTINCT
            n.paper_id               AS paper_id,
            n.paper_name             AS paper_name,
            n.cite_key               AS cite_key,
            n.authors                AS authors,
            n.journal_name           AS journal_name,
            n.publication_month_year AS publication_month_year,
            n.doi                    AS doi
        """
        data = self.structured_query(cypher, param_map={"paper_ids": paper_ids})
        if not data:
            return {}

        result: Dict[str, Dict] = {}
        for row in data:
            pid = row.get("paper_id")
            if not pid:
                continue
            if pid in result:
                continue  # keep first occurrence per paper_id
            entry: Dict[str, str] = {}
            for field in ("paper_name", "cite_key", "authors", "journal_name",
                          "publication_month_year", "doi"):
                val = row.get(field)
                if val and str(val).strip():
                    entry[field] = str(val).strip()
            if entry.get("paper_name"):
                result[pid] = entry
        return result

    # ── Query: scoped 2-hop retrieval + chunk text ───────────────────────────

    def retrieve_scoped_context(
        self,
        query_embedding: List[float],
        paper_ids: List[str],
        top_k: int = 20,
    ) -> Dict[str, list]:
        """Retrieve context for the query pipeline.

        1. Vector similarity search on entity embeddings, scoped to paper_ids
        2. 2-hop neighbor traversal, with ALL neighbors strictly scoped to paper_ids
        3. Collect original chunk text via MENTIONS relationships

        Returns:
            {
                "graph": [  # structured entity/relation records
                    {
                        "source_name", "source_type", "source_description",
                        "source_paper_id", "source_paper_name",
                        "relation", "relation_description",
                        "target_name", "target_type", "target_description",
                        "target_paper_id", "target_paper_name",
                    },
                    ...
                ],
                "chunks": [  # original paper text from source chunks
                    {"text": str, "paper_id": str, "paper_name": str},
                    ...
                ],
            }
        """
        graph_records = self._retrieve_graph_context(
            query_embedding, paper_ids, top_k
        )
        chunk_records = self._retrieve_chunk_text(
            query_embedding, paper_ids, top_k
        )

        return {
            "graph": graph_records,
            "chunks": chunk_records,
        }

    def _retrieve_graph_context(
        self,
        query_embedding: List[float],
        paper_ids: List[str],
        top_k: int,
    ) -> List[Dict]:
        """Vector search → 2-hop traversal, all nodes scoped to paper_ids."""
        # NOTE: The vector index name is 'entity' (set by LlamaIndex).
        # ann_k overscans the ANN index so the paper_id filter doesn't
        # exhaust the budget before any relevant papers appear.
        cypher = """
        // Step 1: Vector search — overscan then filter to paper_ids
        CALL db.index.vector.queryNodes('entity', $ann_k, $embedding)
        YIELD node AS seed, score
        WHERE seed.paper_id IN $paper_ids
        WITH seed, score
        ORDER BY score DESC
        LIMIT $top_k

        // Step 2: 1-hop neighbors (strict project scope)
        OPTIONAL MATCH (seed)-[r1]-(hop1:__Entity__)
        WHERE hop1.paper_id IN $paper_ids

        // Step 3: 2-hop neighbors from hop1 (strict project scope)
        OPTIONAL MATCH (hop1)-[r2]-(hop2:__Entity__)
        WHERE hop2.paper_id IN $paper_ids
          AND hop2 <> seed

        // Collect 1-hop and 2-hop facts
        WITH seed, r1, hop1, r2, hop2
        WITH seed,
             collect(DISTINCT {
                source_name: seed.name,
                source_type: [l IN labels(seed) WHERE NOT l IN ['__Entity__', '__Node__'] | l][0],
                source_description: seed.entity_description,
                source_paper_id: seed.paper_id,
                source_paper_name: seed.paper_name,
                source_cite_key: seed.cite_key,
                source_authors: seed.authors,
                source_publication_month_year: seed.publication_month_year,
                relation: type(r1),
                relation_description: r1.relation_description,
                target_name: hop1.name,
                target_type: [l IN labels(hop1) WHERE NOT l IN ['__Entity__', '__Node__'] | l][0],
                target_description: hop1.entity_description,
                target_paper_id: hop1.paper_id,
                target_paper_name: hop1.paper_name
             }) AS hop1_facts,
             collect(DISTINCT {
                source_name: hop1.name,
                source_type: [l IN labels(hop1) WHERE NOT l IN ['__Entity__', '__Node__'] | l][0],
                source_description: hop1.entity_description,
                source_paper_id: hop1.paper_id,
                source_paper_name: hop1.paper_name,
                source_cite_key: hop1.cite_key,
                source_authors: hop1.authors,
                source_publication_month_year: hop1.publication_month_year,
                relation: type(r2),
                relation_description: r2.relation_description,
                target_name: hop2.name,
                target_type: [l IN labels(hop2) WHERE NOT l IN ['__Entity__', '__Node__'] | l][0],
                target_description: hop2.entity_description,
                target_paper_id: hop2.paper_id,
                target_paper_name: hop2.paper_name
             }) AS hop2_facts

        // Combine and unwind all facts
        WITH hop1_facts + hop2_facts AS all_facts
        UNWIND all_facts AS fact
        WITH fact
        WHERE fact.source_name IS NOT NULL
        RETURN DISTINCT
            fact.source_name                    AS source_name,
            fact.source_type                    AS source_type,
            fact.source_description             AS source_description,
            fact.source_paper_id                AS source_paper_id,
            fact.source_paper_name              AS source_paper_name,
            fact.source_cite_key                AS source_cite_key,
            fact.source_authors                 AS source_authors,
            fact.source_publication_month_year  AS source_publication_month_year,
            fact.relation                       AS relation,
            fact.relation_description           AS relation_description,
            fact.target_name                    AS target_name,
            fact.target_type                    AS target_type,
            fact.target_description             AS target_description,
            fact.target_paper_id                AS target_paper_id,
            fact.target_paper_name              AS target_paper_name
        """
        params = {
            "embedding": query_embedding,
            "paper_ids": paper_ids,
            "top_k": top_k,
            "ann_k": top_k * 10,
        }

        data = self.structured_query(cypher, param_map=params)
        if not data:
            return []

        results = []
        for record in data:
            results.append({
                "source_name": record.get("source_name") or "",
                "source_type": record.get("source_type") or "",
                "source_description": record.get("source_description") or "",
                "source_paper_id": record.get("source_paper_id") or "",
                "source_paper_name": record.get("source_paper_name") or "",
                "source_cite_key": record.get("source_cite_key") or "",
                "source_authors": record.get("source_authors") or "",
                "source_publication_month_year": record.get("source_publication_month_year") or "",
                "relation": record.get("relation") or "",
                "relation_description": record.get("relation_description") or "",
                "target_name": record.get("target_name") or "",
                "target_type": record.get("target_type") or "",
                "target_description": record.get("target_description") or "",
                "target_paper_id": record.get("target_paper_id") or "",
                "target_paper_name": record.get("target_paper_name") or "",
            })

        return results

    def _retrieve_chunk_text(
        self,
        query_embedding: List[float],
        paper_ids: List[str],
        top_k: int,
    ) -> List[Dict]:
        cypher = """
        CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $embedding)
        YIELD node AS chunk, score
        WHERE chunk.paper_id IN $paper_ids
        WITH chunk, score
        ORDER BY score DESC
        LIMIT $top_k
        RETURN DISTINCT
            chunk.text                     AS text,
            chunk.paper_id                 AS paper_id,
            chunk.paper_name               AS paper_name,
            chunk.cite_key                 AS cite_key,
            chunk.authors                  AS authors,
            chunk.journal_name             AS journal_name,
            chunk.publication_month_year   AS publication_month_year,
            chunk.doi                      AS doi
        """
        params = {
            "embedding": query_embedding,
            "paper_ids": paper_ids,
            "top_k": top_k,
        }
        data = self.structured_query(cypher, param_map=params)
        if not data:
            return []

        results = []
        for record in data:
            text = record.get("text") or ""
            if text.strip():
                results.append({
                    "text": text.strip(),
                    "paper_id": record.get("paper_id") or "",
                    "paper_name": record.get("paper_name") or "",
                    "cite_key": record.get("cite_key") or "",
                    "authors": record.get("authors") or "",
                    "journal_name": record.get("journal_name") or "",
                    "publication_month_year": record.get("publication_month_year") or "",
                    "doi": record.get("doi") or "",
                })
        return results

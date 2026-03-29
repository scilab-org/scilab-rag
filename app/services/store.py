"""GraphRAG Store with paper-scoped entities and SAME_AS cross-paper linking."""

import logging
from typing import Dict, List, Optional

from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

logger = logging.getLogger(__name__)


class GraphRAGStore(Neo4jPropertyGraphStore):
    """
    Extended Neo4j Property Graph Store with paper-scoped entity isolation.

    Features:
    - Paper-scoped entities (namespaced IDs prevent cross-paper overwrites)
    - SAME_AS edges link equivalent entities across papers
      based on entity_key_normalized matching
    - 2-hop scoped retrieval with SAME_AS traversal
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

    # ── Ingest: SAME_AS linking ──────────────────────────────────────────────

    def create_same_as_links(self, paper_id: str) -> int:
        """Create SAME_AS edges between entities of `paper_id` and entities
        from OTHER papers that share the same `entity_key_normalized`.

        Returns the number of SAME_AS relationships created.
        """
        cypher = """
        MATCH (a:__Entity__)
        WHERE a.paper_id = $paper_id
          AND a.entity_key_normalized IS NOT NULL
          AND a.entity_key_normalized <> ''
        WITH a
        MATCH (b:__Entity__)
        WHERE b.paper_id <> $paper_id
          AND b.entity_key_normalized = a.entity_key_normalized
          AND NOT (a)-[:SAME_AS]-(b)
        MERGE (a)-[:SAME_AS]->(b)
        RETURN count(*) AS created
        """
        result = self.structured_query(cypher, param_map={"paper_id": paper_id})
        created = result[0]["created"] if result else 0
        logger.info(
            "Created %d SAME_AS links for paper_id=%s", created, paper_id
        )
        return created

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

    # ── Query: scoped 2-hop retrieval + chunk text ───────────────────────────

    def retrieve_scoped_context(
        self,
        query_embedding: List[float],
        paper_ids: List[str],
        top_k: int = 20,
    ) -> Dict[str, list]:
        """Retrieve context for the query pipeline.

        1. Vector similarity search on entity embeddings, scoped to paper_ids
        2. 2-hop neighbor traversal (all rel types including SAME_AS),
           with ALL neighbors strictly scoped to paper_ids
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
        cypher = """
        // Step 1: Vector search — seeds scoped to paper_ids
        CALL db.index.vector.queryNodes('entity', $top_k, $embedding)
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

        // Return all three levels of facts
        WITH seed, r1, hop1, r2, hop2

        // Collect 1-hop facts
        WITH seed,
             collect(DISTINCT {
                source_name: seed.entity_key,
                source_type: [l IN labels(seed) WHERE NOT l IN ['__Entity__', '__Node__'] | l][0],
                source_description: seed.entity_description,
                source_paper_id: seed.paper_id,
                source_paper_name: seed.paper_name,
                relation: type(r1),
                relation_description: r1.relation_description,
                target_name: hop1.entity_key,
                target_type: [l IN labels(hop1) WHERE NOT l IN ['__Entity__', '__Node__'] | l][0],
                target_description: hop1.entity_description,
                target_paper_id: hop1.paper_id,
                target_paper_name: hop1.paper_name
             }) AS hop1_facts,
             collect(DISTINCT {
                source_name: hop1.entity_key,
                source_type: [l IN labels(hop1) WHERE NOT l IN ['__Entity__', '__Node__'] | l][0],
                source_description: hop1.entity_description,
                source_paper_id: hop1.paper_id,
                source_paper_name: hop1.paper_name,
                relation: type(r2),
                relation_description: r2.relation_description,
                target_name: hop2.entity_key,
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
            fact.source_name          AS source_name,
            fact.source_type          AS source_type,
            fact.source_description   AS source_description,
            fact.source_paper_id      AS source_paper_id,
            fact.source_paper_name    AS source_paper_name,
            fact.relation             AS relation,
            fact.relation_description AS relation_description,
            fact.target_name          AS target_name,
            fact.target_type          AS target_type,
            fact.target_description   AS target_description,
            fact.target_paper_id      AS target_paper_id,
            fact.target_paper_name    AS target_paper_name
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
            results.append({
                "source_name": record.get("source_name") or "",
                "source_type": record.get("source_type") or "",
                "source_description": record.get("source_description") or "",
                "source_paper_id": record.get("source_paper_id") or "",
                "source_paper_name": record.get("source_paper_name") or "",
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
            chunk.text       AS text,
            chunk.paper_id   AS paper_id,
            chunk.paper_name AS paper_name
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
                })
        return results

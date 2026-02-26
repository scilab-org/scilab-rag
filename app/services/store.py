"""GraphRAG Store with community detection and summarization."""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional

import networkx as nx
from graspologic.partition import hierarchical_leiden
from llama_index.core.llms import ChatMessage, LLM
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

from app.core.prompts import COMMUNITY_SUMMARY_SYSTEM_PROMPT


logger = logging.getLogger(__name__)


class GraphRAGStore(Neo4jPropertyGraphStore):
    """
    Extended Neo4j Property Graph Store with community detection.
    
    Features:
    - Community detection using Hierarchical Leiden algorithm
    - LLM-based community summarization
    - Entity-to-community mapping
    """
    
    def __init__(self, *args, llm: Optional[LLM] = None, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self, "supports_vector_queries"):
            self.supports_vector_queries = False

        self.llm = llm
        self.community_summary = {}
        self.entity_info = {}
        self.max_cluster_size = 5
    
    async def generate_community_summary(self, text: str) -> str:
        """Generate a summary for community relationships using LLM."""
        messages = [
            ChatMessage(role="system", content=COMMUNITY_SUMMARY_SYSTEM_PROMPT),
            ChatMessage(role="user", content=text),
        ]
        response = await self.llm.achat(messages)
        clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return clean_response
    
    async def build_communities(self) -> None:
        """Build communities from the graph and summarize them."""
        nx_graph = self._create_nx_graph()
        
        if nx_graph.number_of_nodes() == 0:
            return
        
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        
        self.entity_info, community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )
        
        await self._summarize_communities(community_info)
    
    def _create_nx_graph(self) -> nx.Graph:
        """Convert Neo4j graph to NetworkX graph."""
        nx_graph = nx.Graph()
        triplets = self.get_triplets()
        logger.debug("Building NetworkX graph from %d triplets", len(triplets))
        
        for entity1, relation, entity2 in triplets:
            nx_graph.add_node(entity1.name)
            nx_graph.add_node(entity2.name)
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties.get("relationship_description", ""),
            )
        
        return nx_graph
    
    def _collect_community_info(
        self, nx_graph: nx.Graph, clusters
    ) -> tuple:
        """
        Collect entity and community information from clusters.
        
        Returns:
            entity_info: Mapping of entity names to community IDs
            community_info: Mapping of community IDs to relationship details
        """
        entity_info = defaultdict(set)
        community_info = defaultdict(list)
        
        for item in clusters:
            node = item.node
            cluster_id = item.cluster
            
            entity_info[node].add(cluster_id)
            
            for neighbor in nx_graph.neighbors(node):
                edge_data = nx_graph.get_edge_data(node, neighbor)
                if edge_data:
                    detail = (
                        f"{node} -> {neighbor} -> "
                        f"{edge_data['relationship']} -> {edge_data['description']}"
                    )
                    community_info[cluster_id].append(detail)
        
        entity_info = {k: list(v) for k, v in entity_info.items()}
        
        return dict(entity_info), dict(community_info)
    
    async def _summarize_communities(self, community_info: Dict[int, List[str]]) -> None:
        """Generate and store summaries for each community."""
        for community_id, details in community_info.items():
            details_text = "\n".join(details) + "."
            self.community_summary[community_id] = await self.generate_community_summary(
                details_text
            )
    
    async def get_community_summaries(self) -> Dict[int, str]:
        """Get community summaries, building them if not already done."""
        if not self.community_summary:
            await self.build_communities()
        return self.community_summary

"""
Utility functions for graph processing.
"""

import json
import re
from typing import List, Tuple

from llama_index.core.llms import ChatMessage, LLM

def parse_fn(response_str: str) -> Tuple[List, List]:
    """
    Parse LLM response to extract entities and relationships.
    
    Args:
        response_str: Raw LLM response string
    
    Returns:
        Tuple of (entities, relationships)
        - entities: List of (name, types, description) tuples
        - relationships: List of (source, target, relation, description) tuples
    """
    entities, relationships = [], []
    
    # Strip markdown formatting
    response_str = re.sub(r"```json", "", response_str, flags=re.IGNORECASE)
    response_str = re.sub(r"```", "", response_str)
    response_str = response_str.strip()
    
    # Extract JSON block
    match = re.search(r"\{[\s\S]*\}", response_str)
    if not match:
        return entities, relationships
    
    json_str = match.group(0)
    
    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return entities, relationships
    
    # Extract entities
    for e in data.get("entities", []):
        name = e.get("entity_name")
        if not name:          
            continue
        entities.append((name, e.get("entity_type"), e.get("entity_description")))
    
    # Extract relationships
    for r in data.get("relationships", []):
        src = r.get("source_entity")
        tgt = r.get("target_entity")
        if not src or not tgt:   
            continue
        relationships.append((src, tgt, r.get("relation"), r.get("relationship_description")))
    
    return entities, relationships


def normalize_entity_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower().strip()
    name = re.sub(r"\s+", "_", name)      # spaces → underscore
    name = re.sub(r"[^a-z0-9_]", "", name)  # remove punctuation
    return name


def normalize_entity_key(name: str) -> str:
    """Light normalization for SAME_AS matching across papers.
    
    Only normalizes case and separators. Preserves word structure
    so that nuance is not lost.
    
    Examples:
        "Transformer"       -> "transformer"
        "Transformer Model" -> "transformer model"  (different from "transformer")
        "SARS-CoV-2"        -> "sars cov 2"
        "sars-cov-2"        -> "sars cov 2"         (matches above)
        "Self-Attention"     -> "self attention"
        "Self Attention"     -> "self attention"     (matches above)
        "COVID-19"           -> "covid 19"
    """
    if not name:
        return ""
    name = name.strip().lower()
    name = re.sub(r"[\s\-_]+", " ", name)  # unify separators to single space
    name = name.strip()
    return name

def normalize_rel_label(label: str) -> str:
    """Normalise to UPPER_SNAKE_CASE for Neo4j relationship type consistency."""
    label = label.strip()
    # Replace spaces and hyphens with underscores, strip special chars
    label = re.sub(r'[\s\-]+', '_', label)
    label = re.sub(r'[^A-Za-z0-9_]', '', label)
    return label.upper()


async def generate_chat_title(llm: LLM, message: str) -> str:
    prompt = (
        "Generate a concise (max 7 words) scientific chat session title for the following message.\n\n"
        f"Message: \"{message.strip()}\"\n\nTitle:"
    )
    response = await llm.achat([ChatMessage(role="user", content=prompt)])
    raw_title = str(response).strip().strip('"')
    words = re.split(r"\s+", raw_title)
    cut_title = " ".join(words[:7]).strip()
    cut_title = re.sub(r'[!?.:,;\-]+$','', cut_title).strip()
    final_title = cut_title[:1].upper() + cut_title[1:]
    return final_title
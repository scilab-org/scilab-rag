"""
Utility functions for graph processing.
"""

import json
import re
from typing import List, Tuple


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
        entities.append((
            e.get("entity_name"),
            e.get("entity_type"),
            e.get("entity_description"),
        ))
    
    # Extract relationships
    for r in data.get("relationships", []):
        relationships.append((
            r.get("source_entity"),
            r.get("target_entity"),
            r.get("relation"),
            r.get("relationship_description"),
        ))
    
    return entities, relationships






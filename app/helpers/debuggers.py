from asyncio.log import logger
import json
import os
from datetime import datetime


def _write(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def dump_extractor_before(paper_id, entities, relationships):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = f"./data/extractor/{paper_id}_before_{ts}.json"

    data = {
        "entities": [
            {"name": e[0], "type": e[1], "description": e[2]}
            for e in entities
        ],
        "relationships": [
            {
                "source": r[0],
                "target": r[1],
                "relation": r[2],
                "description": r[3],
            }
            for r in relationships
        ],
    }

    _write(path, data)

def dump_extractor_after(nodes, relations):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = f"./data/extractor/after_{ts}.json"

    data = {
        "nodes": [
            {
                "id": n.name,
                "label": n.label,
                "properties": n.properties,
            }
            for n in nodes
        ],
        "relations": [
            {
                "source": r.source_id,
                "target": r.target_id,
                "label": r.label,
                "properties": r.properties,
            }
            for r in relations
        ],
    }

    _write(path, data)


def dump_store_before(triplets):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = f"./data/store/before_{ts}.json"

    data = [
        {
            "entity1": entity1.name,
            "entity2": entity2.name,
            "relation": relation.label,
            "source_id": relation.source_id,
            "target_id": relation.target_id,
        }
        for entity1, relation, entity2 in triplets
    ]

    _write(path, data)


def dump_store_after(nx_graph):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = f"./data/store/after_{ts}.json"

    data = {
        "nodes": list(nx_graph.nodes(data=True)),
        "edges": list(nx_graph.edges(data=True)),
    }

    _write(path, data)
    
    
def dump_communities(community_hierarchical_clusters, entity_info, community_info, community_summary=None):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = f"./data/communities/communities_{ts}.json"

    data = {
        "clusters": [
            {
                "node": c.node,
                "cluster": c.cluster,
                "parent_cluster": c.parent_cluster,
                "level": c.level,
                "is_final_cluster": c.is_final_cluster,
            }
            for c in community_hierarchical_clusters
        ],
        "entity_info": {
            entity_name: cluster_ids
            for entity_name, cluster_ids in entity_info.items()
        },
        "community_info": {
            str(community_id): details
            for community_id, details in community_info.items()
        },
        "community_summary": {
            str(community_id): summary
            for community_id, summary in (community_summary or {}).items()
        },
    }

    _write(path, data)
    
def write_to_data_folder(content: str, filename: str | None) -> str:
    from pathlib import Path

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not filename:
        filename = f"document_{timestamp}.md"
    
    name_without_ext = Path(filename).stem
    timestamped_filename = f"{name_without_ext}_{timestamp}.md"

    # project_root/data
    project_root = Path.cwd()
    data_folder = project_root / "data"
    data_folder.mkdir(parents=True, exist_ok=True)

    file_path = data_folder / timestamped_filename

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"File written to {file_path}")

    return str(file_path)
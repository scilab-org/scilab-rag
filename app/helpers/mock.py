"""Mock functions that will be replaced with real DB calls later."""

from typing import List

# Temporary hardcoded mapping: projectId -> list of paper_ids.
# Replace with a real database query once the project-paper
# association table exists in PostgreSQL.
_PROJECT_PAPERS = {
    "project1": ["paper1", "paper2","paper3"],
    "project2": ["paper4", "paper5"],
}


def get_paper_ids_by_project(project_id: str) -> List[str]:
    """Resolve a projectId to its associated paper_ids.

    Returns an empty list if the project is unknown.
    """
    return _PROJECT_PAPERS.get(project_id, [])

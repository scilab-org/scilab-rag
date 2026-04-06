"""
Writing feature — Prism-inspired hierarchical agent pipeline.

Public surface:
    WritingOrchestrator  — classifies intent per request
    PlanningAgent        — gathers context (Q&A + RAG)
    WritingAgent         — produces LaTeX sections
    ValidationAgent      — auto-fix quality loop
"""

from app.agents.writing.orchestrator import WritingOrchestrator
from app.agents.writing.planning_agent import PlanningAgent
from app.agents.writing.writing_agent import WritingAgent
from app.agents.writing.validation_agent import ValidationAgent

__all__ = [
    "WritingOrchestrator",
    "PlanningAgent",
    "WritingAgent",
    "ValidationAgent",
]

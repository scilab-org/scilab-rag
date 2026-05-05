"""
Writing feature — hierarchical agent pipeline.

Public surface:
    WritingOrchestrator  — binary classification (invoke_planning yes/no)
    PlanningAgent        — gathers context (Query Refiner + RAG + Q&A)
    WritingAgent         — produces LaTeX sections + structured explanation
    ValidationAgent      — ruleset compliance + structural LaTeX validation
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

"""Multi-agent system for depression assessment.

Paper Reference:
    - Section 2.3: Multi-agent system architecture
    - Section 2.3.1: Qualitative Assessment Agent
    - Section 2.3.2: Judge Agent
    - Section 2.3.3: Quantitative Assessment Agent
    - Section 2.3.4: Meta-Review Agent

This module provides the agent implementations for the depression
assessment pipeline.
"""

from ai_psychiatrist.agents.judge import JudgeAgent
from ai_psychiatrist.agents.qualitative import QualitativeAssessmentAgent

__all__ = [
    "JudgeAgent",
    "QualitativeAssessmentAgent",
]

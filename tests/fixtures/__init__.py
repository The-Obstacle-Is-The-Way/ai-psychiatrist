"""Test fixtures module.

This module contains test doubles and fixtures that are test-only artifacts.
These MUST NOT be imported from production code (src/).

Per Clean Architecture (Robert C. Martin):
    Test doubles are outer circle concerns and must not pollute inner circles.

Per ISO 27001 Control 8.31:
    Development, testing and production environments should be separated.
"""

from __future__ import annotations

from tests.fixtures.mock_llm import MockLLMClient

__all__ = ["MockLLMClient"]

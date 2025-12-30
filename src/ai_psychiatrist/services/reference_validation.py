"""Reference validation (CRAG-style) strategies.

This module implements Spec 36: CRAG-Style Runtime Reference Validation.
It provides a mechanism to validate retrieved references against the query
using an LLM, filtering out irrelevant or contradictory matches.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from ai_psychiatrist.domain.enums import PHQ8Item
    from ai_psychiatrist.infrastructure.llm.responses import SimpleChatClient

logger = get_logger(__name__)

Decision = Literal["accept", "reject", "unsure"]


@dataclass(frozen=True, slots=True)
class ReferenceValidationRequest:
    """Request to validate a retrieved reference."""

    item: PHQ8Item
    evidence_text: str
    reference_text: str
    reference_score: int | None


class ReferenceValidator(Protocol):
    """Protocol for reference validation strategies."""

    async def validate(self, request: ReferenceValidationRequest) -> Decision:
        """Validate if the reference is relevant and accurate for the evidence.

        Args:
            request: Validation request containing item, query, and reference.

        Returns:
            Decision: "accept", "reject", or "unsure".
        """
        ...


class NoOpReferenceValidator:
    """Validator that accepts all references (default behavior)."""

    async def validate(self, _request: ReferenceValidationRequest) -> Decision:
        """Always accept."""
        return "accept"


class LLMReferenceValidator:
    """Validator that uses an LLM to evaluate relevance."""

    def __init__(self, client: SimpleChatClient, model: str) -> None:
        """Initialize LLM validator.

        Args:
            client: Chat client for LLM calls.
            model: Model identifier.
        """
        self._client = client
        self._model = model

    async def validate(self, request: ReferenceValidationRequest) -> Decision:
        """Validate reference using LLM."""
        prompt = self._build_prompt(request)

        try:
            response = await self._client.simple_chat(
                user_prompt=prompt,
                system_prompt="You are a strict data validator. Output JSON only.",
                model=self._model,
                temperature=0.0,  # Deterministic
            )

            return self._parse_decision(response)
        except Exception as e:
            logger.warning("Reference validation failed", error=str(e))
            return "unsure"  # Fail safe -> treated as reject by default logic

    def _build_prompt(self, request: ReferenceValidationRequest) -> str:
        """Build validation prompt."""
        return f"""You are validating a retrieved reference for a PHQ-8 assessment.

Item: {request.item.value}
Evidence (Query): "{request.evidence_text}"
Retrieved Reference: "{request.reference_text}"
Reference Score: {request.reference_score if request.reference_score is not None else "N/A"}

Task:
- Determine if the retrieved reference is semantically relevant to the evidence.
- The reference should be a good example of the symptom described in the evidence.
- If the reference contradicts the evidence or is irrelevant, reject it.

Return JSON only:
{{
  "decision": "accept" | "reject" | "unsure"
}}
"""

    def _parse_decision(self, response: str) -> Decision:
        """Parse LLM response."""
        try:
            content = response.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            data = json.loads(content)
            decision = data.get("decision")

            if decision in ("accept", "reject", "unsure"):
                return decision  # type: ignore

            return "unsure"
        except json.JSONDecodeError:
            return "unsure"

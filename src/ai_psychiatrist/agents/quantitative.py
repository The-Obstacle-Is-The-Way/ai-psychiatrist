"""Quantitative assessment agent for PHQ-8 scoring.

Paper Reference:
    - Section 2.3.2: Quantitative Assessment
    - Section 2.4.2: Few-shot prompting workflow
    - Appendix D: Optimal hyperparameters (chunk_size=8, top_k=2, dim=4096)
    - Appendix F: MedGemma achieves 18% better MAE

This agent implements embedding-based few-shot prompting for PHQ-8 score
prediction, with multi-level JSON repair for robust parsing.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import spacy

from ai_psychiatrist.agents.prompts.quantitative import (
    QUANTITATIVE_SYSTEM_PROMPT,
    make_evidence_prompt,
    make_scoring_prompt,
)
from ai_psychiatrist.config.domain_constants import DOMAIN_KEYWORDS
from ai_psychiatrist.domain.entities import PHQ8Assessment, Transcript
from ai_psychiatrist.domain.enums import AssessmentMode, PHQ8Item
from ai_psychiatrist.domain.value_objects import ItemAssessment
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from ai_psychiatrist.config import ModelSettings
    from ai_psychiatrist.infrastructure.llm.responses import SimpleChatClient
    from ai_psychiatrist.services.embedding import EmbeddingService

logger = get_logger(__name__)

# Mapping from legacy string keys to PHQ8Item enum
PHQ8_KEY_MAP: dict[str, PHQ8Item] = {
    "PHQ8_NoInterest": PHQ8Item.NO_INTEREST,
    "PHQ8_Depressed": PHQ8Item.DEPRESSED,
    "PHQ8_Sleep": PHQ8Item.SLEEP,
    "PHQ8_Tired": PHQ8Item.TIRED,
    "PHQ8_Appetite": PHQ8Item.APPETITE,
    "PHQ8_Failure": PHQ8Item.FAILURE,
    "PHQ8_Concentrating": PHQ8Item.CONCENTRATING,
    "PHQ8_Moving": PHQ8Item.MOVING,
}


class QuantitativeAssessmentAgent:
    """Agent for predicting PHQ-8 scores from interview transcripts.

    Supports two modes:
    - Zero-shot: Direct prediction without reference examples
    - Few-shot: Uses embedding-based reference retrieval (paper optimal)

    The few-shot approach achieves MAE of 0.619 vs 0.796 for zero-shot
    (22% improvement per paper Section 3.2).

    Example:
        >>> from tests.fixtures.mock_llm import MockLLMClient
        >>> client = MockLLMClient(chat_responses=[...])
        >>> agent = QuantitativeAssessmentAgent(llm_client=client)
        >>> transcript = Transcript(participant_id=123, text="...")
        >>> assessment = await agent.assess(transcript)
    """

    def __init__(
        self,
        llm_client: SimpleChatClient,
        embedding_service: EmbeddingService | None = None,
        mode: AssessmentMode = AssessmentMode.FEW_SHOT,
        model_settings: ModelSettings | None = None,
    ) -> None:
        """Initialize quantitative assessment agent.

        Args:
            llm_client: LLM client for chat completions.
            embedding_service: Service for few-shot retrieval (required for FEW_SHOT mode).
            mode: Assessment mode (ZERO_SHOT or FEW_SHOT).
            model_settings: Model configuration. If None, uses OllamaClient defaults.
        """
        self._llm = llm_client
        self._embedding = embedding_service
        self._mode = mode
        self._model_settings = model_settings

        # Warn if FEW_SHOT mode is used without embedding service
        if mode == AssessmentMode.FEW_SHOT and embedding_service is None:
            logger.warning(
                "FEW_SHOT mode selected but no embedding_service provided; "
                "will operate without reference examples (similar to zero-shot)"
            )

    async def assess(self, transcript: Transcript) -> PHQ8Assessment:
        """Generate PHQ-8 assessment for transcript.

        Pipeline:
        1. Extract evidence for each PHQ-8 item (with keyword backfill)
        2. (Few-shot) Build reference bundle via embeddings
        3. Score with LLM using evidence + references
        4. Parse and validate scores with repair

        Args:
            transcript: Interview transcript to assess.

        Returns:
            PHQ8Assessment with scores for all 8 items.
        """
        logger.info(
            "Starting quantitative assessment",
            participant_id=transcript.participant_id,
            mode=self._mode.value,
        )

        # Step 1: Extract evidence
        evidence_dict = await self._extract_evidence(transcript.text)

        logger.debug(
            "Evidence extracted",
            items_with_evidence=sum(1 for v in evidence_dict.values() if v),
        )

        # Step 2: Build references (few-shot only)
        reference_text = ""
        if self._mode == AssessmentMode.FEW_SHOT and self._embedding:
            # Convert string keys to PHQ8Item for embedding service
            evidence_for_embedding = {
                PHQ8_KEY_MAP[k]: v for k, v in evidence_dict.items() if k in PHQ8_KEY_MAP
            }
            bundle = await self._embedding.build_reference_bundle(evidence_for_embedding)
            reference_text = bundle.format_for_prompt()
            logger.debug("Reference bundle built", bundle_length=len(reference_text))

        # Step 3: Score with LLM
        prompt = make_scoring_prompt(transcript.text, reference_text)

        # Use model settings if provided (Paper Appendix F: MedGemma achieves 18% better MAE)
        model = self._model_settings.quantitative_model if self._model_settings else None
        temperature = self._model_settings.temperature if self._model_settings else 0.2
        top_k = self._model_settings.top_k if self._model_settings else 20
        top_p = self._model_settings.top_p if self._model_settings else 0.8

        # BUG-002: Use robust JSON mode instead of regex parsing
        raw_response = await self._llm.simple_chat(
            user_prompt=prompt,
            system_prompt=QUANTITATIVE_SYSTEM_PROMPT,
            model=model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            response_format="json",
        )

        logger.debug("LLM scoring complete", response_length=len(raw_response))

        # Step 4: Parse response (now much simpler)
        items = await self._parse_response(raw_response)

        assessment = PHQ8Assessment(
            items=items,
            mode=self._mode,
            participant_id=transcript.participant_id,
        )

        logger.info(
            "Quantitative assessment complete",
            participant_id=transcript.participant_id,
            total_score=assessment.total_score,
            severity=assessment.severity.name,
            na_count=assessment.na_count,
        )

        return assessment

    async def _extract_evidence(self, transcript_text: str) -> dict[str, list[str]]:
        """Extract evidence quotes for each PHQ-8 item.

        Uses LLM extraction with keyword backfill to ensure coverage.

        Args:
            transcript_text: Interview transcript text.

        Returns:
            Dictionary of PHQ8 key -> list of evidence quotes.
        """
        user_prompt = make_evidence_prompt(transcript_text)

        # Use model settings if provided
        model = self._model_settings.quantitative_model if self._model_settings else None
        temperature = self._model_settings.temperature if self._model_settings else 0.2
        top_k = self._model_settings.top_k if self._model_settings else 20
        top_p = self._model_settings.top_p if self._model_settings else 0.8

        raw = await self._llm.simple_chat(
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Parse JSON response
        try:
            # We don't force JSON mode for extraction yet, so strip markdown
            clean = self._strip_markdown(raw)
            obj = json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Failed to parse evidence JSON, using empty evidence",
                response_preview=raw[:200] if raw else "",
            )
            obj = {}

        # Clean up extraction, ensure all keys present
        evidence_dict: dict[str, list[str]] = {}
        for key in DOMAIN_KEYWORDS:
            arr = obj.get(key, []) if isinstance(obj, dict) else []
            if not isinstance(arr, list):
                arr = []
            # Dedupe and clean quotes
            cleaned = list({str(q).strip() for q in arr if str(q).strip()})
            evidence_dict[key] = cleaned

        # Keyword backfill for missed evidence
        enriched = self._keyword_backfill(transcript_text, evidence_dict)
        return enriched

    def _keyword_backfill(
        self,
        transcript: str,
        current: dict[str, list[str]],
        cap: int = 3,
    ) -> dict[str, list[str]]:
        """Add keyword-matched sentences when LLM misses evidence.

        Args:
            transcript: Original transcript text.
            current: Current evidence dictionary.
            cap: Maximum evidence items per PHQ-8 domain.

        Returns:
            Enriched evidence dictionary.
        """
        # BUG-003: Use Spacy for robust sentence splitting
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(transcript.strip())
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except Exception:
            # Fallback to simple splitting if model fails to load (though it should be installed)
            logger.warning("Spacy failed, falling back to simple sentence splitting")
            sentences = [s.strip() for s in transcript.split("\n") if s.strip()]

        out = {k: list(v) for k, v in current.items()}

        for key, keywords in DOMAIN_KEYWORDS.items():
            need = max(0, cap - len(out.get(key, [])))
            if need == 0:
                continue

            hits: list[str] = []
            for sent in sentences:
                sent_lower = sent.lower()
                if any(kw in sent_lower for kw in keywords):
                    hits.append(sent)
                if len(hits) >= need:
                    break

            if hits:
                existing = set(out.get(key, []))
                merged = out.get(key, []) + [h for h in hits if h not in existing]
                out[key] = merged[:cap]

        return out

    async def _parse_response(self, raw: str) -> dict[PHQ8Item, ItemAssessment]:
        """Parse JSON response.

        Assumes LLM returns valid JSON due to `format='json'`.
        Falls back to parsing if wrapped in tags, but avoids complex repairs.

        Args:
            raw: Raw LLM response text.

        Returns:
            Dictionary mapping PHQ8Item to ItemAssessment.
        """
        try:
            # Even with JSON mode, some models might wrap in markdown
            clean = self._strip_markdown(raw)
            data = json.loads(clean)
            if not isinstance(data, dict):
                raise ValueError("Quantitative response JSON must be an object")
            return self._validate_and_normalize(data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse quantitative response", error=str(e), raw=raw[:200])
            return self._validate_and_normalize({})

    def _strip_markdown(self, text: str) -> str:
        """Strip markdown code blocks if present."""
        cleaned = text.strip()
        if "```json" in cleaned:
            start = cleaned.find("```json") + 7
            end = cleaned.rfind("```")
            if end > start:
                return cleaned[start:end].strip()
        elif "```" in cleaned:
            start = cleaned.find("```") + 3
            end = cleaned.rfind("```")
            if end > start:
                return cleaned[start:end].strip()
        return cleaned

    def _validate_and_normalize(self, data: dict[str, object]) -> dict[PHQ8Item, ItemAssessment]:
        """Convert raw dict to typed ItemAssessment objects.

        Args:
            data: Parsed JSON dict with PHQ8 keys.

        Returns:
            Dictionary mapping PHQ8Item to validated ItemAssessment.
        """
        result: dict[PHQ8Item, ItemAssessment] = {}

        for key, item_enum in PHQ8_KEY_MAP.items():
            item_data = data.get(key, {})

            if not isinstance(item_data, dict):
                item_data = {}

            # Extract fields with defaults
            evidence = str(item_data.get("evidence", "No relevant evidence found"))
            reason = str(item_data.get("reason", "Unable to assess"))

            # Parse score (can be int, float, string "N/A", or None)
            raw_score = item_data.get("score")
            score: int | None = None

            if raw_score is not None:
                if isinstance(raw_score, int) and 0 <= raw_score <= 3:
                    score = raw_score
                elif isinstance(raw_score, float) and raw_score == int(raw_score):
                    # Handle float scores like 2.0 from JSON parsing
                    parsed = int(raw_score)
                    if 0 <= parsed <= 3:
                        score = parsed
                elif isinstance(raw_score, str) and raw_score.upper() != "N/A":
                    try:
                        parsed = int(raw_score)
                        if 0 <= parsed <= 3:
                            score = parsed
                    except ValueError:
                        pass

            result[item_enum] = ItemAssessment(
                item=item_enum,
                evidence=evidence,
                reason=reason,
                score=score,
            )

        return result

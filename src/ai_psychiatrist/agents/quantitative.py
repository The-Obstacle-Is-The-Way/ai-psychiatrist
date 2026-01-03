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

import asyncio
import json
import re
from typing import TYPE_CHECKING, Literal

from ai_psychiatrist.agents.prompts.quantitative import (
    DOMAIN_KEYWORDS,
    QUANTITATIVE_SYSTEM_PROMPT,
    make_evidence_prompt,
    make_scoring_prompt,
)
from ai_psychiatrist.config import (
    PydanticAISettings,
    QuantitativeSettings,
    get_model_name,
)
from ai_psychiatrist.domain.entities import PHQ8Assessment, Transcript
from ai_psychiatrist.domain.enums import AssessmentMode, NAReason, PHQ8Item
from ai_psychiatrist.domain.value_objects import ItemAssessment
from ai_psychiatrist.infrastructure.llm.responses import tolerant_json_fixups
from ai_psychiatrist.infrastructure.logging import get_logger
from ai_psychiatrist.services.embedding import ReferenceBundle, compute_retrieval_similarity_stats

if TYPE_CHECKING:
    from pydantic_ai import Agent

    from ai_psychiatrist.agents.output_models import QuantitativeOutput
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

# Reverse mapping for lookups
ITEM_TO_LEGACY_KEY: dict[PHQ8Item, str] = {v: k for k, v in PHQ8_KEY_MAP.items()}


class QuantitativeAssessmentAgent:
    """Agent for predicting PHQ-8 scores from interview transcripts.

    Supports two modes:
    - Zero-shot: Direct prediction without reference examples
    - Few-shot: Uses embedding-based reference retrieval (Appendix D hyperparameters)

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
        quantitative_settings: QuantitativeSettings | None = None,
        pydantic_ai_settings: PydanticAISettings | None = None,
        ollama_base_url: str | None = None,
    ) -> None:
        """Initialize quantitative assessment agent.

        Args:
            llm_client: LLM client for chat completions.
            embedding_service: Service for few-shot retrieval (required for FEW_SHOT mode).
            mode: Assessment mode (ZERO_SHOT or FEW_SHOT).
            model_settings: Model configuration. If None, uses OllamaClient defaults.
            quantitative_settings: Quantitative settings. If None, uses defaults.
            pydantic_ai_settings: Pydantic AI configuration. If None, uses defaults.
            ollama_base_url: Ollama base URL for Pydantic AI agent. Required when
                pydantic_ai_settings.enabled is True.
        """
        self._llm = llm_client
        self._embedding = embedding_service
        self._mode = mode
        self._model_settings = model_settings
        self._settings = quantitative_settings or QuantitativeSettings()
        self._pydantic_ai = pydantic_ai_settings or PydanticAISettings()
        self._ollama_base_url = ollama_base_url
        self._scoring_agent: Agent[None, QuantitativeOutput] | None = None

        if self._pydantic_ai.enabled:
            if not self._ollama_base_url:
                raise ValueError(
                    "Pydantic AI enabled but no ollama_base_url provided. "
                    "Legacy fallback is disabled."
                )
            else:
                from ai_psychiatrist.agents.pydantic_agents import (  # noqa: PLC0415
                    create_quantitative_agent,
                )

                self._scoring_agent = create_quantitative_agent(
                    model_name=get_model_name(model_settings, "quantitative"),
                    base_url=self._ollama_base_url,
                    retries=self._pydantic_ai.retries,
                    system_prompt=QUANTITATIVE_SYSTEM_PROMPT,
                )

        # Warn if FEW_SHOT mode is used without embedding service
        if mode == AssessmentMode.FEW_SHOT and embedding_service is None:
            logger.warning(
                "FEW_SHOT mode selected but no embedding_service provided; "
                "will operate without reference examples (similar to zero-shot)"
            )

    async def assess(self, transcript: Transcript) -> PHQ8Assessment:
        """Generate PHQ-8 assessment for transcript.

        Pipeline:
        1. Extract evidence for each PHQ-8 item (optional keyword backfill)
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
        llm_evidence = await self._extract_evidence(transcript.text)
        llm_counts = {k: len(v) for k, v in llm_evidence.items()}

        # Step 2: Find keyword hits (always computed for observability/N/A reasons)
        keyword_hits: dict[str, list[str]] = {}
        keyword_hit_counts: dict[str, int] = {}
        if self._settings.enable_keyword_backfill or self._settings.track_na_reasons:
            keyword_hits = self._find_keyword_hits(
                transcript.text,
                cap=self._settings.keyword_backfill_cap,
            )
            keyword_hit_counts = {k: len(v) for k, v in keyword_hits.items()}

        # Step 3: Conditional backfill
        if self._settings.enable_keyword_backfill:
            # Merge LLM evidence with keyword hits
            final_evidence = self._merge_evidence(
                llm_evidence, keyword_hits, cap=self._settings.keyword_backfill_cap
            )
        else:
            final_evidence = llm_evidence

        # Calculate added evidence from backfill
        keyword_added_counts = {
            k: len(final_evidence.get(k, [])) - llm_counts.get(k, 0) for k in final_evidence
        }

        logger.debug(
            "Evidence extracted",
            items_with_evidence=sum(1 for v in final_evidence.values() if v),
        )

        # Step 2: Build references (few-shot only)
        reference_text = ""
        reference_bundle: ReferenceBundle | None = None
        if self._mode == AssessmentMode.FEW_SHOT and self._embedding:
            # Convert string keys to PHQ8Item for embedding service
            evidence_for_embedding = {
                PHQ8_KEY_MAP[k]: v for k, v in final_evidence.items() if k in PHQ8_KEY_MAP
            }
            reference_bundle = await self._embedding.build_reference_bundle(evidence_for_embedding)
            reference_text = reference_bundle.format_for_prompt()
            logger.debug("Reference bundle built", bundle_length=len(reference_text))

        # Step 3: Score with LLM
        prompt = make_scoring_prompt(transcript.text, reference_text)

        # Model settings (Appendix F: MedGemma; GAP-001: temp=0.0 for reproducibility)
        model = get_model_name(self._model_settings, "quantitative")
        temperature = self._model_settings.temperature if self._model_settings else 0.0

        parsed_items = await self._score_items(prompt=prompt, _model=model, temperature=temperature)

        # Step 5: Construct ItemAssessments with extended fields
        final_items = {}
        for phq_item in PHQ8Item:
            legacy_key = ITEM_TO_LEGACY_KEY[phq_item]

            # Get parsing result (or default empty)
            parsed = parsed_items.get(phq_item)

            # If parsed result exists and has a score, use it
            # If parsed result exists but score is None (explicit N/A), check reasons

            if parsed:
                score = parsed.score
                evidence = parsed.evidence
                reason = parsed.reason
            else:
                score = None
                evidence = "No relevant evidence found"
                reason = "Unable to assess"

            # Determine NA reason and Evidence Source
            na_reason: NAReason | None = None
            evidence_source = self._determine_evidence_source(
                llm_count=llm_counts.get(legacy_key, 0),
                keyword_added_count=keyword_added_counts.get(legacy_key, 0),
            )

            if score is None and self._settings.track_na_reasons:
                na_reason = self._determine_na_reason(
                    llm_count=llm_counts.get(legacy_key, 0),
                    keyword_count=keyword_hit_counts.get(legacy_key, 0),
                    backfill_enabled=self._settings.enable_keyword_backfill,
                )

            # Fix evidence source if we found nothing but the LLM still produced a score.
            if evidence_source is None and score is not None:
                # This shouldn't happen usually unless LLM hallucinated evidence
                evidence_source = "llm"

            retrieval_stats = compute_retrieval_similarity_stats(
                reference_bundle.item_references.get(phq_item, []) if reference_bundle else []
            )

            final_items[phq_item] = ItemAssessment(
                item=phq_item,
                evidence=evidence,
                reason=reason,
                score=score,
                na_reason=na_reason,
                evidence_source=evidence_source,
                llm_evidence_count=llm_counts.get(legacy_key, 0),
                keyword_evidence_count=keyword_added_counts.get(legacy_key, 0),
                retrieval_reference_count=retrieval_stats.retrieval_reference_count,
                retrieval_similarity_mean=retrieval_stats.retrieval_similarity_mean,
                retrieval_similarity_max=retrieval_stats.retrieval_similarity_max,
            )

        assessment = PHQ8Assessment(
            items=final_items,
            mode=self._mode,
            participant_id=transcript.participant_id,
        )

        severity = assessment.severity.name if assessment.severity is not None else None

        logger.info(
            "Quantitative assessment complete",
            participant_id=transcript.participant_id,
            total_score=assessment.total_score,
            total_score_min=assessment.min_total_score,
            total_score_max=assessment.max_total_score,
            severity=severity,
            severity_lower_bound=assessment.severity_lower_bound.name,
            severity_upper_bound=assessment.severity_upper_bound.name,
            na_count=assessment.na_count,
        )

        return assessment

    async def _score_items(
        self,
        *,
        prompt: str,
        _model: str | None,
        temperature: float,
    ) -> dict[PHQ8Item, ItemAssessment]:
        """Score transcript and return parsed per-item assessments."""
        if self._scoring_agent is None:
            raise ValueError("Pydantic AI scoring agent not initialized")

        try:
            timeout = self._pydantic_ai.timeout_seconds
            result = await self._scoring_agent.run(
                prompt,
                model_settings={
                    "temperature": temperature,
                    **({"timeout": timeout} if timeout is not None else {}),
                },
            )
            return self._from_quantitative_output(result.output)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                "Pydantic AI call failed during scoring",
                error=str(e),
                error_type=type(e).__name__,
                prompt_chars=len(prompt),
                temperature=temperature,
            )
            raise

    @staticmethod
    def _from_quantitative_output(output: QuantitativeOutput) -> dict[PHQ8Item, ItemAssessment]:
        """Convert validated QuantitativeOutput into ItemAssessment mapping."""
        result: dict[PHQ8Item, ItemAssessment] = {}
        for key, item_enum in PHQ8_KEY_MAP.items():
            evidence_output = getattr(output, key)
            result[item_enum] = ItemAssessment(
                item=item_enum,
                evidence=evidence_output.evidence,
                reason=evidence_output.reason,
                score=evidence_output.score,
            )
        return result

    async def _extract_evidence(self, transcript_text: str) -> dict[str, list[str]]:
        """Extract evidence quotes for each PHQ-8 item.

        Uses LLM extraction only. Keyword backfill (if enabled) is applied later in
        `assess()` via `_find_keyword_hits()` and `_merge_evidence()`.

        Args:
            transcript_text: Interview transcript text.

        Returns:
            Dictionary of PHQ8 key -> list of evidence quotes.
        """
        user_prompt = make_evidence_prompt(transcript_text)

        # Use model settings if provided (GAP-001: temp=0.0 for clinical reproducibility)
        model = get_model_name(self._model_settings, "quantitative")
        temperature = self._model_settings.temperature if self._model_settings else 0.0

        raw = await self._llm.simple_chat(
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
        )

        # Parse JSON response with tolerant fixups (BUG-011: Apply repair before parsing)
        try:
            clean = self._strip_json_block(raw)
            clean = tolerant_json_fixups(clean)
            obj = json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            # BUG-011: Include response preview in warning to aid debugging
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

        return evidence_dict

    def _find_keyword_hits(
        self,
        transcript: str,
        cap: int = 3,
    ) -> dict[str, list[str]]:
        """Find keyword-matched sentences in transcript.

        Args:
            transcript: Original transcript text.
            cap: Maximum evidence items per PHQ-8 domain to find.

        Returns:
            Dictionary of PHQ8 key -> list of matched sentences.
        """
        # Split transcript into sentences
        parts = re.split(r"(?<=[.?!])\s+|\n+", transcript.strip())
        sentences = [p.strip() for p in parts if p and len(p.strip()) > 0]

        hits: dict[str, list[str]] = {}

        for key, keywords in DOMAIN_KEYWORDS.items():
            key_hits: list[str] = []
            for sent in sentences:
                sent_lower = sent.lower()
                if any(kw in sent_lower for kw in keywords):
                    key_hits.append(sent)
                if len(key_hits) >= cap:
                    break
            hits[key] = key_hits

        return hits

    def _merge_evidence(
        self,
        current: dict[str, list[str]],
        hits: dict[str, list[str]],
        cap: int = 3,
    ) -> dict[str, list[str]]:
        """Merge keyword hits into current evidence, respecting cap.

        Args:
            current: Current evidence dictionary (from LLM).
            hits: Keyword hits dictionary.
            cap: Maximum total evidence items per PHQ-8 domain.

        Returns:
            Enriched evidence dictionary.
        """
        out = {k: list(v) for k, v in current.items()}

        for key, key_hits in hits.items():
            current_items = out.get(key, [])
            if len(current_items) >= cap:
                continue

            need = cap - len(current_items)

            # Filter hits that are already in current (exact string match)
            existing = set(current_items)
            new_hits = [h for h in key_hits if h not in existing]

            # Take only what we need
            merged = current_items + new_hits[:need]
            out[key] = merged

        return out

    def _strip_json_block(self, text: str) -> str:
        """Strip markdown code blocks and XML tags.

        Args:
            text: Raw text that may contain JSON.

        Returns:
            Cleaned JSON string.
        """
        cleaned = text.strip()

        # Extract from <answer> tags
        if "<answer>" in cleaned and "</answer>" in cleaned:
            cleaned = cleaned.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()

        # Extract from markdown code blocks (handles embedded blocks)
        if "```json" in cleaned:
            # Find the JSON block start and end
            start_idx = cleaned.find("```json") + len("```json")
            end_idx = cleaned.find("```", start_idx)
            if end_idx > start_idx:
                cleaned = cleaned[start_idx:end_idx].strip()
        elif "```" in cleaned:
            # Generic code block
            start_idx = cleaned.find("```") + len("```")
            end_idx = cleaned.find("```", start_idx)
            if end_idx > start_idx:
                cleaned = cleaned[start_idx:end_idx].strip()

        # Handle case where block starts at beginning
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json") :].strip()
        if cleaned.startswith("```"):
            cleaned = cleaned[len("```") :].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        return cleaned

    def _determine_na_reason(
        self,
        llm_count: int,
        keyword_count: int,
        backfill_enabled: bool,
    ) -> NAReason:
        """Determine why an item has no score."""
        if llm_count == 0 and keyword_count == 0:
            return NAReason.NO_MENTION
        if llm_count == 0 and keyword_count > 0 and not backfill_enabled:
            return NAReason.LLM_ONLY_MISSED
        if llm_count == 0 and keyword_count > 0 and backfill_enabled:
            return NAReason.KEYWORDS_INSUFFICIENT
        return NAReason.SCORE_NA_WITH_EVIDENCE

    def _determine_evidence_source(
        self, llm_count: int, keyword_added_count: int
    ) -> Literal["llm", "keyword", "both"] | None:
        """Determine source of evidence."""
        if llm_count > 0 and keyword_added_count > 0:
            return "both"
        if llm_count > 0:
            return "llm"
        if keyword_added_count > 0:
            return "keyword"
        return None

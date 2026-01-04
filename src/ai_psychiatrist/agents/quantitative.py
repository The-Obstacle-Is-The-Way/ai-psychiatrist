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
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

from ai_psychiatrist.agents.prompts.quantitative import (
    QUANTITATIVE_SYSTEM_PROMPT,
    make_evidence_prompt,
    make_scoring_prompt,
)
from ai_psychiatrist.confidence.consistency import compute_consistency_metrics
from ai_psychiatrist.confidence.token_csfs import (
    compute_token_energy,
    compute_token_msp,
    compute_token_pe,
)
from ai_psychiatrist.config import (
    PydanticAISettings,
    QuantitativeSettings,
    get_model_name,
)
from ai_psychiatrist.domain.entities import PHQ8Assessment, Transcript
from ai_psychiatrist.domain.enums import AssessmentMode, NAReason, PHQ8Item
from ai_psychiatrist.domain.value_objects import ItemAssessment
from ai_psychiatrist.infrastructure.hashing import stable_text_hash
from ai_psychiatrist.infrastructure.llm.responses import parse_llm_json
from ai_psychiatrist.infrastructure.logging import get_logger
from ai_psychiatrist.infrastructure.observability import (
    FailureCategory,
    FailureSeverity,
    record_failure,
)
from ai_psychiatrist.services.embedding import ReferenceBundle, compute_retrieval_similarity_stats
from ai_psychiatrist.services.evidence_validation import (
    EvidenceGroundingError,
    EvidenceSchemaError,
    validate_evidence_grounding,
    validate_evidence_schema,
)

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModelSettings

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
_DEFAULT_TOP_LOGPROBS = 5


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
        1. Extract evidence for each PHQ-8 item
        2. (Few-shot) Build reference bundle via embeddings
        3. Score with LLM using transcript + references
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
        final_evidence = await self._extract_evidence(
            transcript.text,
            participant_id=transcript.participant_id,
        )
        llm_counts = {k: len(v) for k, v in final_evidence.items()}

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
                verbalized_confidence = parsed.verbalized_confidence if score is not None else None
                token_msp = parsed.token_msp
                token_pe = parsed.token_pe
                token_energy = parsed.token_energy
            else:
                score = None
                evidence = "No relevant evidence found"
                reason = "Unable to assess"
                verbalized_confidence = None
                token_msp = None
                token_pe = None
                token_energy = None

            # Determine NA reason and Evidence Source
            na_reason: NAReason | None = None
            llm_count = llm_counts.get(legacy_key, 0)
            evidence_source: Literal["llm"] | None = "llm" if llm_count > 0 else None

            if score is None and self._settings.track_na_reasons:
                na_reason = self._determine_na_reason(llm_count)

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
                retrieval_reference_count=retrieval_stats.retrieval_reference_count,
                retrieval_similarity_mean=retrieval_stats.retrieval_similarity_mean,
                retrieval_similarity_max=retrieval_stats.retrieval_similarity_max,
                verbalized_confidence=verbalized_confidence,
                token_msp=token_msp,
                token_pe=token_pe,
                token_energy=token_energy,
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

    async def assess_with_consistency(
        self,
        transcript: Transcript,
        *,
        n_samples: int,
        temperature: float,
    ) -> PHQ8Assessment:
        """Run multiple scoring samples and attach consistency-based confidence signals.

        Spec 050.
        """
        self._validate_consistency_params(n_samples=n_samples, temperature=temperature)

        logger.info(
            "Starting quantitative assessment (consistency mode)",
            participant_id=transcript.participant_id,
            mode=self._mode.value,
            n_samples=n_samples,
            temperature=temperature,
        )

        final_evidence = await self._extract_evidence(
            transcript.text,
            participant_id=transcript.participant_id,
        )
        llm_counts = {k: len(v) for k, v in final_evidence.items()}

        reference_bundle = await self._maybe_build_reference_bundle(final_evidence)
        reference_text = reference_bundle.format_for_prompt() if reference_bundle else ""

        prompt = make_scoring_prompt(transcript.text, reference_text)
        model = get_model_name(self._model_settings, "quantitative")

        sample_outputs = await self._collect_consistency_samples(
            prompt=prompt,
            model=model,
            temperature=temperature,
            n_samples=n_samples,
            participant_id=transcript.participant_id,
        )

        final_items = self._build_consistency_items(
            llm_counts=llm_counts,
            sample_outputs=sample_outputs,
            reference_bundle=reference_bundle,
        )

        return PHQ8Assessment(
            items=final_items,
            mode=self._mode,
            participant_id=transcript.participant_id,
        )

    @staticmethod
    def _validate_consistency_params(*, n_samples: int, temperature: float) -> None:
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be in [0.0, 2.0]")

    async def _maybe_build_reference_bundle(
        self, evidence: dict[str, list[str]]
    ) -> ReferenceBundle | None:
        if self._mode != AssessmentMode.FEW_SHOT or not self._embedding:
            return None

        evidence_for_embedding = {
            PHQ8_KEY_MAP[key]: values for key, values in evidence.items() if key in PHQ8_KEY_MAP
        }
        return await self._embedding.build_reference_bundle(evidence_for_embedding)

    async def _collect_consistency_samples(
        self,
        *,
        prompt: str,
        model: str | None,
        temperature: float,
        n_samples: int,
        participant_id: int,
    ) -> list[dict[PHQ8Item, ItemAssessment]]:
        sample_outputs: list[dict[PHQ8Item, ItemAssessment]] = []
        last_error: Exception | None = None
        max_attempts = max(n_samples * 2, n_samples)

        for attempt in range(1, max_attempts + 1):
            if len(sample_outputs) >= n_samples:
                break
            try:
                sample_outputs.append(
                    await self._score_items(prompt=prompt, _model=model, temperature=temperature)
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Consistency sample failed",
                    participant_id=participant_id,
                    mode=self._mode.value,
                    attempt=attempt,
                    requested_samples=n_samples,
                    collected_samples=len(sample_outputs),
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

        if not sample_outputs and last_error is not None:
            raise last_error
        if len(sample_outputs) < n_samples:
            logger.warning(
                "Consistency run produced fewer samples than requested",
                participant_id=participant_id,
                mode=self._mode.value,
                requested_samples=n_samples,
                collected_samples=len(sample_outputs),
                max_attempts=max_attempts,
            )

        return sample_outputs

    def _build_consistency_items(
        self,
        *,
        llm_counts: dict[str, int],
        sample_outputs: list[dict[PHQ8Item, ItemAssessment]],
        reference_bundle: ReferenceBundle | None,
    ) -> dict[PHQ8Item, ItemAssessment]:
        if not sample_outputs:
            msg = "sample_outputs must not be empty"
            raise ValueError(msg)

        final_items: dict[PHQ8Item, ItemAssessment] = {}
        for phq_item in PHQ8Item:
            legacy_key = ITEM_TO_LEGACY_KEY[phq_item]
            llm_count = llm_counts.get(legacy_key, 0)

            per_sample = [s[phq_item] for s in sample_outputs]
            sample_scores = tuple(a.score for a in per_sample)
            metrics = compute_consistency_metrics(sample_scores)

            selected = per_sample[0]
            if metrics.modal_confidence >= 0.5:
                selected = next((a for a in per_sample if a.score == metrics.modal_score), selected)

            na_reason: NAReason | None = None
            evidence_source: Literal["llm"] | None = "llm" if llm_count > 0 else None
            if selected.score is None and self._settings.track_na_reasons:
                na_reason = self._determine_na_reason(llm_count)
            if evidence_source is None and selected.score is not None:
                evidence_source = "llm"

            retrieval_stats = compute_retrieval_similarity_stats(
                reference_bundle.item_references.get(phq_item, []) if reference_bundle else []
            )

            final_items[phq_item] = ItemAssessment(
                item=phq_item,
                evidence=selected.evidence,
                reason=selected.reason,
                score=selected.score,
                na_reason=na_reason,
                evidence_source=evidence_source,
                llm_evidence_count=llm_count,
                retrieval_reference_count=retrieval_stats.retrieval_reference_count,
                retrieval_similarity_mean=retrieval_stats.retrieval_similarity_mean,
                retrieval_similarity_max=retrieval_stats.retrieval_similarity_max,
                verbalized_confidence=selected.verbalized_confidence,
                token_msp=selected.token_msp,
                token_pe=selected.token_pe,
                token_energy=selected.token_energy,
                consistency_modal_score=metrics.modal_score,
                consistency_modal_count=metrics.modal_count,
                consistency_modal_confidence=metrics.modal_confidence,
                consistency_score_std=metrics.score_std,
                consistency_na_rate=metrics.na_rate,
                consistency_samples=sample_scores,
            )

        return final_items

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
            model_settings: OpenAIChatModelSettings = {
                "temperature": temperature,
                # Spec 051: request logprobs to compute token-level confidence signals
                # when supported by the backend.
                "openai_logprobs": True,
                "openai_top_logprobs": _DEFAULT_TOP_LOGPROBS,
                **({"timeout": timeout} if timeout is not None else {}),
            }
            result = await self._scoring_agent.run(
                prompt,
                model_settings=model_settings,
            )
            parsed = self._from_quantitative_output(result.output)
            token_signals = self._extract_token_signals(getattr(result, "response", None))
            if token_signals is None:
                return parsed

            return {
                item: replace(
                    assessment,
                    token_msp=token_signals["token_msp"],
                    token_pe=token_signals["token_pe"],
                    token_energy=token_signals["token_energy"],
                )
                for item, assessment in parsed.items()
            }
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
    def _extract_token_signals(response: Any) -> dict[str, float] | None:
        provider_details = getattr(response, "provider_details", None)
        if not isinstance(provider_details, dict):
            return None

        logprobs = provider_details.get("logprobs")
        if not isinstance(logprobs, list) or not logprobs:
            return None

        try:
            return {
                "token_msp": compute_token_msp(logprobs),
                "token_pe": compute_token_pe(logprobs),
                "token_energy": compute_token_energy(logprobs),
            }
        except Exception as exc:
            logger.warning(
                "Failed to compute token-level confidence signals",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return None

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
                verbalized_confidence=(
                    evidence_output.confidence if evidence_output.score is not None else None
                ),
            )
        return result

    async def _extract_evidence(
        self, transcript_text: str, *, participant_id: int
    ) -> dict[str, list[str]]:
        """Extract evidence quotes for each PHQ-8 item.

        ⚠️ CRITICAL: NO SILENT FALLBACKS

        This function MUST raise on parse/schema failures. Returning empty dict {}
        on failure would violate mode isolation between zero-shot and few-shot:

        - Few-shot + empty evidence → empty references → functionally zero-shot
        - This corrupts research results without indication
        - Published comparisons between modes become invalid

        Evidence grounding failures (all extracted quotes rejected) are recorded via the
        failure registry and may be configured to raise.

        See: docs/_bugs/ANALYSIS-026-JSON-PARSING-ARCHITECTURE-AUDIT.md

        Args:
            transcript_text: Interview transcript text.
            participant_id: Transcript participant ID (for observability only).

        Returns:
            Dictionary of PHQ8 key -> list of evidence quotes.

        Raises:
            json.JSONDecodeError: If evidence JSON parsing fails. The caller
                must decide how to handle failures (retry, fail, etc.)
        """
        user_prompt = make_evidence_prompt(transcript_text)

        # Use model settings if provided (GAP-001: temp=0.0 for clinical reproducibility)
        model = get_model_name(self._model_settings, "quantitative")
        temperature = self._model_settings.temperature if self._model_settings else 0.0

        # Use format="json" to guarantee well-formed JSON at grammar level
        # See: https://docs.ollama.com/capabilities/structured-outputs
        raw = await self._llm.simple_chat(
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            format="json",
        )

        # Parse JSON response using canonical parser - NO SILENT FALLBACKS
        # If parsing fails, the exception propagates to the caller
        clean = self._strip_json_block(raw)
        obj = parse_llm_json(clean)

        try:
            evidence = validate_evidence_schema(obj)
        except EvidenceSchemaError as exc:
            logger.error(
                "evidence_schema_validation_failed",
                component="evidence_extraction",
                violations=exc.violations,
                response_hash=stable_text_hash(clean),
                response_len=len(clean),
            )
            raise

        if self._settings.evidence_quote_validation_enabled:
            evidence, stats = validate_evidence_grounding(
                evidence,
                transcript_text,
                mode=self._settings.evidence_quote_validation_mode,
                fuzzy_threshold=self._settings.evidence_quote_fuzzy_threshold,
                log_rejections=self._settings.evidence_quote_log_rejections,
            )

            if stats.validated_count == 0 and stats.extracted_count > 0:
                transcript_hash = stable_text_hash(transcript_text)
                message = (
                    "LLM returned evidence quotes but none could be grounded in the transcript."
                )
                record_failure(
                    FailureCategory.EVIDENCE_HALLUCINATION,
                    FailureSeverity.ERROR,
                    message,
                    participant_id=participant_id,
                    stage="evidence_extraction",
                    mode=self._mode.value,
                    extracted_count=stats.extracted_count,
                    validation_mode=self._settings.evidence_quote_validation_mode,
                    transcript_hash=transcript_hash,
                    transcript_len=len(transcript_text),
                )
                if self._settings.evidence_quote_fail_on_all_rejected:
                    raise EvidenceGroundingError(message)

        return evidence

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

    def _determine_na_reason(self, llm_count: int) -> NAReason:
        """Determine why an item has no score."""
        if llm_count == 0:
            return NAReason.NO_MENTION
        return NAReason.SCORE_NA_WITH_EVIDENCE

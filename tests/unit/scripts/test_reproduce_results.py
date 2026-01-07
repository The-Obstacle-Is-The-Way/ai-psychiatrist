"""Unit tests for scripts/reproduce_results.py helpers."""

from __future__ import annotations

import argparse
import importlib.util
from typing import TYPE_CHECKING, Any, cast

import pytest
from scripts.reproduce_results import (
    EvaluationResult,
    ExperimentResults,
    apply_cli_overrides,
    evaluate_participant,
    init_embedding_service,
    print_run_configuration,
)

from ai_psychiatrist.config import (
    DataSettings,
    EmbeddingBackend,
    EmbeddingBackendSettings,
    EmbeddingSettings,
    ModelSettings,
    OllamaSettings,
    Settings,
    get_settings,
)
from ai_psychiatrist.domain.entities import PHQ8Assessment, Transcript
from ai_psychiatrist.domain.enums import AssessmentMode, PHQ8Item
from ai_psychiatrist.domain.value_objects import ItemAssessment
from ai_psychiatrist.infrastructure.llm.huggingface import MissingHuggingFaceDependenciesError
from ai_psychiatrist.metrics.binary_classification import BinaryClassificationMetrics
from ai_psychiatrist.metrics.total_score import TotalScoreMetrics

if TYPE_CHECKING:
    from pathlib import Path

    from ai_psychiatrist.agents import QuantitativeAssessmentAgent
    from ai_psychiatrist.infrastructure.llm.ollama import OllamaClient
    from ai_psychiatrist.infrastructure.llm.protocols import EmbeddingClient
    from ai_psychiatrist.services import TranscriptService

pytestmark = pytest.mark.unit


def test_apply_cli_overrides_severity_inference_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEVERITY_INFERENCE_MODE", "strict")
    settings = get_settings()
    assert str(settings.quantitative.severity_inference_mode) == "strict"

    args = argparse.Namespace(
        embedding_backend=None,
        severity_inference="infer",
    )
    apply_cli_overrides(settings=settings, args=args)
    assert str(settings.quantitative.severity_inference_mode) == "infer"


def test_apply_cli_overrides_prediction_mode_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PREDICTION_MODE", "item")
    settings = get_settings()
    assert str(settings.prediction.prediction_mode) == "item"

    args = argparse.Namespace(
        embedding_backend=None,
        severity_inference=None,
        prediction_mode="total",
        total_min_coverage=None,
        binary_threshold=None,
        binary_strategy=None,
    )
    apply_cli_overrides(settings=settings, args=args)
    assert str(settings.prediction.prediction_mode) == "total"


def test_experiment_results_to_dict_includes_total_metrics_when_total_mode() -> None:
    metrics = TotalScoreMetrics(
        n_total=3,
        n_predicted=2,
        coverage=2 / 3,
        mae=1.5,
        rmse=1.0,
        pearson_r=0.9,
        severity_tier_accuracy=1.0,
    )

    participant = EvaluationResult(
        participant_id=303,
        mode="zero_shot",
        duration_seconds=0.1,
        success=True,
        ground_truth_total=14,
        predicted_total=12,
        predicted_total_min=12,
        predicted_total_max=18,
        available_items=6,
        na_items=2,
    )

    exp = ExperimentResults(
        mode="zero_shot",
        model="stub",
        prediction_mode="total",
        total_score_min_coverage=0.5,
        binary_threshold=10,
        binary_strategy="threshold",
        results=[participant],
        total_subjects=1,
        successful_subjects=1,
        failed_subjects=0,
        excluded_no_evidence=0,
        evaluated_subjects=1,
        item_mae_weighted=0.0,
        item_mae_by_item=0.0,
        item_mae_by_subject=0.0,
        prediction_coverage=0.0,
        per_item={},
        total_duration_seconds=0.1,
        total_metrics=metrics,
    )

    data = cast("dict[str, Any]", exp.to_dict())
    assert data["prediction_mode"] == "total"
    assert data["total_metrics"]["mae"] == pytest.approx(1.5)
    assert data["results"][0]["prediction_mode"] == "total"
    assert data["results"][0]["total_score"]["predicted"] == 12
    assert data["results"][0]["total_score"]["actual"] == 14


def test_experiment_results_to_dict_includes_binary_metrics_when_binary_mode() -> None:
    metrics = BinaryClassificationMetrics(
        n_total=2,
        n_predicted=2,
        coverage=1.0,
        accuracy=0.5,
        precision=0.5,
        recall=1.0,
        f1=2 / 3,
        auroc=0.75,
        confusion_matrix={
            "true_positive": 1,
            "true_negative": 0,
            "false_positive": 1,
            "false_negative": 0,
        },
    )

    participant = EvaluationResult(
        participant_id=303,
        mode="zero_shot",
        duration_seconds=0.1,
        success=True,
        ground_truth_total=14,
        predicted_total=12,
        predicted_total_min=12,
        predicted_total_max=18,
        ground_truth_binary="depressed",
        predicted_binary="not_depressed",
        binary_correct=False,
        available_items=6,
        na_items=2,
    )

    exp = ExperimentResults(
        mode="zero_shot",
        model="stub",
        prediction_mode="binary",
        total_score_min_coverage=0.5,
        binary_threshold=10,
        binary_strategy="threshold",
        results=[participant],
        total_subjects=1,
        successful_subjects=1,
        failed_subjects=0,
        excluded_no_evidence=0,
        evaluated_subjects=1,
        item_mae_weighted=0.0,
        item_mae_by_item=0.0,
        item_mae_by_subject=0.0,
        prediction_coverage=0.0,
        per_item={},
        total_duration_seconds=0.1,
        binary_metrics=metrics,
    )

    data = cast("dict[str, Any]", exp.to_dict())
    assert data["prediction_mode"] == "binary"
    assert data["binary_threshold"] == 10
    assert data["binary_strategy"] == "threshold"
    assert data["binary_metrics"]["accuracy"] == pytest.approx(0.5)
    assert data["results"][0]["binary_classification"]["predicted"] == "not_depressed"
    assert data["results"][0]["binary_classification"]["actual"] == "depressed"


@pytest.mark.asyncio
async def test_evaluate_participant_binary_mode_predicts_depressed_when_lower_bound_crosses_threshold() -> (
    None
):
    class StubTranscriptService:
        def __init__(self, text: str) -> None:
            self._text = text

        def load_transcript(self, participant_id: int) -> Transcript:
            return Transcript(participant_id=participant_id, text=self._text)

    class StubAgent:
        def __init__(self, assessment: PHQ8Assessment) -> None:
            self._assessment = assessment

        async def assess(self, _transcript: Transcript) -> PHQ8Assessment:
            return self._assessment

    participant_id = 303
    transcript_service = StubTranscriptService(text="Participant: I'm always exhausted.")

    available_items = set(PHQ8Item.all_items()[:4])
    items = {
        item: ItemAssessment(
            item=item,
            evidence="test",
            reason="test",
            score=3 if item in available_items else None,
        )
        for item in PHQ8Item.all_items()
    }
    assessment = PHQ8Assessment(
        items=items,
        mode=AssessmentMode.ZERO_SHOT,
        participant_id=participant_id,
    )
    agent = StubAgent(assessment=assessment)

    ground_truth_items = cast(
        "dict[PHQ8Item, int]",
        dict.fromkeys(PHQ8Item.all_items(), 2),
    )

    result = await evaluate_participant(
        participant_id=participant_id,
        ground_truth_items=ground_truth_items,
        agent=cast("QuantitativeAssessmentAgent", agent),
        transcript_service=cast("TranscriptService", transcript_service),
        mode="zero_shot",
        consistency_enabled=False,
        consistency_n_samples=1,
        consistency_temperature=0.0,
        prediction_mode="binary",
        total_min_coverage=0.5,
        binary_threshold=10,
        binary_strategy="threshold",
    )

    assert result.predicted_total == 12
    assert result.ground_truth_binary == "depressed"
    assert result.predicted_binary == "depressed"
    assert result.binary_correct is True


@pytest.mark.asyncio
async def test_evaluate_participant_binary_mode_predicts_not_depressed_when_upper_bound_below_threshold() -> (
    None
):
    class StubTranscriptService:
        def __init__(self, text: str) -> None:
            self._text = text

        def load_transcript(self, participant_id: int) -> Transcript:
            return Transcript(participant_id=participant_id, text=self._text)

    class StubAgent:
        def __init__(self, assessment: PHQ8Assessment) -> None:
            self._assessment = assessment

        async def assess(self, _transcript: Transcript) -> PHQ8Assessment:
            return self._assessment

    participant_id = 304
    transcript_service = StubTranscriptService(text="Participant: I'm fine.")

    available_items = set(PHQ8Item.all_items()[:7])
    items = {
        item: ItemAssessment(
            item=item,
            evidence="test",
            reason="test",
            score=0 if item in available_items else None,
        )
        for item in PHQ8Item.all_items()
    }
    assessment = PHQ8Assessment(
        items=items,
        mode=AssessmentMode.ZERO_SHOT,
        participant_id=participant_id,
    )
    agent = StubAgent(assessment=assessment)

    ground_truth_items = cast(
        "dict[PHQ8Item, int]",
        dict.fromkeys(PHQ8Item.all_items(), 0),
    )

    result = await evaluate_participant(
        participant_id=participant_id,
        ground_truth_items=ground_truth_items,
        agent=cast("QuantitativeAssessmentAgent", agent),
        transcript_service=cast("TranscriptService", transcript_service),
        mode="zero_shot",
        consistency_enabled=False,
        consistency_n_samples=1,
        consistency_temperature=0.0,
        prediction_mode="binary",
        total_min_coverage=0.5,
        binary_threshold=10,
        binary_strategy="threshold",
    )

    assert result.predicted_total == 0
    assert result.ground_truth_binary == "not_depressed"
    assert result.predicted_binary == "not_depressed"
    assert result.binary_correct is True


@pytest.mark.asyncio
async def test_evaluate_participant_binary_mode_abstains_when_bounds_straddle_threshold() -> None:
    class StubTranscriptService:
        def __init__(self, text: str) -> None:
            self._text = text

        def load_transcript(self, participant_id: int) -> Transcript:
            return Transcript(participant_id=participant_id, text=self._text)

    class StubAgent:
        def __init__(self, assessment: PHQ8Assessment) -> None:
            self._assessment = assessment

        async def assess(self, _transcript: Transcript) -> PHQ8Assessment:
            return self._assessment

    participant_id = 305
    transcript_service = StubTranscriptService(text="Participant: Sometimes I feel down.")

    available_items = set(PHQ8Item.all_items()[:4])
    items = {
        item: ItemAssessment(
            item=item,
            evidence="test",
            reason="test",
            score=0 if item in available_items else None,
        )
        for item in PHQ8Item.all_items()
    }
    assessment = PHQ8Assessment(
        items=items,
        mode=AssessmentMode.ZERO_SHOT,
        participant_id=participant_id,
    )
    agent = StubAgent(assessment=assessment)

    ground_truth_items = cast(
        "dict[PHQ8Item, int]",
        dict.fromkeys(PHQ8Item.all_items(), 2),
    )

    result = await evaluate_participant(
        participant_id=participant_id,
        ground_truth_items=ground_truth_items,
        agent=cast("QuantitativeAssessmentAgent", agent),
        transcript_service=cast("TranscriptService", transcript_service),
        mode="zero_shot",
        consistency_enabled=False,
        consistency_n_samples=1,
        consistency_temperature=0.0,
        prediction_mode="binary",
        total_min_coverage=0.5,
        binary_threshold=10,
        binary_strategy="threshold",
    )

    assert result.predicted_total == 0
    assert result.ground_truth_binary == "depressed"
    assert result.predicted_binary is None
    assert result.binary_correct is None


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["direct", "ensemble"])
async def test_evaluate_participant_binary_mode_fails_on_unimplemented_strategy(
    strategy: str,
) -> None:
    """Spec 062: direct/ensemble strategies record failure (not implemented).

    The evaluate_participant function catches exceptions and returns a failed result
    to avoid aborting full runs. This test verifies the failure is properly recorded.
    """

    class StubTranscriptService:
        def __init__(self, text: str) -> None:
            self._text = text

        def load_transcript(self, participant_id: int) -> Transcript:
            return Transcript(participant_id=participant_id, text=self._text)

    class StubAgent:
        def __init__(self, assessment: PHQ8Assessment) -> None:
            self._assessment = assessment

        async def assess(self, _transcript: Transcript) -> PHQ8Assessment:
            return self._assessment

    participant_id = 306
    transcript_service = StubTranscriptService(text="Participant: test.")

    items = {
        item: ItemAssessment(
            item=item,
            evidence="test",
            reason="test",
            score=2,
        )
        for item in PHQ8Item.all_items()
    }
    assessment = PHQ8Assessment(
        items=items,
        mode=AssessmentMode.ZERO_SHOT,
        participant_id=participant_id,
    )
    agent = StubAgent(assessment=assessment)

    ground_truth_items = cast(
        "dict[PHQ8Item, int]",
        dict.fromkeys(PHQ8Item.all_items(), 2),
    )

    result = await evaluate_participant(
        participant_id=participant_id,
        ground_truth_items=ground_truth_items,
        agent=cast("QuantitativeAssessmentAgent", agent),
        transcript_service=cast("TranscriptService", transcript_service),
        mode="zero_shot",
        consistency_enabled=False,
        consistency_n_samples=1,
        consistency_temperature=0.0,
        prediction_mode="binary",
        total_min_coverage=0.5,
        binary_threshold=10,
        binary_strategy=strategy,  # type: ignore[arg-type]
    )

    # Verify the failure is recorded properly
    assert result.success is False
    assert result.error is not None
    assert "binary_strategy is currently limited" in result.error


@pytest.mark.asyncio
async def test_evaluate_participant_includes_inference_fields_in_item_signals() -> None:
    """Inference fields should be included in output JSON (Spec 063)."""

    class StubTranscriptService:
        def __init__(self, text: str) -> None:
            self._text = text

        def load_transcript(self, participant_id: int) -> Transcript:
            return Transcript(participant_id=participant_id, text=self._text)

    class StubAgent:
        def __init__(self, assessment: PHQ8Assessment) -> None:
            self._assessment = assessment

        async def assess(self, _transcript: Transcript) -> PHQ8Assessment:
            return self._assessment

    participant_id = 300
    transcript_service = StubTranscriptService(text="Participant: I'm always exhausted.")

    items = {
        item: ItemAssessment(
            item=item,
            evidence="test",
            reason="test",
            score=0,
            inference_used=(item == PHQ8Item.TIRED),
            inference_type="intensity_marker" if item == PHQ8Item.TIRED else None,
            inference_marker="always" if item == PHQ8Item.TIRED else None,
        )
        for item in PHQ8Item.all_items()
    }
    assessment = PHQ8Assessment(
        items=items,
        mode=AssessmentMode.ZERO_SHOT,
        participant_id=participant_id,
    )
    agent = StubAgent(assessment=assessment)

    ground_truth_items = cast(
        "dict[PHQ8Item, int]",
        dict.fromkeys(PHQ8Item.all_items(), 0),
    )

    result = await evaluate_participant(
        participant_id=participant_id,
        ground_truth_items=ground_truth_items,
        agent=cast("QuantitativeAssessmentAgent", agent),
        transcript_service=cast("TranscriptService", transcript_service),
        mode="zero_shot",
        consistency_enabled=False,
        consistency_n_samples=1,
        consistency_temperature=0.0,
    )

    tired_signals = result.item_signals[PHQ8Item.TIRED]
    assert tired_signals["inference_used"] is True
    assert tired_signals["inference_type"] == "intensity_marker"
    assert tired_signals["inference_marker"] == "always"

    # Verify non-inferred items have correct default values
    interest_signals = result.item_signals[PHQ8Item.NO_INTEREST]
    assert interest_signals["inference_used"] is False
    assert interest_signals["inference_type"] is None
    assert interest_signals["inference_marker"] is None


@pytest.mark.asyncio
async def test_evaluate_participant_total_mode_abstains_below_min_coverage() -> None:
    """Total-score mode should abstain when item coverage is too low (Spec 061)."""

    class StubTranscriptService:
        def __init__(self, text: str) -> None:
            self._text = text

        def load_transcript(self, participant_id: int) -> Transcript:
            return Transcript(participant_id=participant_id, text=self._text)

    class StubAgent:
        def __init__(self, assessment: PHQ8Assessment) -> None:
            self._assessment = assessment

        async def assess(self, _transcript: Transcript) -> PHQ8Assessment:
            return self._assessment

    participant_id = 301
    transcript_service = StubTranscriptService(text="Participant: I'm exhausted.")

    available_items = set(PHQ8Item.all_items()[:3])
    items = {
        item: ItemAssessment(
            item=item,
            evidence="test",
            reason="test",
            score=1 if item in available_items else None,
        )
        for item in PHQ8Item.all_items()
    }
    assessment = PHQ8Assessment(
        items=items,
        mode=AssessmentMode.ZERO_SHOT,
        participant_id=participant_id,
    )
    agent = StubAgent(assessment=assessment)

    ground_truth_items = cast(
        "dict[PHQ8Item, int]",
        dict.fromkeys(PHQ8Item.all_items(), 0),
    )

    result = await evaluate_participant(
        participant_id=participant_id,
        ground_truth_items=ground_truth_items,
        agent=cast("QuantitativeAssessmentAgent", agent),
        transcript_service=cast("TranscriptService", transcript_service),
        mode="zero_shot",
        consistency_enabled=False,
        consistency_n_samples=1,
        consistency_temperature=0.0,
        prediction_mode="total",
        total_min_coverage=0.5,
    )

    assert result.available_items == 3
    assert result.predicted_total is None


@pytest.mark.asyncio
async def test_evaluate_participant_total_mode_predicts_at_min_coverage() -> None:
    """Total-score mode should emit a prediction at/above min coverage (Spec 061)."""

    class StubTranscriptService:
        def __init__(self, text: str) -> None:
            self._text = text

        def load_transcript(self, participant_id: int) -> Transcript:
            return Transcript(participant_id=participant_id, text=self._text)

    class StubAgent:
        def __init__(self, assessment: PHQ8Assessment) -> None:
            self._assessment = assessment

        async def assess(self, _transcript: Transcript) -> PHQ8Assessment:
            return self._assessment

    participant_id = 302
    transcript_service = StubTranscriptService(text="Participant: I'm exhausted.")

    available_items = set(PHQ8Item.all_items()[:4])
    items = {
        item: ItemAssessment(
            item=item,
            evidence="test",
            reason="test",
            score=2 if item in available_items else None,
        )
        for item in PHQ8Item.all_items()
    }
    assessment = PHQ8Assessment(
        items=items,
        mode=AssessmentMode.ZERO_SHOT,
        participant_id=participant_id,
    )
    agent = StubAgent(assessment=assessment)

    ground_truth_items = cast(
        "dict[PHQ8Item, int]",
        dict.fromkeys(PHQ8Item.all_items(), 0),
    )

    result = await evaluate_participant(
        participant_id=participant_id,
        ground_truth_items=ground_truth_items,
        agent=cast("QuantitativeAssessmentAgent", agent),
        transcript_service=cast("TranscriptService", transcript_service),
        mode="zero_shot",
        consistency_enabled=False,
        consistency_n_samples=1,
        consistency_temperature=0.0,
        prediction_mode="total",
        total_min_coverage=0.5,
    )

    assert result.available_items == 4
    assert result.predicted_total == 8


def test_print_run_configuration_displays_embedding_settings(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "data"
    base_dir.mkdir()
    transcripts_dir = base_dir / "transcripts"
    transcripts_dir.mkdir()
    embeddings_dir = base_dir / "embeddings"
    embeddings_dir.mkdir()

    embeddings_path = embeddings_dir / "embeddings.npz"
    tags_path = embeddings_path.with_suffix(".tags.json")
    tags_path.write_text("{}", encoding="utf-8")

    settings = Settings(
        enable_few_shot=False,
        data=DataSettings(
            base_dir=base_dir,
            transcripts_dir=transcripts_dir,
            embeddings_path=embeddings_path,
            train_csv=base_dir / "train.csv",
            dev_csv=base_dir / "dev.csv",
        ),
        model=ModelSettings(
            quantitative_model="gemma3:27b-it-qat",
            embedding_model="qwen3-embedding:8b",
        ),
        ollama=OllamaSettings(host="127.0.0.1", port=11434),
        embedding=EmbeddingSettings(
            enable_item_tag_filter=True,
            enable_retrieval_audit=True,
            min_reference_similarity=0.5,
            max_reference_chars_per_item=2000,
        ),
    )

    print_run_configuration(settings=settings, split="paper-test")

    output = capsys.readouterr().out
    assert f"Tags Sidecar: {tags_path} (FOUND)" in output
    assert "Item Tag Filter: True" in output
    assert "Retrieval Audit: True" in output
    assert "Min Reference Similarity: 0.5" in output
    assert "Max Reference Chars Per Item: 2000" in output


def test_init_embedding_service_fails_fast_when_hf_deps_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "data"
    base_dir.mkdir()
    transcripts_dir = base_dir / "transcripts"
    transcripts_dir.mkdir()

    def fake_find_spec(module: str, *_args: Any, **_kwargs: Any) -> object | None:
        if module == "torch":
            return None
        return object()

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    args = argparse.Namespace(zero_shot_only=False, split="paper-test")

    with pytest.raises(MissingHuggingFaceDependenciesError, match="make dev"):
        init_embedding_service(
            args=args,
            data_settings=DataSettings(base_dir=base_dir, transcripts_dir=transcripts_dir),
            embedding_backend_settings=EmbeddingBackendSettings(
                backend=EmbeddingBackend.HUGGINGFACE
            ),
            embedding_settings=EmbeddingSettings(
                embeddings_file="huggingface_qwen3_8b_paper_train_participant_only"
            ),
            model_settings=ModelSettings(),
            embedding_client=cast("EmbeddingClient", object()),
            chat_client=cast("OllamaClient", object()),
        )

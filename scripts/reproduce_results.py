#!/usr/bin/env python3
"""Reproduce paper evaluation metrics for quantitative PHQ-8 scoring.

The paper reports **item-level MAE** between predicted PHQ-8 item scores (0-3) and
ground truth, excluding items where the model outputs "N/A" due to insufficient
evidence.

This script computes:
- Item-level MAE (multiple aggregation views; see output)
- Coverage (% of item predictions that are non-N/A)
- Per-item MAE + coverage (reflecting Figure 5 discussion)

Important data note (DAIC-WOZ / AVEC2017):
- The repo includes per-item PHQ-8 labels for the AVEC2017 **train/dev** splits:
  `data/train_split_Depression_AVEC2017.csv`, `data/dev_split_Depression_AVEC2017.csv`.
- The provided **test** split CSVs do not include per-item PHQ-8 labels, so paper-
  style item-level MAE cannot be computed for test from the checked-in files.

Paper Reference:
    - Section 3.2: quantitative assessment + MAE definition
    - Figure 4/5: per-item prediction performance and variability

Usage:
    # Evaluate dev split (has per-item labels)
    python scripts/reproduce_results.py --split dev

    # Train split (has per-item labels)
    python scripts/reproduce_results.py --split train

    # Combined train+dev (matches paper's 142-participant qualitative figures)
    python scripts/reproduce_results.py --split train+dev

    # Dry run to check configuration
    python scripts/reproduce_results.py --dry-run

    # Limit to N participants (for testing)
    python scripts/reproduce_results.py --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_psychiatrist.agents import QuantitativeAssessmentAgent
from ai_psychiatrist.config import (
    DataSettings,
    EmbeddingBackend,
    EmbeddingSettings,
    LoggingSettings,
    ModelSettings,
    Settings,
    get_settings,
)
from ai_psychiatrist.domain.enums import AssessmentMode, PHQ8Item
from ai_psychiatrist.infrastructure.llm import OllamaClient
from ai_psychiatrist.infrastructure.llm.factory import create_embedding_client
from ai_psychiatrist.infrastructure.logging import get_logger, setup_logging
from ai_psychiatrist.services import EmbeddingService, ReferenceStore, TranscriptService

if TYPE_CHECKING:
    from ai_psychiatrist.infrastructure.llm.protocols import EmbeddingClient

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Result for a single participant evaluation."""

    participant_id: int
    mode: str
    duration_seconds: float
    success: bool
    error: str | None = None

    ground_truth_items: dict[PHQ8Item, int] = field(default_factory=dict)
    predicted_items: dict[PHQ8Item, int | None] = field(default_factory=dict)
    ground_truth_total: int | None = None
    predicted_total: int | None = None

    available_items: int = 0
    na_items: int = 0
    mae_available: float | None = None


@dataclass
class ExperimentResults:
    """Aggregate results for an experiment."""

    mode: str
    model: str
    results: list[EvaluationResult]
    total_subjects: int
    successful_subjects: int
    failed_subjects: int
    excluded_no_evidence: int
    evaluated_subjects: int

    item_mae_weighted: float
    item_mae_by_item: float
    item_mae_by_subject: float
    prediction_coverage: float
    per_item: dict[PHQ8Item, dict[str, float | int | None]]

    total_duration_seconds: float

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization."""
        return {
            "mode": self.mode,
            "model": self.model,
            "total_subjects": self.total_subjects,
            "successful_subjects": self.successful_subjects,
            "failed_subjects": self.failed_subjects,
            "excluded_no_evidence": self.excluded_no_evidence,
            "evaluated_subjects": self.evaluated_subjects,
            "item_mae_weighted": round(self.item_mae_weighted, 6),
            "item_mae_by_item": round(self.item_mae_by_item, 6),
            "item_mae_by_subject": round(self.item_mae_by_subject, 6),
            "prediction_coverage": round(self.prediction_coverage, 6),
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "per_item": {item.value: stats for item, stats in self.per_item.items()},
            "results": [
                {
                    "participant_id": r.participant_id,
                    "success": r.success,
                    "error": r.error,
                    "ground_truth_total": r.ground_truth_total,
                    "predicted_total": r.predicted_total,
                    "available_items": r.available_items,
                    "na_items": r.na_items,
                    "mae_available": r.mae_available,
                    "ground_truth_items": {
                        item.value: score for item, score in r.ground_truth_items.items()
                    },
                    "predicted_items": {
                        item.value: score for item, score in r.predicted_items.items()
                    },
                }
                for r in self.results
            ],
        }


def _split_csv_path(data_dir: Path, split: str) -> Path:
    """Resolve split CSV path.

    Supported split families:
    - AVEC2017 official: train/dev
    - Paper custom: paper-train/paper-val/paper-test (and paper alias for paper-test)
    """
    if split in {"train", "dev"}:
        mapping = {
            "train": data_dir / "train_split_Depression_AVEC2017.csv",
            "dev": data_dir / "dev_split_Depression_AVEC2017.csv",
        }
        return mapping[split]

    if split in {"paper", "paper-train", "paper-val", "paper-test"}:
        name = "test" if split == "paper" else split.removeprefix("paper-")
        return data_dir / "paper_splits" / f"paper_split_{name}.csv"

    raise ValueError(f"Unsupported split: {split}")


def load_item_ground_truth_csv(csv_path: Path) -> dict[int, dict[PHQ8Item, int]]:
    """Load per-item PHQ-8 ground truth from a CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Split not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["Participant_ID"] = df["Participant_ID"].astype(int)

    result: dict[int, dict[PHQ8Item, int]] = {}
    for _, row in df.iterrows():
        participant_id = int(row["Participant_ID"])
        scores: dict[PHQ8Item, int] = {}
        for item in PHQ8Item.all_items():
            col = f"PHQ8_{item.value}"
            scores[item] = int(row[col])
        result[participant_id] = scores

    return result


def load_item_ground_truth(data_dir: Path, split: str) -> dict[int, dict[PHQ8Item, int]]:
    """Load per-item PHQ-8 ground truth for a split.

    Args:
        data_dir: Path to data directory.
        split: "train" or "dev".

    Returns:
        Mapping from participant_id to per-item ground truth scores (0-3).
    """
    csv_path = _split_csv_path(data_dir, split)
    return load_item_ground_truth_csv(csv_path)


def load_total_ground_truth_test(data_dir: Path) -> dict[int, int]:
    """Load total PHQ-8 scores for the test split.

    This is NOT the paper's MAE metric (paper uses item-level MAE excluding N/A),
    but it can be useful as an additional sanity check.
    """
    test_csv = data_dir / "full_test_split.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test split not found: {test_csv}")

    df = pd.read_csv(test_csv)
    return dict(zip(df["Participant_ID"].astype(int), df["PHQ_Score"].astype(int), strict=True))


async def evaluate_participant(
    participant_id: int,
    ground_truth_items: dict[PHQ8Item, int],
    agent: QuantitativeAssessmentAgent,
    transcript_service: TranscriptService,
    mode: str,
) -> EvaluationResult:
    """Evaluate a single participant.

    Args:
        participant_id: DAIC-WOZ participant ID.
        ground_truth_items: Ground truth PHQ-8 per-item scores.
        agent: Quantitative assessment agent.
        transcript_service: Transcript loading service.
        mode: Assessment mode name.

    Returns:
        EvaluationResult with prediction and metrics.
    """
    start = time.perf_counter()

    try:
        transcript = transcript_service.load_transcript(participant_id)
        assessment = await agent.assess(transcript)
        predicted_items = {item: assessment.items[item].score for item in PHQ8Item.all_items()}
        errors: list[int] = []
        for item in PHQ8Item.all_items():
            pred = predicted_items[item]
            if pred is None:
                continue
            errors.append(abs(pred - ground_truth_items[item]))
        mae_available = float(np.mean(errors)) if errors else None

        duration = time.perf_counter() - start
        return EvaluationResult(
            participant_id=participant_id,
            mode=mode,
            duration_seconds=duration,
            success=True,
            ground_truth_items=ground_truth_items,
            predicted_items=predicted_items,
            ground_truth_total=sum(ground_truth_items.values()),
            predicted_total=assessment.total_score,
            available_items=assessment.available_count,
            na_items=assessment.na_count,
            mae_available=mae_available,
        )

    except Exception as e:
        duration = time.perf_counter() - start
        logger.error(
            "Evaluation failed",
            participant_id=participant_id,
            error=str(e),
        )
        return EvaluationResult(
            participant_id=participant_id,
            mode=mode,
            duration_seconds=duration,
            success=False,
            error=str(e),
        )


def compute_item_level_metrics(
    results: list[EvaluationResult],
) -> tuple[
    int,
    int,
    int,
    float,
    float,
    float,
    float,
    dict[PHQ8Item, dict[str, float | int | None]],
]:
    """Compute paper-style quantitative metrics.

    Paper metric: item-level MAE on predicted scores, excluding "N/A".

    Returns:
        - successful_subjects: Number of participants with a completed run.
        - excluded_no_evidence: Successful participants with 0 available items.
        - evaluated_subjects: Successful participants with >=1 available item.
        - item_mae_weighted: Mean error across all predicted items (weighted by count).
        - item_mae_by_item: Mean of per-item MAEs (each item equally weighted).
        - item_mae_by_subject: Mean of per-subject MAE on available items.
        - prediction_coverage: Predicted items / (evaluated_subjects * 8).
        - per_item: Per-item MAE + counts + coverage.
    """
    successful = [r for r in results if r.success]
    successful_subjects = len(successful)

    excluded_no_evidence = sum(1 for r in successful if r.available_items == 0)
    evaluated = [r for r in successful if r.available_items > 0]
    evaluated_subjects = len(evaluated)

    per_item_errors: dict[PHQ8Item, list[int]] = {item: [] for item in PHQ8Item.all_items()}
    all_errors: list[int] = []
    subject_maes: list[float] = []

    for r in evaluated:
        if r.mae_available is not None:
            subject_maes.append(r.mae_available)
        for item in PHQ8Item.all_items():
            pred = r.predicted_items.get(item)
            if pred is None:
                continue
            err = abs(pred - r.ground_truth_items[item])
            per_item_errors[item].append(err)
            all_errors.append(err)

    item_mae_weighted = float(np.mean(all_errors)) if all_errors else float("nan")
    per_item_maes = [float(np.mean(errors)) for errors in per_item_errors.values() if errors]
    item_mae_by_item = float(np.mean(per_item_maes)) if per_item_maes else float("nan")
    item_mae_by_subject = float(np.mean(subject_maes)) if subject_maes else float("nan")

    total_predictions = len(all_errors)
    prediction_coverage = (
        total_predictions / (evaluated_subjects * 8) if evaluated_subjects else float("nan")
    )

    per_item: dict[PHQ8Item, dict[str, float | int | None]] = {}
    for item, errors in per_item_errors.items():
        count = len(errors)
        per_item[item] = {
            "mae": float(np.mean(errors)) if errors else None,
            "count": count,
            "coverage": (count / evaluated_subjects) if evaluated_subjects else float("nan"),
            "na_count": (evaluated_subjects - count) if evaluated_subjects else 0,
        }

    return (
        successful_subjects,
        excluded_no_evidence,
        evaluated_subjects,
        item_mae_weighted,
        item_mae_by_item,
        item_mae_by_subject,
        prediction_coverage,
        per_item,
    )


async def run_experiment(
    mode: AssessmentMode,
    ground_truth: dict[int, dict[PHQ8Item, int]],
    ollama_client: OllamaClient,
    transcript_service: TranscriptService,
    embedding_service: EmbeddingService | None,
    model_name: str,
    limit: int | None = None,
) -> ExperimentResults:
    """Run evaluation experiment for a given mode.

    Args:
        mode: Zero-shot or few-shot mode.
        ground_truth: Ground truth per-item PHQ-8 scores.
        ollama_client: Ollama LLM client.
        transcript_service: Transcript loading service.
        embedding_service: Embedding service (required for few-shot).
        model_name: Model name for logging.
        limit: Optional limit on number of participants.

    Returns:
        ExperimentResults with aggregate metrics.
    """
    settings = get_settings()

    agent = QuantitativeAssessmentAgent(
        llm_client=ollama_client,
        embedding_service=embedding_service if mode == AssessmentMode.FEW_SHOT else None,
        mode=mode,
        model_settings=settings.model,
        quantitative_settings=settings.quantitative,
    )

    participant_ids = list(ground_truth.keys())
    if limit:
        participant_ids = participant_ids[:limit]

    mode_name = mode.value
    print(f"\n{'=' * 60}")
    print(f"RUNNING {mode_name.upper()} EVALUATION")
    print(f"{'=' * 60}")
    print(f"  Model: {model_name}")
    print(f"  Participants: {len(participant_ids)}")
    print()

    results: list[EvaluationResult] = []
    start_time = time.perf_counter()

    for idx, pid in enumerate(participant_ids, 1):
        gt_items = ground_truth[pid]
        result = await evaluate_participant(
            participant_id=pid,
            ground_truth_items=gt_items,
            agent=agent,
            transcript_service=transcript_service,
            mode=mode_name,
        )
        results.append(result)

        # Progress update every 5 participants or on completion
        if idx % 5 == 0 or idx == len(participant_ids):
            successful = sum(1 for r in results if r.success)
            print(f"  Progress: {idx}/{len(participant_ids)} (success: {successful})")

    total_duration = time.perf_counter() - start_time
    failed_subjects = sum(1 for r in results if not r.success)
    (
        successful_subjects,
        excluded_no_evidence,
        evaluated_subjects,
        item_mae_weighted,
        item_mae_by_item,
        item_mae_by_subject,
        prediction_coverage,
        per_item,
    ) = compute_item_level_metrics(results)

    return ExperimentResults(
        mode=mode_name,
        model=model_name,
        results=results,
        total_subjects=len(participant_ids),
        successful_subjects=successful_subjects,
        failed_subjects=failed_subjects,
        excluded_no_evidence=excluded_no_evidence,
        evaluated_subjects=evaluated_subjects,
        item_mae_weighted=item_mae_weighted,
        item_mae_by_item=item_mae_by_item,
        item_mae_by_subject=item_mae_by_subject,
        prediction_coverage=prediction_coverage,
        per_item=per_item,
        total_duration_seconds=total_duration,
    )


def print_summary(experiments: list[ExperimentResults]) -> None:
    """Print summary of all experiments."""
    print("\n" + "=" * 70)
    print("REPRODUCTION RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(
        f"{'Mode':<12} {'Model':<22} {'N_eval':>6} {'Cov%':>7} "
        f"{'MAE_w':>8} {'MAE_i':>8} {'MAE_s':>8} {'Time':>10}"
    )
    print("-" * 70)

    for exp in experiments:
        time_str = f"{exp.total_duration_seconds:.1f}s"
        print(
            f"{exp.mode:<12} {exp.model:<22} {exp.evaluated_subjects:>6} "
            f"{(exp.prediction_coverage * 100):>6.1f}% "
            f"{exp.item_mae_weighted:>8.4f} {exp.item_mae_by_item:>8.4f} "
            f"{exp.item_mae_by_subject:>8.4f} {time_str:>10}"
        )

    print("-" * 70)
    print()
    print("Paper Reference (Section 3.2, test set, item-level MAE excluding N/A):")
    print("  Gemma 3 27B few-shot MAE: 0.619")
    print("  Gemma 3 27B zero-shot MAE: 0.796")
    print("  MedGemma 27B few-shot MAE: 0.505 (Appendix F; fewer predictions)")
    print()


def save_results(
    experiments: list[ExperimentResults],
    output_dir: Path,
    *,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Save results to JSON file with provenance tracking.

    Args:
        experiments: List of experiment results.
        output_dir: Directory to save results.
        provenance: Optional provenance metadata (embeddings path, split, config).

    Returns:
        Path to saved results file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"reproduction_results_{timestamp}.json"

    data: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "experiments": [exp.to_dict() for exp in experiments],
    }

    # Add provenance for reproducibility verification (BUG-023 fix)
    if provenance:
        data["provenance"] = provenance

    with output_file.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to: {output_file}")
    return output_file


def print_run_configuration(*, settings: Settings, split: str) -> None:
    """Print run configuration header."""
    data_settings = settings.data
    model_settings = settings.model
    ollama_settings = settings.ollama

    embeddings_path = data_settings.embeddings_path
    if split.startswith("paper"):
        embeddings_path = data_settings.base_dir / "embeddings" / "paper_reference_embeddings.npz"

    print("=" * 60)
    print("PAPER REPRODUCTION: Quantitative PHQ-8 Evaluation (Item-level MAE)")
    print("=" * 60)
    print(f"  Ollama: {ollama_settings.base_url}")
    print(f"  Quantitative Model: {model_settings.quantitative_model}")
    print(f"  Embedding Model: {model_settings.embedding_model}")
    print(f"  Embeddings Artifact: {embeddings_path}")
    print(f"  Data Directory: {data_settings.base_dir}")
    print(f"  Split: {split}")
    print("=" * 60)


def load_ground_truth_for_split(data_dir: Path, *, split: str) -> dict[int, dict[PHQ8Item, int]]:
    """Load per-item ground truth for a split selection."""
    if split == "train+dev":
        gt = load_item_ground_truth(data_dir, "train")
        gt.update(load_item_ground_truth(data_dir, "dev"))
        return gt
    return load_item_ground_truth(data_dir, split)


async def check_ollama_connectivity(ollama_client: OllamaClient) -> bool:
    """Check Ollama connectivity and return health status."""
    print("\nChecking Ollama connectivity...")
    try:
        is_healthy = await ollama_client.ping()
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama: {e}")
        return False

    if not is_healthy:
        print("ERROR: Ollama is not responding")
        return False

    print("  Ollama: OK")
    return True


def get_effective_embeddings_path(data_settings: DataSettings, split: str) -> Path:
    """Get the effective embeddings path for a given split.

    Paper splits use paper_reference_embeddings.npz; others use config default.
    """
    if split.startswith("paper"):
        return data_settings.base_dir / "embeddings" / "paper_reference_embeddings.npz"
    return data_settings.embeddings_path


def init_embedding_service(
    *,
    args: argparse.Namespace,
    data_settings: DataSettings,
    embedding_settings: EmbeddingSettings,
    model_settings: ModelSettings,
    embedding_client: EmbeddingClient,
) -> EmbeddingService | None:
    """Initialize embedding service for few-shot mode (or return None)."""
    if args.zero_shot_only:
        return None

    npz_path = get_effective_embeddings_path(data_settings, args.split)
    effective_data_settings = DataSettings(
        base_dir=data_settings.base_dir,
        transcripts_dir=data_settings.transcripts_dir,
        embeddings_path=npz_path,
        train_csv=data_settings.train_csv,
        dev_csv=data_settings.dev_csv,
    )

    json_path = npz_path.with_suffix(".json")
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            "Few-shot evaluation requires a precomputed embeddings artifact. "
            f"Missing: {npz_path} and/or {json_path}. "
            "Run: uv run python scripts/generate_embeddings.py"
        )
    reference_store = ReferenceStore(effective_data_settings, embedding_settings)
    return EmbeddingService(
        llm_client=embedding_client,
        reference_store=reference_store,
        settings=embedding_settings,
        model_settings=model_settings,
    )


async def run_requested_experiments(
    *,
    args: argparse.Namespace,
    ground_truth: dict[int, dict[PHQ8Item, int]],
    ollama_client: OllamaClient,
    transcript_service: TranscriptService,
    embedding_service: EmbeddingService | None,
    model_name: str,
) -> list[ExperimentResults]:
    """Run experiments based on CLI flags."""
    experiments: list[ExperimentResults] = []

    if not args.few_shot_only:
        experiments.append(
            await run_experiment(
                mode=AssessmentMode.ZERO_SHOT,
                ground_truth=ground_truth,
                ollama_client=ollama_client,
                transcript_service=transcript_service,
                embedding_service=None,
                model_name=model_name,
                limit=args.limit,
            )
        )

    if not args.zero_shot_only:
        experiments.append(
            await run_experiment(
                mode=AssessmentMode.FEW_SHOT,
                ground_truth=ground_truth,
                ollama_client=ollama_client,
                transcript_service=transcript_service,
                embedding_service=embedding_service,
                model_name=model_name,
                limit=args.limit,
            )
        )

    return experiments


async def main_async(args: argparse.Namespace) -> int:
    """Async main entry point."""
    setup_logging(LoggingSettings(level="INFO", format="console"))

    settings = get_settings()

    # Override embedding backend if specified
    if args.embedding_backend:
        settings.embedding_backend.backend = EmbeddingBackend(args.embedding_backend)

    data_settings = settings.data
    model_settings = settings.model
    ollama_settings = settings.ollama
    embedding_settings = settings.embedding
    embedding_backend_settings = settings.embedding_backend

    print_run_configuration(settings=settings, split=args.split)
    print(f"  Embedding Backend: {embedding_backend_settings.backend.value}")

    if args.dry_run:
        print("\n[DRY RUN] Would run evaluation with above settings.")
        return 0

    try:
        ground_truth = load_ground_truth_for_split(data_settings.base_dir, split=args.split)
        print(f"\nLoaded {len(ground_truth)} participants with per-item ground truth")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return 1

    # Initialize services
    # Chat client (Ollama)
    async with OllamaClient(ollama_settings) as ollama_client:
        if not await check_ollama_connectivity(ollama_client):
            return 1

        transcript_service = TranscriptService(data_settings)

        # Embedding client (Factory)
        embedding_client = create_embedding_client(settings)
        try:
            try:
                embedding_service = init_embedding_service(
                    args=args,
                    data_settings=data_settings,
                    embedding_settings=embedding_settings,
                    model_settings=model_settings,
                    embedding_client=embedding_client,
                )
            except FileNotFoundError as e:
                print(f"\nERROR: {e}")
                return 1

            experiments = await run_requested_experiments(
                args=args,
                ground_truth=ground_truth,
                ollama_client=ollama_client,
                transcript_service=transcript_service,
                embedding_service=embedding_service,
                model_name=model_settings.quantitative_model,
            )

            # Print summary and save with provenance (BUG-023 fix)
            print_summary(experiments)
            output_dir = data_settings.base_dir / "outputs"

            # Build provenance metadata for reproducibility verification
            embeddings_path = get_effective_embeddings_path(data_settings, args.split)
            provenance = {
                "split": args.split,
                "embeddings_path": str(embeddings_path),
                "embedding_backend": embedding_backend_settings.backend.value,
                "quantitative_model": model_settings.quantitative_model,
                "embedding_model": model_settings.embedding_model,
                "embedding_dimension": embedding_settings.dimension,
                "embedding_chunk_size": embedding_settings.chunk_size,
                "embedding_chunk_step": embedding_settings.chunk_step,
                "embedding_top_k": embedding_settings.top_k_references,
                "enable_keyword_backfill": settings.quantitative.enable_keyword_backfill,
                "participants_evaluated": list(ground_truth.keys()),
            }

            save_results(experiments, output_dir, provenance=provenance)

        finally:
            await embedding_client.close()

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reproduce paper results for PHQ-8 assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full reproduction (zero-shot and few-shot)
    python scripts/reproduce_results.py

    # Quick test with 5 participants
    python scripts/reproduce_results.py --limit 5

        # Zero-shot only (faster)
        python scripts/reproduce_results.py --zero-shot-only

        # Few-shot only
        python scripts/reproduce_results.py --few-shot-only

        # Use HuggingFace for embeddings
        python scripts/reproduce_results.py --embedding-backend huggingface        """,
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "train+dev", "paper", "paper-train", "paper-val", "paper-test"],
        default="dev",
        help=(
            "Which split to evaluate. Paper split requires running "
            "scripts/create_paper_split.py first."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without running evaluation",
    )
    parser.add_argument(
        "--zero-shot-only",
        action="store_true",
        help="Only run zero-shot evaluation",
    )
    parser.add_argument(
        "--few-shot-only",
        action="store_true",
        help="Only run few-shot evaluation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to N participants (for testing)",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["ollama", "huggingface"],
        default=None,
        help="Override embedding backend",
    )
    args = parser.parse_args()

    if args.zero_shot_only and args.few_shot_only:
        print("ERROR: Cannot specify both --zero-shot-only and --few-shot-only")
        return 1

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Reproduce paper results: Zero-shot vs Few-shot PHQ-8 assessment.

This script runs the full evaluation pipeline on the DAIC-WOZ test split,
comparing predicted PHQ-8 scores to ground truth to compute MAE.

Paper Reference:
    - Section 3.2: Zero-shot vs few-shot performance
    - Table 3: MAE comparison across modes and models

Usage:
    # Run both zero-shot and few-shot (recommended)
    python scripts/reproduce_results.py

    # Zero-shot only (faster, no embeddings needed)
    python scripts/reproduce_results.py --zero-shot-only

    # Few-shot only
    python scripts/reproduce_results.py --few-shot-only

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_psychiatrist.agents import QuantitativeAssessmentAgent
from ai_psychiatrist.config import LoggingSettings, get_settings
from ai_psychiatrist.domain.enums import AssessmentMode
from ai_psychiatrist.infrastructure.llm import OllamaClient
from ai_psychiatrist.infrastructure.logging import get_logger, setup_logging
from ai_psychiatrist.services import EmbeddingService, ReferenceStore, TranscriptService

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Result for a single participant evaluation."""

    participant_id: int
    ground_truth: int
    predicted: int
    absolute_error: int
    mode: str
    duration_seconds: float
    success: bool
    error: str | None = None


@dataclass
class ExperimentResults:
    """Aggregate results for an experiment."""

    mode: str
    model: str
    results: list[EvaluationResult]
    total_participants: int
    successful_count: int
    failed_count: int
    mae: float
    rmse: float
    median_ae: float
    total_duration_seconds: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "mode": self.mode,
            "model": self.model,
            "total_participants": self.total_participants,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "median_ae": round(self.median_ae, 4),
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "results": [
                {
                    "participant_id": r.participant_id,
                    "ground_truth": r.ground_truth,
                    "predicted": r.predicted,
                    "absolute_error": r.absolute_error,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


def load_test_ground_truth(data_dir: Path) -> dict[int, int]:
    """Load ground truth PHQ-8 scores for test split.

    Args:
        data_dir: Path to data directory.

    Returns:
        Dictionary mapping participant_id to PHQ_Score.
    """
    test_csv = data_dir / "full_test_split.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test split not found: {test_csv}")

    df = pd.read_csv(test_csv)
    return dict(zip(df["Participant_ID"].astype(int), df["PHQ_Score"].astype(int), strict=True))


async def evaluate_participant(
    participant_id: int,
    ground_truth: int,
    agent: QuantitativeAssessmentAgent,
    transcript_service: TranscriptService,
    mode: str,
) -> EvaluationResult:
    """Evaluate a single participant.

    Args:
        participant_id: DAIC-WOZ participant ID.
        ground_truth: Ground truth PHQ-8 score.
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
        predicted = assessment.total_score
        absolute_error = abs(predicted - ground_truth)

        duration = time.perf_counter() - start
        return EvaluationResult(
            participant_id=participant_id,
            ground_truth=ground_truth,
            predicted=predicted,
            absolute_error=absolute_error,
            mode=mode,
            duration_seconds=duration,
            success=True,
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
            ground_truth=ground_truth,
            predicted=-1,
            absolute_error=-1,
            mode=mode,
            duration_seconds=duration,
            success=False,
            error=str(e),
        )


def compute_metrics(results: list[EvaluationResult]) -> tuple[float, float, float]:
    """Compute MAE, RMSE, and Median AE from results.

    Args:
        results: List of evaluation results.

    Returns:
        Tuple of (MAE, RMSE, Median AE).
    """
    successful = [r for r in results if r.success]
    if not successful:
        return float("nan"), float("nan"), float("nan")

    errors = [r.absolute_error for r in successful]
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean([e**2 for e in errors])))
    median_ae = float(np.median(errors))

    return mae, rmse, median_ae


async def run_experiment(
    mode: AssessmentMode,
    ground_truth: dict[int, int],
    ollama_client: OllamaClient,
    transcript_service: TranscriptService,
    embedding_service: EmbeddingService | None,
    model_name: str,
    limit: int | None = None,
) -> ExperimentResults:
    """Run evaluation experiment for a given mode.

    Args:
        mode: Zero-shot or few-shot mode.
        ground_truth: Ground truth PHQ-8 scores.
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
    )

    participant_ids = list(ground_truth.keys())
    if limit:
        participant_ids = participant_ids[:limit]

    mode_name = mode.value
    print(f"\n{'='*60}")
    print(f"RUNNING {mode_name.upper()} EVALUATION")
    print(f"{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  Participants: {len(participant_ids)}")
    print()

    results: list[EvaluationResult] = []
    start_time = time.perf_counter()

    for idx, pid in enumerate(participant_ids, 1):
        gt = ground_truth[pid]
        result = await evaluate_participant(
            participant_id=pid,
            ground_truth=gt,
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
    mae, rmse, median_ae = compute_metrics(results)
    successful_count = sum(1 for r in results if r.success)

    return ExperimentResults(
        mode=mode_name,
        model=model_name,
        results=results,
        total_participants=len(participant_ids),
        successful_count=successful_count,
        failed_count=len(participant_ids) - successful_count,
        mae=mae,
        rmse=rmse,
        median_ae=median_ae,
        total_duration_seconds=total_duration,
    )


def print_summary(experiments: list[ExperimentResults]) -> None:
    """Print summary of all experiments."""
    print("\n" + "=" * 70)
    print("REPRODUCTION RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Mode':<15} {'Model':<25} {'N':>5} {'MAE':>8} {'RMSE':>8} {'Time':>10}")
    print("-" * 70)

    for exp in experiments:
        time_str = f"{exp.total_duration_seconds:.1f}s"
        print(
            f"{exp.mode:<15} {exp.model:<25} {exp.successful_count:>5} "
            f"{exp.mae:>8.4f} {exp.rmse:>8.4f} {time_str:>10}"
        )

    print("-" * 70)
    print()
    print("Paper Reference (Table 3, Llama 3.3 70B on DAIC-WOZ):")
    print("  Zero-shot MAE: ~4.47 (reported)")
    print("  Few-shot MAE:  ~3.76 (reported, 18% improvement)")
    print()


def save_results(experiments: list[ExperimentResults], output_dir: Path) -> Path:
    """Save results to JSON file.

    Args:
        experiments: List of experiment results.
        output_dir: Directory to save results.

    Returns:
        Path to saved results file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"reproduction_results_{timestamp}.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "experiments": [exp.to_dict() for exp in experiments],
    }

    with output_file.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to: {output_file}")
    return output_file


async def main_async(args: argparse.Namespace) -> int:
    """Async main entry point."""
    setup_logging(LoggingSettings(level="INFO", format="console"))

    settings = get_settings()
    data_settings = settings.data
    model_settings = settings.model
    ollama_settings = settings.ollama
    embedding_settings = settings.embedding

    print("=" * 60)
    print("PAPER REPRODUCTION: PHQ-8 Assessment Evaluation")
    print("=" * 60)
    print(f"  Ollama: {ollama_settings.base_url}")
    print(f"  Quantitative Model: {model_settings.quantitative_model}")
    print(f"  Embedding Model: {model_settings.embedding_model}")
    print(f"  Data Directory: {data_settings.base_dir}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would run evaluation with above settings.")
        return 0

    # Load ground truth
    try:
        ground_truth = load_test_ground_truth(data_settings.base_dir)
        print(f"\nLoaded {len(ground_truth)} test participants with ground truth scores")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return 1

    # Initialize services
    async with OllamaClient(ollama_settings) as ollama_client:
        # Check connectivity
        print("\nChecking Ollama connectivity...")
        try:
            is_healthy = await ollama_client.ping()
            if not is_healthy:
                print("ERROR: Ollama is not responding")
                return 1
            print("  Ollama: OK")
        except Exception as e:
            print(f"ERROR: Cannot connect to Ollama: {e}")
            return 1

        transcript_service = TranscriptService(data_settings)

        # Initialize embedding service for few-shot
        embedding_service: EmbeddingService | None = None
        if not args.zero_shot_only:
            reference_store = ReferenceStore(data_settings, embedding_settings)
            embedding_service = EmbeddingService(
                llm_client=ollama_client,
                reference_store=reference_store,
                settings=embedding_settings,
                model_settings=model_settings,
            )

        experiments: list[ExperimentResults] = []

        # Run zero-shot if requested
        if not args.few_shot_only:
            exp = await run_experiment(
                mode=AssessmentMode.ZERO_SHOT,
                ground_truth=ground_truth,
                ollama_client=ollama_client,
                transcript_service=transcript_service,
                embedding_service=None,
                model_name=model_settings.quantitative_model,
                limit=args.limit,
            )
            experiments.append(exp)

        # Run few-shot if requested
        if not args.zero_shot_only:
            exp = await run_experiment(
                mode=AssessmentMode.FEW_SHOT,
                ground_truth=ground_truth,
                ollama_client=ollama_client,
                transcript_service=transcript_service,
                embedding_service=embedding_service,
                model_name=model_settings.quantitative_model,
                limit=args.limit,
            )
            experiments.append(exp)

        # Print summary and save
        print_summary(experiments)
        output_dir = data_settings.base_dir / "outputs"
        save_results(experiments, output_dir)

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
        """,
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
    args = parser.parse_args()

    if args.zero_shot_only and args.few_shot_only:
        print("ERROR: Cannot specify both --zero-shot-only and --few-shot-only")
        return 1

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())

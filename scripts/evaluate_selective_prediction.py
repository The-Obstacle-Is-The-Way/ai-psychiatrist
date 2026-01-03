#!/usr/bin/env python3
"""Evaluate selective prediction metrics (AURC/AUGRC) from run outputs.

Implements Spec 25 Phase 4.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_psychiatrist.calibration.calibrators import (
    IsotonicCalibrator,
    LinearCalibrator,
    LogisticCalibrator,
    StandardScalerParams,
    TemperatureScalingCalibrator,
)
from ai_psychiatrist.calibration.feature_extraction import CalibratorFeatureExtractor
from ai_psychiatrist.confidence.csf_registry import get_csf, parse_csf_variant
from ai_psychiatrist.domain.enums import PHQ8Item
from ai_psychiatrist.metrics.bootstrap import (
    bootstrap_by_participant,
    paired_bootstrap_delta_by_participant,
)
from ai_psychiatrist.metrics.selective_prediction import (
    ItemPrediction,
    compute_augrc,
    compute_augrc_at_coverage,
    compute_augrc_optimal,
    compute_aurc,
    compute_aurc_achievable,
    compute_aurc_at_coverage,
    compute_aurc_optimal,
    compute_cmax,
    compute_eaugrc,
    compute_eaurc,
    compute_risk_at_coverage,
    compute_risk_coverage_curve,
)

# Constants
DEFAULT_COVERAGE_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_AREA_COVERAGE = 0.5
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 42

CONFIDENCE_VARIANTS = {
    "llm",
    "total_evidence",
    "retrieval_similarity_mean",
    "retrieval_similarity_max",
    "hybrid_evidence_similarity",
    # Spec 048: verbalized confidence
    "verbalized",
    "verbalized_calibrated",
    "hybrid_verbalized",
    # Spec 051: token-level CSFs (requires logprobs in run artifacts)
    "token_msp",
    "token_pe",
    "token_energy",
    # Spec 050: consistency-based confidence
    "consistency",
    "consistency_inverse_std",
    "hybrid_consistency",
    # Spec 049: supervised calibrator
    "calibrated",
}

CONFIDENCE_DEFAULT_VARIANTS = ["llm", "total_evidence"]


@dataclass
class InputConfig:
    path: Path
    mode: str | None = None


def parse_coverage_grid(s: str) -> list[float]:
    try:
        values = [float(x.strip()) for x in s.split(",")]
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"Invalid coverage grid: {s}") from err
    if not values:
        raise argparse.ArgumentTypeError("Coverage grid must contain at least one value")
    invalid = [v for v in values if not 0.0 < v <= 1.0]
    if invalid:
        raise argparse.ArgumentTypeError(f"Coverage values must be in (0, 1], got {invalid}")
    return values


def load_run_data(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return cast("dict[str, Any]", json.load(f))


def extract_experiment(
    run_data: dict[str, Any], mode_request: str | None, path_str: str
) -> dict[str, Any]:
    experiments = cast("list[dict[str, Any]]", run_data.get("experiments", []))
    if not experiments:
        raise ValueError(f"No experiments found in {path_str}")

    if mode_request:
        # Filter by mode in provenance or top-level
        matches = [
            exp
            for exp in experiments
            if exp.get("provenance", {}).get("mode") == mode_request
            or exp.get("results", {}).get("mode") == mode_request
        ]
        if not matches:
            available = [
                exp.get("provenance", {}).get("mode") or exp.get("results", {}).get("mode")
                for exp in experiments
            ]
            raise ValueError(
                f"Mode '{mode_request}' not found in {path_str}. Available: {available}"
            )
        if len(matches) > 1:
            # Ambiguous? Usually one experiment per mode.
            pass
        return matches[0]
    # No mode requested. If only 1, take it.
    elif len(experiments) == 1:
        return experiments[0]
    else:
        available = [
            exp.get("provenance", {}).get("mode") or exp.get("results", {}).get("mode")
            for exp in experiments
        ]
        raise ValueError(
            f"Multiple experiments found in {path_str}. "
            f"Please specify --mode. Available: {available}"
        )


def _require_dict(value: Any, *, participant_id: int, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Participant {participant_id}: {field_name} must be a dict")
    return cast("dict[str, Any]", value)


def _require_contains_all_keys(
    mapping: dict[str, Any], item_keys: list[str], *, participant_id: int, field_name: str
) -> None:
    missing = [k for k in item_keys if k not in mapping]
    if missing:
        raise ValueError(f"Participant {participant_id}: missing {field_name} keys: {missing}")


@dataclass(frozen=True, slots=True)
class SupervisedCalibration:
    method: str
    target: str
    features: tuple[str, ...]
    extractor: CalibratorFeatureExtractor
    calibrator: LogisticCalibrator | LinearCalibrator | IsotonicCalibrator

    def predict_confidence(self, item_signals: dict[str, Any]) -> float:
        x = self.extractor.extract(item_signals)
        if isinstance(self.calibrator, IsotonicCalibrator):
            # Isotonic is 1D by definition (enforced at load time)
            return float(self.calibrator.predict_proba(np.array([x[0]], dtype=float))[0])
        if isinstance(self.calibrator, LogisticCalibrator):
            return float(self.calibrator.predict_proba(x)[0])
        return float(self.calibrator.predict(x)[0])


Calibration: TypeAlias = TemperatureScalingCalibrator | SupervisedCalibration


def _require_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Calibration artifact '{field_name}' must be a dict")
    return cast("dict[str, Any]", value)


def _require_str(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Calibration artifact '{field_name}' must be a string")
    return value


def _require_number(value: Any, *, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"Calibration artifact '{field_name}' must be a number")
    return float(value)


def _require_str_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise TypeError(f"Calibration artifact '{field_name}' must be a list[str]")
    return value


def _require_number_list(value: Any, *, field_name: str) -> list[float]:
    if not isinstance(value, list) or not all(isinstance(x, (int, float)) for x in value):
        raise TypeError(f"Calibration artifact '{field_name}' must be a list[number]")
    return [float(x) for x in value]


def _load_temperature_scaling(data: dict[str, Any]) -> TemperatureScalingCalibrator:
    temperature = data.get("temperature")
    if temperature is None:
        model = data.get("model")
        if model is not None:
            temperature = _require_mapping(model, field_name="model").get("temperature")
    return TemperatureScalingCalibrator(
        temperature=_require_number(temperature, field_name="temperature")
    )


def _load_logistic_or_platt(data: dict[str, Any], *, method: str) -> SupervisedCalibration:
    features = _require_str_list(data.get("features"), field_name="features")
    target = _require_str(data.get("target"), field_name="target")
    model = _require_mapping(data.get("model"), field_name="model")

    coefficients = np.asarray(
        _require_number_list(model.get("coefficients"), field_name="model.coefficients"),
        dtype=float,
    )
    intercept = _require_number(model.get("intercept"), field_name="model.intercept")
    link = _require_str(model.get("link", "sigmoid"), field_name="model.link")

    scaler_dict = _require_mapping(model.get("scaler"), field_name="model.scaler")
    mean = np.asarray(
        _require_number_list(scaler_dict.get("mean"), field_name="model.scaler.mean"), dtype=float
    )
    std = np.asarray(
        _require_number_list(scaler_dict.get("std"), field_name="model.scaler.std"), dtype=float
    )

    scaler_params = StandardScalerParams(mean=mean, std=std)
    if link == "sigmoid":
        calibrator: LogisticCalibrator | LinearCalibrator = LogisticCalibrator(
            coefficients=coefficients, intercept=intercept, scaler=scaler_params
        )
    elif link == "identity":
        calibrator = LinearCalibrator(
            coefficients=coefficients, intercept=intercept, scaler=scaler_params
        )
    else:
        raise ValueError(f"Unsupported link function '{link}'")

    return SupervisedCalibration(
        method=method,
        target=target,
        features=tuple(features),
        extractor=CalibratorFeatureExtractor(features),
        calibrator=calibrator,
    )


def _load_isotonic(data: dict[str, Any]) -> SupervisedCalibration:
    features = _require_str_list(data.get("features"), field_name="features")
    if len(features) != 1:
        raise ValueError("Isotonic calibration requires exactly one feature")
    target = _require_str(data.get("target"), field_name="target")
    model = _require_mapping(data.get("model"), field_name="model")

    x_thresholds = np.asarray(
        _require_number_list(model.get("x_thresholds"), field_name="model.x_thresholds"),
        dtype=float,
    )
    y_thresholds = np.asarray(
        _require_number_list(model.get("y_thresholds"), field_name="model.y_thresholds"),
        dtype=float,
    )

    return SupervisedCalibration(
        method="isotonic",
        target=target,
        features=tuple(features),
        extractor=CalibratorFeatureExtractor(features),
        calibrator=IsotonicCalibrator(x_thresholds=x_thresholds, y_thresholds=y_thresholds),
    )


def _load_calibration(path: Path) -> Calibration:
    data = load_run_data(path)
    method = data.get("method")
    if method == "temperature_scaling":
        return _load_temperature_scaling(data)
    if method in {"platt", "logistic"}:
        return _load_logistic_or_platt(data, method=method)
    if method == "isotonic":
        return _load_isotonic(data)
    raise ValueError(f"Unsupported calibration method '{method}'")


def _compute_confidence(
    sig: dict[str, Any],
    *,
    confidence_key: str,
    participant_id: int,
    item_key: str,
    calibration: Calibration | None,
) -> float:
    if confidence_key == "verbalized_calibrated":
        if not isinstance(calibration, TemperatureScalingCalibrator):
            raise ValueError(
                "confidence='verbalized_calibrated' requires --calibration "
                "(temperature_scaling.json)."
            )
        raw = float(get_csf("verbalized")(sig))
        conf = float(calibration.apply(np.array([raw], dtype=float))[0])
    elif confidence_key == "calibrated":
        if not isinstance(calibration, SupervisedCalibration):
            raise ValueError("confidence='calibrated' requires --calibration from Spec 049")
        conf = calibration.predict_confidence(sig)
    else:
        try:
            csf = parse_csf_variant(confidence_key)
            conf = float(csf(sig))
        except Exception as exc:
            raise type(exc)(
                f"Participant {participant_id} item {item_key}: {exc} "
                f"(confidence='{confidence_key}')"
            ) from exc

    return conf


def parse_items(
    experiment: dict[str, Any],
    confidence_key: str,
    *,
    calibration: Calibration | None = None,
) -> tuple[list[ItemPrediction], set[int], set[int]]:
    """Parse items from experiment results.

    Returns:
        items: List of ItemPrediction
        participants_included: Set of PIDs with success=True
        participants_failed: Set of PIDs with success=False
    """
    results = experiment.get("results", {}).get("results", [])
    if not isinstance(results, list):
        raise TypeError(f"Experiment results must be a list, got {type(results).__name__}")

    if confidence_key not in CONFIDENCE_VARIANTS and not confidence_key.startswith("secondary:"):
        raise ValueError(f"Unknown confidence_key: {confidence_key}")

    item_keys = [item.value for item in PHQ8Item.all_items()]

    items: list[ItemPrediction] = []
    included = set()
    failed = set()

    for r in results:
        pid = int(cast("int", r["participant_id"]))
        if not r.get("success"):
            failed.add(pid)
            continue

        included.add(pid)

        pred_map = _require_dict(
            r.get("predicted_items"), participant_id=pid, field_name="predicted_items"
        )
        gt_map = _require_dict(
            r.get("ground_truth_items"), participant_id=pid, field_name="ground_truth_items"
        )
        signals = _require_dict(
            r.get("item_signals"), participant_id=pid, field_name="item_signals"
        )

        _require_contains_all_keys(
            gt_map, item_keys, participant_id=pid, field_name="ground_truth_items"
        )
        _require_contains_all_keys(
            pred_map, item_keys, participant_id=pid, field_name="predicted_items"
        )
        _require_contains_all_keys(
            signals, item_keys, participant_id=pid, field_name="item_signals"
        )

        for idx, key in enumerate(item_keys):
            gt = cast("int", gt_map[key])
            pred = cast("int | None", pred_map[key])
            sig = _require_dict(signals[key], participant_id=pid, field_name=f"item_signals[{key}]")

            conf = _compute_confidence(
                sig,
                confidence_key=confidence_key,
                participant_id=pid,
                item_key=key,
                calibration=calibration,
            )

            items.append(
                ItemPrediction(
                    participant_id=pid, item_index=idx, pred=pred, gt=gt, confidence=conf
                )
            )

    return items, included, failed


def compute_metrics_for_variant(
    items: list[ItemPrediction],
    loss: Literal["abs", "abs_norm"],
    coverage_grid: list[float],
    area_coverage: float,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    cmax = compute_cmax(items)

    # Curves
    curve = compute_risk_coverage_curve(items, loss=loss)

    # Scalar metrics
    aurc_full = compute_aurc(items, loss=loss)
    augrc_full = compute_augrc(items, loss=loss)

    # Optimal and Excess metrics (Spec 052)
    aurc_opt = compute_aurc_optimal(items, loss=loss)
    augrc_opt = compute_augrc_optimal(items, loss=loss)
    eaurc = compute_eaurc(items, loss=loss)
    eaugrc = compute_eaugrc(items, loss=loss)
    aurc_achievable = compute_aurc_achievable(items, loss=loss)

    # Interpretation
    aurc_gap_pct = (eaurc / aurc_opt * 100) if aurc_opt > 0 else 0.0
    augrc_gap_pct = (eaugrc / augrc_opt * 100) if augrc_opt > 0 else 0.0
    achievable_gain_pct = 0.0
    if aurc_full > 0:
        achievable_gain_pct = (aurc_full - aurc_achievable) / aurc_full * 100

    # Truncated
    aurc_at_c = compute_aurc_at_coverage(items, max_coverage=area_coverage, loss=loss)
    augrc_at_c = compute_augrc_at_coverage(items, max_coverage=area_coverage, loss=loss)

    # MAE grid
    # Dict mapping coverage string to optional dict
    mae_grid: dict[str, dict[str, float | None] | None] = {}
    for c in coverage_grid:
        val = compute_risk_at_coverage(items, target_coverage=c, loss=loss)
        if val is None:
            mae_grid[f"{c:.2f}"] = None
        else:
            # Find achieved coverage
            # Inefficient but correct: re-scan curve
            achieved = next((cov for cov in curve.coverage if cov >= c), None)
            mae_grid[f"{c:.2f}"] = {"requested": c, "achieved": achieved, "value": val}

    # Bootstrap
    bs_data: dict[str, Any] = {}
    if n_resamples > 0:

        def wrap_aurc_full(x: Sequence[ItemPrediction]) -> float:
            return compute_aurc(x, loss=loss)

        def wrap_augrc_full(x: Sequence[ItemPrediction]) -> float:
            return compute_augrc(x, loss=loss)

        def wrap_cmax(x: Sequence[ItemPrediction]) -> float:
            return compute_cmax(x)

        def wrap_aurc_at_c(x: Sequence[ItemPrediction]) -> float:
            return compute_aurc_at_coverage(x, max_coverage=area_coverage, loss=loss)

        def wrap_augrc_at_c(x: Sequence[ItemPrediction]) -> float:
            return compute_augrc_at_coverage(x, max_coverage=area_coverage, loss=loss)

        # Run bootstraps
        # We can optimize by sharing resamples, but utilizing the helper is cleaner.

        bs_aurc = bootstrap_by_participant(
            items, metric_fn=wrap_aurc_full, n_resamples=n_resamples, seed=seed
        )
        bs_augrc = bootstrap_by_participant(
            items, metric_fn=wrap_augrc_full, n_resamples=n_resamples, seed=seed
        )
        bs_cmax = bootstrap_by_participant(
            items, metric_fn=wrap_cmax, n_resamples=n_resamples, seed=seed
        )
        bs_aurc_c = bootstrap_by_participant(
            items, metric_fn=wrap_aurc_at_c, n_resamples=n_resamples, seed=seed
        )
        bs_augrc_c = bootstrap_by_participant(
            items, metric_fn=wrap_augrc_at_c, n_resamples=n_resamples, seed=seed
        )

        mae_bs: dict[str, list[float] | None] = {}
        drop_rates: dict[str, float] = {}

        for c in coverage_grid:

            def wrap_mae(x: Sequence[ItemPrediction], c: float = c) -> float | None:
                return compute_risk_at_coverage(x, target_coverage=c, loss=loss)

            res = bootstrap_by_participant(
                items, metric_fn=wrap_mae, n_resamples=n_resamples, seed=seed
            )
            k = f"{c:.2f}"
            if res.point_estimate is None:
                mae_bs[k] = None
                drop_rates[k] = 1.0
            else:
                mae_bs[k] = list(res.ci95)
                drop_rates[k] = res.drop_rate

        bs_data = {
            "seed": seed,
            "n_resamples": n_resamples,
            "ci95": {
                "aurc_full": list(bs_aurc.ci95),
                "augrc_full": list(bs_augrc.ci95),
                "aurc_at_c": list(bs_aurc_c.ci95),
                "augrc_at_c": list(bs_augrc_c.ci95),
                "cmax": list(bs_cmax.ci95),
                "mae_at_coverage": mae_bs,
            },
            "drop_rate": {"mae_at_coverage": drop_rates},
        }

    return {
        "cmax": cmax,
        "aurc_full": aurc_full,
        "augrc_full": augrc_full,
        "aurc_optimal": aurc_opt,
        "augrc_optimal": augrc_opt,
        "eaurc": eaurc,
        "eaugrc": eaugrc,
        "aurc_achievable": aurc_achievable,
        "interpretation": {
            "aurc_gap_pct": aurc_gap_pct,
            "augrc_gap_pct": augrc_gap_pct,
            "achievable_gain_pct": achievable_gain_pct,
        },
        "aurc_at_c": {
            "requested": area_coverage,
            "used": min(area_coverage, cmax),
            "value": aurc_at_c,
        },
        "augrc_at_c": {
            "requested": area_coverage,
            "used": min(area_coverage, cmax),
            "value": augrc_at_c,
        },
        "mae_at_coverage": mae_grid,
        "bootstrap": bs_data,
        "curve": {
            "coverage": curve.coverage,
            "selective_risk": curve.selective_risk,
            "generalized_risk": curve.generalized_risk,
            "threshold": curve.threshold,
        },
    }


def main() -> int:  # noqa: PLR0912, PLR0915
    parser = argparse.ArgumentParser(description="Evaluate selective prediction metrics.")
    parser.add_argument("--input", action="append", required=True, help="Path to run JSON output")
    parser.add_argument(
        "--mode",
        action="append",
        help="Mode selection for input (matched by position or broadcast)",
    )
    parser.add_argument("--loss", choices=["abs", "abs_norm"], default="abs_norm")
    parser.add_argument(
        "--confidence",
        choices=[*sorted(CONFIDENCE_VARIANTS), "all"],
        default="all",
        help="Confidence variant for risk-coverage ranking.",
    )
    parser.add_argument(
        "--coverage-grid",
        type=parse_coverage_grid,
        default=DEFAULT_COVERAGE_GRID,
        help="Comma-separated coverage points",
    )
    parser.add_argument(
        "--area-coverage",
        type=float,
        default=DEFAULT_AREA_COVERAGE,
        help="Max coverage for truncated AURC",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--calibration", type=Path, help="Optional calibration artifact JSON")
    parser.add_argument("--output", type=Path, help="Path to save metrics JSON")
    parser.add_argument(
        "--intersection-only",
        action="store_true",
        help="Restrict comparison to overlapping participants",
    )

    args = parser.parse_args()

    inputs = args.input
    modes = args.mode or []

    if len(inputs) > 2:
        print("Error: Max 2 inputs supported.", file=sys.stderr)
        return 1

    # Match modes to inputs
    input_configs = []
    if len(modes) == 0:
        input_configs = [InputConfig(Path(p), None) for p in inputs]
    elif len(modes) == 1:
        input_configs = [InputConfig(Path(p), modes[0]) for p in inputs]
    elif len(modes) == len(inputs):
        input_configs = [InputConfig(Path(p), m) for p, m in zip(inputs, modes, strict=False)]
    else:
        print("Error: --mode count must be 0, 1, or match --input count.", file=sys.stderr)
        return 1

    # Load data

    # Stats for population block
    pop_stats = {
        "participants_total": 0,
        "participants_included": 0,
        "participants_failed": 0,
        "items_total": 0,
    }

    # Resolve items
    input_records: list[dict[str, Any]] = []

    calibration: Calibration | None = None
    if args.calibration is not None:
        calibration = _load_calibration(args.calibration)

    for cfg in input_configs:
        try:
            raw = load_run_data(cfg.path)
            exp = extract_experiment(raw, cfg.mode, str(cfg.path))

            # We need items for all requested confidence keys to build the set?
            # Actually, item set depends on confidence? No, item set is fixed by run.
            # But we need to load signals.

            # Helper to load items for a given confidence variant
            # removed

            # We'll load "all" to determine participants, then re-parse or adjust for
            # specific confidence later?
            # actually parse_items returns ItemPrediction objects which embed the confidence.
            # So we need to call parse_items for each confidence variant requested.

            # Wait, for paired comparison, we need the SAME participant set.
            # Let's load items with a dummy confidence just to get the PID sets first.
            _, included, failed = parse_items(exp, "llm", calibration=calibration)

            input_records.append(
                {
                    "path": str(cfg.path),
                    "run_id": raw.get("run_metadata", {}).get("run_id"),
                    "git_commit": raw.get("run_metadata", {}).get("git_commit"),
                    "mode": exp.get("results", {}).get("mode"),  # resolved mode
                    "included_pids": included,
                    "failed_pids": failed,
                    "exp": exp,
                }
            )

        except Exception as e:
            print(f"Error loading {cfg.path}: {e}", file=sys.stderr)
            return 1

    # Determine analysis set
    is_paired = len(input_records) == 2
    analysis_pids: set[int] = set()

    if not is_paired:
        analysis_pids = input_records[0]["included_pids"]
    else:
        left_total = input_records[0]["included_pids"] | input_records[0]["failed_pids"]
        right_total = input_records[1]["included_pids"] | input_records[1]["failed_pids"]

        overlap_total = left_total & right_total

        if not args.intersection_only and left_total != right_total:
            print(
                "Error: Participant sets differ and --intersection-only not set.", file=sys.stderr
            )
            print(
                f"Left total: {len(left_total)}, "
                f"Right total: {len(right_total)}, "
                f"Overlap: {len(overlap_total)}",
                file=sys.stderr,
            )
            return 1

        left_success = input_records[0]["included_pids"] & overlap_total
        right_success = input_records[1]["included_pids"] & overlap_total
        analysis_pids = left_success & right_success

    # Compute population stats
    # Total is union of included + failed in the scope of analysis?

    if not is_paired:
        rec = input_records[0]
        # In single mode, we use all included
        pop_stats["participants_included"] = len(analysis_pids)
        # Failed count from that run
        pop_stats["participants_failed"] = len(rec["failed_pids"])
        pop_stats["participants_total"] = len(analysis_pids) + len(rec["failed_pids"])
        pop_stats["items_total"] = len(analysis_pids) * 8
    else:
        # Paired stats
        rec1 = input_records[0]
        rec2 = input_records[1]

        all_pids_1 = rec1["included_pids"] | rec1["failed_pids"]
        all_pids_2 = rec2["included_pids"] | rec2["failed_pids"]

        overlap_total = all_pids_1 & all_pids_2

        # Of the overlap total, how many are included (successful in both)?
        pop_stats["participants_included"] = len(analysis_pids)

        # Failed? Those in overlap_total but not in analysis_pids
        failed_in_overlap = len(overlap_total) - len(analysis_pids)
        pop_stats["participants_failed"] = failed_in_overlap
        pop_stats["participants_total"] = len(overlap_total)
        pop_stats["items_total"] = len(analysis_pids) * 8

    # Process Confidence Variants
    variants = CONFIDENCE_DEFAULT_VARIANTS if args.confidence == "all" else [args.confidence]

    results_map = {}
    deltas_map = {}

    for variant in variants:
        # Load items for this variant, filtering to analysis_pids

        variant_results = []

        for rec in input_records:
            items_all, _, _ = parse_items(rec["exp"], variant, calibration=calibration)
            # Filter
            items_filtered = [i for i in items_all if i.participant_id in analysis_pids]
            expected_n = len(analysis_pids) * 8
            if len(items_filtered) != expected_n:
                raise ValueError(
                    f"Parsed {len(items_filtered)} items for {len(analysis_pids)} participants; "
                    f"expected {expected_n}. "
                    "Check predicted_items/ground_truth_items/item_signals keys."
                )

            # Compute metrics
            metrics = compute_metrics_for_variant(
                items_filtered,
                loss=args.loss,
                coverage_grid=args.coverage_grid,
                area_coverage=args.area_coverage,
                n_resamples=args.bootstrap_resamples,
                seed=args.seed,
            )
            variant_results.append(metrics)

        if not is_paired:
            results_map[variant] = variant_results[0]
        else:
            # Handle Paired Logic
            results_map[variant] = {f"input_{i}": res for i, res in enumerate(variant_results)}

            # Compute deltas
            items_left = [
                i
                for i in parse_items(input_records[0]["exp"], variant, calibration=calibration)[0]
                if i.participant_id in analysis_pids
            ]
            items_right = [
                i
                for i in parse_items(input_records[1]["exp"], variant, calibration=calibration)[0]
                if i.participant_id in analysis_pids
            ]

            def wrap_aurc(x: Sequence[ItemPrediction]) -> float:
                return compute_aurc(x, loss=args.loss)

            def wrap_augrc(x: Sequence[ItemPrediction]) -> float:
                return compute_augrc(x, loss=args.loss)

            def wrap_cmax(x: Sequence[ItemPrediction]) -> float:
                return compute_cmax(x)

            # Bootstrapped Deltas
            d_aurc = paired_bootstrap_delta_by_participant(
                items_left,
                items_right,
                metric_fn=wrap_aurc,
                n_resamples=args.bootstrap_resamples,
                seed=args.seed,
            )
            d_augrc = paired_bootstrap_delta_by_participant(
                items_left,
                items_right,
                metric_fn=wrap_augrc,
                n_resamples=args.bootstrap_resamples,
                seed=args.seed,
            )
            d_cmax = paired_bootstrap_delta_by_participant(
                items_left,
                items_right,
                metric_fn=wrap_cmax,
                n_resamples=args.bootstrap_resamples,
                seed=args.seed,
            )

            deltas_map[variant] = {
                "aurc_full": {"delta": d_aurc.delta_point_estimate, "ci95": list(d_aurc.ci95)},
                "augrc_full": {"delta": d_augrc.delta_point_estimate, "ci95": list(d_augrc.ci95)},
                "cmax": {"delta": d_cmax.delta_point_estimate, "ci95": list(d_cmax.ci95)},
            }

    # Construct Artifact
    artifact = {
        "schema_version": "1",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "inputs": [
            {
                "path": r["path"],
                "run_id": r["run_id"],
                "git_commit": r["git_commit"],
                "mode": r["mode"],
            }
            for r in input_records
        ],
        "population": pop_stats,
        "loss": {
            "name": args.loss,
            "definition": "abs(pred - gt)" + (" / 3" if args.loss == "abs_norm" else ""),
            "raw_multiplier": 3 if args.loss == "abs_norm" else 1,
        },
        "confidence_variants": results_map,
        "comparison": {
            "enabled": is_paired,
            "intersection_only": args.intersection_only,
            "deltas": deltas_map if is_paired else None,
        },
    }

    if is_paired:
        # Fill comparison details
        rec1 = input_records[0]
        rec2 = input_records[1]

        all1 = rec1["included_pids"] | rec1["failed_pids"]
        all2 = rec2["included_pids"] | rec2["failed_pids"]
        overlap = all1 & all2

        # Cast for mypy because update expects SupportsKeysAndGetItem
        # But dict is sufficient. Mypy issue?
        # The error was: "Collection[Collection[str]]" has no attribute "update"
        # artifact["comparison"] is typed as what? Dict[str, Any]?
        # MyPy infers types.

        comp_dict = cast("dict[str, Any]", artifact["comparison"])
        comp_dict.update(
            {
                "participants_left_only": len(all1 - all2),
                "participants_right_only": len(all2 - all1),
                "participants_overlap_total": len(overlap),
                "participants_overlap_included": len(analysis_pids),
                "participants_failed_left": len([p for p in overlap if p in rec1["failed_pids"]]),
                "participants_failed_right": len([p for p in overlap if p in rec2["failed_pids"]]),
            }
        )

    # Output
    output_path = args.output
    if output_path is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = Path("data/outputs") / f"selective_prediction_metrics_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(artifact, f, indent=2)
    print(f"Metrics saved to {output_path}")

    # Console Summary
    print("\nSELECTIVE PREDICTION EVALUATION")
    print("===============================")
    print(f"Loss: {args.loss}")
    print(f"Participants: {pop_stats['participants_included']} included")

    for var, res in results_map.items():
        print(f"\nConfidence: {var}")

        # Helper to print one set of metrics
        def print_metrics(m: dict[str, Any], label: str = "") -> None:
            prefix = f"  {label}: " if label else "  "
            bs = m.get("bootstrap", {}).get("ci95", {})

            def fmt(key: str, val: float | None) -> str:
                if val is None:
                    return "N/A"
                ci = bs.get(key)
                s = f"{val:.4f}"
                if ci:
                    s += f" [{ci[0]:.4f}, {ci[1]:.4f}]"
                return s

            print(f"{prefix}Cmax:       {fmt('cmax', m['cmax'])}")

            aurc_info = f"{fmt('aurc_full', m['aurc_full'])} "
            aurc_info += f"(Opt: {m.get('aurc_optimal', 0):.4f}, Excess: {m.get('eaurc', 0):.4f})"
            print(f"{prefix}AURC:       {aurc_info}")

            augrc_info = f"{fmt('augrc_full', m['augrc_full'])} "
            augrc_info += f"(Opt: {m.get('augrc_optimal', 0):.4f}, "
            augrc_info += f"Excess: {m.get('eaugrc', 0):.4f})"
            print(f"{prefix}AUGRC:      {augrc_info}")

        if is_paired:
            print_metrics(res["input_0"], "Left ")
            print_metrics(res["input_1"], "Right")

            d = deltas_map[var]
            print("  Deltas (Right - Left):")
            for k in ["cmax", "aurc_full", "augrc_full"]:
                dd = d[k]
                ci = dd["ci95"]
                # MyPy complains about indexing optional list
                ci_list = cast("list[float]", ci)
                print(f"    d{k}: {dd['delta']:.4f} [{ci_list[0]:.4f}, {ci_list[1]:.4f}]")

        else:
            print_metrics(res)

    return 0


if __name__ == "__main__":
    sys.exit(main())

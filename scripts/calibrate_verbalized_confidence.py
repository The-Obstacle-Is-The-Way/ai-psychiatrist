#!/usr/bin/env python3
"""Fit temperature scaling for verbalized confidence (Spec 048).

This script calibrates the 1-5 verbalized confidence scale into an estimated
probability of correctness using single-parameter temperature scaling.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_psychiatrist.calibration.calibrators import (
    TemperatureScalingCalibrator,
    compute_binary_nll,
    compute_ece,
)
from ai_psychiatrist.domain.enums import PHQ8Item


@dataclass(frozen=True, slots=True)
class InputConfig:
    path: Path
    mode: str


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return cast("dict[str, Any]", json.load(f))


def _extract_experiment(run_data: dict[str, Any], *, mode: str, path_str: str) -> dict[str, Any]:
    experiments = cast("list[dict[str, Any]]", run_data.get("experiments", []))
    if not experiments:
        raise ValueError(f"No experiments found in {path_str}")

    matches = [
        exp
        for exp in experiments
        if exp.get("provenance", {}).get("mode") == mode
        or exp.get("results", {}).get("mode") == mode
    ]
    if not matches:
        available = [
            exp.get("provenance", {}).get("mode") or exp.get("results", {}).get("mode")
            for exp in experiments
        ]
        raise ValueError(f"Mode '{mode}' not found in {path_str}. Available: {available}")
    return matches[0]


def _normalize_verbalized_confidence(v: int) -> float:
    # 1-5 -> [0, 1]
    normalized = (float(v) - 1.0) / 4.0
    return max(0.0, min(1.0, normalized))


def extract_training_pairs(exp: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Extract (p_raw, y_correct) pairs from an experiment."""
    results = exp.get("results", {}).get("results", [])
    if not isinstance(results, list):
        raise TypeError(f"Experiment results must be a list, got {type(results).__name__}")

    item_keys = [item.value for item in PHQ8Item.all_items()]

    ps: list[float] = []
    ys: list[int] = []

    for r in results:
        if not r.get("success"):
            continue
        pred_map = r.get("predicted_items")
        gt_map = r.get("ground_truth_items")
        signals = r.get("item_signals")
        if (
            not isinstance(pred_map, dict)
            or not isinstance(gt_map, dict)
            or not isinstance(signals, dict)
        ):
            raise TypeError("predicted_items/ground_truth_items/item_signals must be dicts")

        for key in item_keys:
            pred = pred_map.get(key)
            if pred is None:
                continue
            gt = gt_map.get(key)
            if not isinstance(gt, int):
                raise TypeError(f"ground_truth_items['{key}'] must be int")
            sig = signals.get(key)
            if not isinstance(sig, dict):
                raise TypeError(f"item_signals['{key}'] must be a dict")
            v = sig.get("verbalized_confidence")
            if v is None:
                continue
            if isinstance(v, float) and v == int(v):
                v = int(v)
            if not isinstance(v, int):
                raise TypeError("verbalized_confidence must be int or null")
            if not 1 <= v <= 5:
                raise ValueError(f"verbalized_confidence must be 1-5, got {v}")

            ps.append(_normalize_verbalized_confidence(v))
            ys.append(1 if int(pred) == gt else 0)

    return np.asarray(ps, dtype=float), np.asarray(ys, dtype=int)


def build_artifact(
    *,
    input_path: Path,
    run_data: dict[str, Any],
    mode: str,
    p_raw: np.ndarray,
    y: np.ndarray,
    calibrator: TemperatureScalingCalibrator,
) -> dict[str, Any]:
    p_cal = calibrator.apply(p_raw)
    return {
        "schema_version": "1",
        "method": "temperature_scaling",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "temperature": calibrator.temperature,
        "fitted_on": {
            "path": str(input_path),
            "run_id": run_data.get("run_metadata", {}).get("run_id"),
            "git_commit": run_data.get("run_metadata", {}).get("git_commit"),
            "mode": mode,
            "n_samples": int(p_raw.size),
        },
        "metrics": {
            "nll_before": compute_binary_nll(p_raw, y),
            "nll_after": compute_binary_nll(p_cal, y),
            "ece_before": compute_ece(p_raw, y),
            "ece_after": compute_ece(p_cal, y),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate verbalized confidence (Spec 048).")
    parser.add_argument("--input", type=Path, required=True, help="Path to run JSON output")
    parser.add_argument("--mode", required=True, help="Mode to select from the run output")
    parser.add_argument("--output", type=Path, required=True, help="Path to save calibration JSON")
    args = parser.parse_args()

    cfg = InputConfig(path=args.input, mode=args.mode)

    run_data = _load_json(cfg.path)
    exp = _extract_experiment(run_data, mode=cfg.mode, path_str=str(cfg.path))
    p_raw, y = extract_training_pairs(exp)

    if p_raw.size == 0:
        raise ValueError(
            "No training samples found (need predicted items with verbalized_confidence)"
        )

    calibrator = TemperatureScalingCalibrator.fit(p_raw, y)
    artifact = build_artifact(
        input_path=cfg.path,
        run_data=run_data,
        mode=cfg.mode,
        p_raw=p_raw,
        y=y,
        calibrator=calibrator,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Calibration saved to {args.output}")
    print(f"Temperature: {calibrator.temperature:.4f}")
    print(f"NLL: {artifact['metrics']['nll_before']:.4f} -> {artifact['metrics']['nll_after']:.4f}")
    print(f"ECE: {artifact['metrics']['ece_before']:.4f} -> {artifact['metrics']['ece_after']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

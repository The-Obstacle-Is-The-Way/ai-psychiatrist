#!/usr/bin/env python3
"""Train supervised confidence calibrators from run artifacts (Spec 049)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_psychiatrist.calibration.calibrators import (
    IsotonicCalibrator,
    LinearCalibrator,
    LogisticCalibrator,
    StandardScalerParams,
    TemperatureScalingCalibrator,
    compute_ece,
)
from ai_psychiatrist.calibration.feature_extraction import CalibratorFeatureExtractor
from ai_psychiatrist.domain.enums import PHQ8Item

Target = Literal["correctness", "near_correct", "loss"]
Method = Literal["temperature", "platt", "logistic", "isotonic"]


@dataclass(frozen=True, slots=True)
class TrainConfig:
    input_path: Path
    mode: str
    method: Method
    features: tuple[str, ...]
    target: Target
    output_path: Path


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


def _build_dataset(
    exp: dict[str, Any],
    *,
    features: tuple[str, ...],
    target: Target,
) -> tuple[np.ndarray, np.ndarray]:
    results = exp.get("results", {}).get("results", [])
    if not isinstance(results, list):
        raise TypeError(f"Experiment results must be a list, got {type(results).__name__}")

    extractor = CalibratorFeatureExtractor(list(features))
    item_keys = [item.value for item in PHQ8Item.all_items()]

    xs: list[np.ndarray] = []
    ys: list[float] = []

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

            x = extractor.extract(sig)
            abs_err = abs(int(pred) - gt)
            if target == "correctness":
                y = 1.0 if abs_err == 0 else 0.0
            elif target == "near_correct":
                y = 1.0 if abs_err <= 1 else 0.0
            else:
                y = 1.0 - (abs_err / 3.0)
            xs.append(x)
            ys.append(y)

    if not xs:
        raise ValueError("No training samples found (need predicted items with required signals)")

    x_mat = np.vstack(xs)
    y_vec = np.asarray(ys, dtype=float)
    return x_mat, y_vec


def _fit_standard_scaler(x: np.ndarray) -> StandardScalerParams:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std == 0, 1.0, std)
    return StandardScalerParams(mean=mean, std=std)


def _fit_logistic_model(x: np.ndarray, y: np.ndarray) -> LogisticCalibrator:
    scaler = _fit_standard_scaler(x)
    x_scaled = scaler.transform(x)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0)
    model.fit(x_scaled, y.astype(int))
    return LogisticCalibrator(
        coefficients=np.asarray(model.coef_[0], dtype=float),
        intercept=float(model.intercept_[0]),
        scaler=scaler,
    )


def _fit_linear_model(x: np.ndarray, y: np.ndarray) -> LinearCalibrator:
    scaler = _fit_standard_scaler(x)
    x_scaled = scaler.transform(x)
    model = Ridge(alpha=1.0, random_state=0)
    model.fit(x_scaled, y.astype(float))
    return LinearCalibrator(
        coefficients=np.asarray(model.coef_, dtype=float),
        intercept=float(model.intercept_),
        scaler=scaler,
    )


def _fit_isotonic_model(x: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(x.astype(float), y.astype(float))
    return IsotonicCalibrator(
        x_thresholds=np.asarray(model.X_thresholds_, dtype=float),
        y_thresholds=np.asarray(model.y_thresholds_, dtype=float),
    )


def _predict(
    *,
    method: Method,
    target: Target,
    x: np.ndarray,
    calibrator: TemperatureScalingCalibrator
    | LogisticCalibrator
    | LinearCalibrator
    | IsotonicCalibrator,
) -> np.ndarray:
    if method == "temperature":
        # temperature scaling uses the single feature as probability input
        p = x[:, 0]
        return calibrator.apply(p) if isinstance(calibrator, TemperatureScalingCalibrator) else p
    if isinstance(calibrator, LogisticCalibrator):
        return calibrator.predict_proba(x)
    if isinstance(calibrator, LinearCalibrator):
        return calibrator.predict(x)
    if isinstance(calibrator, IsotonicCalibrator):
        return calibrator.predict_proba(x[:, 0])
    raise TypeError(f"Unsupported calibrator type for method '{method}' and target '{target}'")


def _binary_metrics(y_true: np.ndarray, p_pred: np.ndarray) -> dict[str, float | None]:
    y_int = y_true.astype(int)
    try:
        auc = float(roc_auc_score(y_int, p_pred))
    except ValueError:
        auc = None

    return {
        "auc_roc": auc,
        "brier_score": float(brier_score_loss(y_int, p_pred)),
        "ece": compute_ece(p_pred, y_int),
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    mae = float(np.mean(np.abs(y_pred - y_true)))
    mse = float(np.mean((y_pred - y_true) ** 2))
    return {"mae": mae, "mse": mse}


def build_artifact(
    *,
    cfg: TrainConfig,
    run_data: dict[str, Any],
    n_samples: int,
    positive_rate: float | None,
    calibrator: TemperatureScalingCalibrator
    | LogisticCalibrator
    | LinearCalibrator
    | IsotonicCalibrator,
    validation_metrics: dict[str, Any],
) -> dict[str, Any]:
    method = "temperature_scaling" if cfg.method == "temperature" else cfg.method
    artifact: dict[str, Any] = {
        "schema_version": "1",
        "method": method,
        "version": "1.0",
        "features": list(cfg.features),
        "target": cfg.target,
        "training_metadata": {
            "path": str(cfg.input_path),
            "run_id": run_data.get("run_metadata", {}).get("run_id"),
            "git_commit": run_data.get("run_metadata", {}).get("git_commit"),
            "mode": cfg.mode,
            "n_samples": n_samples,
            "positive_rate": positive_rate,
            "trained_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
        "validation_metrics": validation_metrics,
    }

    if isinstance(calibrator, TemperatureScalingCalibrator):
        artifact["model"] = {"temperature": calibrator.temperature}
    elif isinstance(calibrator, IsotonicCalibrator):
        artifact["model"] = {
            "x_thresholds": calibrator.x_thresholds.tolist(),
            "y_thresholds": calibrator.y_thresholds.tolist(),
        }
    else:
        link = "sigmoid" if isinstance(calibrator, LogisticCalibrator) else "identity"
        artifact["model"] = {
            "coefficients": calibrator.coefficients.tolist(),
            "intercept": calibrator.intercept,
            "link": link,
            "scaler": {
                "mean": calibrator.scaler.mean.tolist(),
                "std": calibrator.scaler.std.tolist(),
            },
        }

    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Train confidence calibrator (Spec 049).")
    parser.add_argument("--input", type=Path, required=True, help="Path to run JSON output")
    parser.add_argument("--mode", required=True, help="Mode to select from the run output")
    parser.add_argument(
        "--method", choices=["temperature", "platt", "logistic", "isotonic"], required=True
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Comma-separated list of item_signals feature names",
    )
    parser.add_argument(
        "--target",
        choices=["correctness", "near_correct", "loss"],
        default="correctness",
        help="Training target for calibration",
    )
    parser.add_argument("--output", type=Path, required=True, help="Path to save calibrator JSON")
    args = parser.parse_args()

    cfg = TrainConfig(
        input_path=args.input,
        mode=args.mode,
        method=cast("Method", args.method),
        features=tuple([s.strip() for s in args.features.split(",") if s.strip()]),
        target=cast("Target", args.target),
        output_path=args.output,
    )

    if not cfg.features:
        raise ValueError("--features must include at least one feature name")

    if cfg.method in {"temperature", "platt", "isotonic"} and len(cfg.features) != 1:
        raise ValueError(f"method='{cfg.method}' requires exactly one feature")

    if cfg.method == "temperature" and cfg.target == "loss":
        raise ValueError("method='temperature' does not support target='loss'")

    run_data = _load_json(cfg.input_path)
    exp = _extract_experiment(run_data, mode=cfg.mode, path_str=str(cfg.input_path))
    x, y = _build_dataset(exp, features=cfg.features, target=cfg.target)

    calibrator: (
        TemperatureScalingCalibrator | LogisticCalibrator | LinearCalibrator | IsotonicCalibrator
    )

    if cfg.method == "temperature":
        p = x[:, 0]
        if np.any(p < 0.0) or np.any(p > 1.0):
            raise ValueError(
                "temperature scaling requires probability-like feature values in [0,1]"
            )
        calibrator = TemperatureScalingCalibrator.fit(p, y.astype(int))
    elif cfg.method in {"platt", "logistic"}:
        if cfg.target in {"correctness", "near_correct"}:
            calibrator = _fit_logistic_model(x, y)
        else:
            calibrator = _fit_linear_model(x, y)
    else:
        calibrator = _fit_isotonic_model(x[:, 0], y)

    y_pred = _predict(method=cfg.method, target=cfg.target, x=x, calibrator=calibrator)

    if cfg.target in {"correctness", "near_correct"}:
        metrics = _binary_metrics(y, y_pred)
        positive_rate = float(np.mean(y)) if y.size else None
    else:
        metrics = _regression_metrics(y, y_pred)
        positive_rate = None

    artifact = build_artifact(
        cfg=cfg,
        run_data=run_data,
        n_samples=int(y.size),
        positive_rate=positive_rate,
        calibrator=calibrator,
        validation_metrics=metrics,
    )

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output_path.open("w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Calibrator saved to {cfg.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

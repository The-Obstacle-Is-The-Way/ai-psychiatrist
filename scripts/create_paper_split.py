#!/usr/bin/env python3
"""Create stratified train/val/test splits matching paper methodology (Appendix C).

The paper does NOT use the original DAIC-WOZ train/dev/test splits. Instead, it:
1. Combines the 142 subjects from DAIC-WOZ train (107) + dev (35) who have per-item PHQ-8 labels
2. Creates a CUSTOM 58/43/41 split stratified by PHQ-8 total score AND gender

From Paper Section 2.4.1:
    "We split 142 subjects with eight-item PHQ-8 scores from the DAIC-WOZ database
    into training, validation, and test sets. [...] We used a 41% training (58
    participants), 30% validation (43), and 29% test (41) split"

Modes:
    --mode ground-truth (DEFAULT):
        Uses the exact participant IDs reverse-engineered from the paper authors'
        output files (see docs/data/paper-split-registry.md). This is required for
        exact reproduction of paper results.

    --mode algorithmic:
        Uses the Appendix C stratification algorithm with a random seed. This produces
        valid splits of the correct sizes (58/43/41) but different participant
        assignments than the paper used.

Usage:
    # Use ground truth IDs (default)
    python scripts/create_paper_split.py

    # Verify against AVEC data + registry
    python scripts/create_paper_split.py --verify

    # Use algorithmic generation with seed
    python scripts/create_paper_split.py --mode algorithmic --seed 123
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_TARGET_TRAIN_COUNT = 58
_TARGET_VAL_COUNT = 43
_TARGET_TEST_COUNT = 41

# Ground Truth IDs from docs/data/paper-split-registry.md
_GROUND_TRUTH_TRAIN_IDS = [
    303,
    304,
    305,
    310,
    312,
    313,
    315,
    317,
    318,
    321,
    324,
    327,
    335,
    338,
    340,
    343,
    344,
    346,
    347,
    350,
    352,
    356,
    363,
    368,
    369,
    388,
    391,
    395,
    397,
    400,
    402,
    404,
    406,
    412,
    414,
    415,
    416,
    418,
    426,
    429,
    433,
    434,
    437,
    439,
    444,
    458,
    463,
    464,
    473,
    474,
    475,
    476,
    477,
    478,
    483,
    486,
    488,
    491,
]

_GROUND_TRUTH_VAL_IDS = [
    302,
    307,
    320,
    322,
    325,
    326,
    328,
    331,
    333,
    336,
    341,
    348,
    351,
    353,
    355,
    358,
    360,
    364,
    366,
    371,
    372,
    374,
    376,
    380,
    381,
    382,
    392,
    401,
    403,
    419,
    420,
    425,
    440,
    443,
    446,
    448,
    454,
    457,
    471,
    479,
    482,
    490,
    492,
]

_GROUND_TRUTH_TEST_IDS = [
    316,
    319,
    330,
    339,
    345,
    357,
    362,
    367,
    370,
    375,
    377,
    379,
    383,
    385,
    386,
    389,
    390,
    393,
    409,
    413,
    417,
    422,
    423,
    427,
    428,
    430,
    436,
    441,
    445,
    447,
    449,
    451,
    455,
    456,
    459,
    468,
    472,
    484,
    485,
    487,
    489,
]


@dataclass
class _SplitBuckets:
    train: set[int] = field(default_factory=set)
    val: set[int] = field(default_factory=set)
    test: set[int] = field(default_factory=set)


@dataclass
class _SplitAllocation:
    fixed: _SplitBuckets = field(default_factory=_SplitBuckets)
    flexible: _SplitBuckets = field(default_factory=_SplitBuckets)
    all: _SplitBuckets = field(default_factory=_SplitBuckets)


@dataclass
class SplitStatistics:
    """Statistics for a data split."""

    count: int
    male_count: int
    female_count: int
    phq8_mean: float
    phq8_std: float
    phq8_min: int
    phq8_max: int


def load_combined_data(data_dir: Path) -> pd.DataFrame:
    """Load and combine train + dev splits from AVEC2017.

    Returns:
        DataFrame with columns: Participant_ID, Gender, PHQ8_Total, PHQ8_* items
    """
    train_csv = data_dir / "train_split_Depression_AVEC2017.csv"
    dev_csv = data_dir / "dev_split_Depression_AVEC2017.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Train split not found: {train_csv}")
    if not dev_csv.exists():
        raise FileNotFoundError(f"Dev split not found: {dev_csv}")

    train_df = pd.read_csv(train_csv)
    dev_df = pd.read_csv(dev_csv)

    # Combine
    combined = pd.concat([train_df, dev_df], ignore_index=True)

    # Compute total PHQ-8 score from the 8 per-item columns.
    # NOTE: The AVEC2017 train/dev CSVs also contain derived columns like
    # `PHQ8_Binary` and `PHQ8_Score`. These MUST NOT be included in the per-item sum.
    item_cols = [
        c
        for c in combined.columns
        if c.startswith("PHQ8_") and c not in {"PHQ8_Binary", "PHQ8_Score"}
    ]

    # Fail loudly on missing PHQ-8 item values (BUG-025)
    missing = combined[combined[item_cols].isna().any(axis=1)]
    if not missing.empty:
        affected = missing["Participant_ID"].tolist()
        raise ValueError(
            f"Missing PHQ-8 item values for participants: {affected}. "
            f"Run 'uv run python scripts/patch_missing_phq8_values.py --apply' to fix. "
            f"See docs/archive/bugs/bug-025-missing-phq8-ground-truth-paper-test.md"
        )

    combined["PHQ8_Total"] = combined[item_cols].sum(axis=1).astype(int)

    return combined


def _build_groups(
    df: pd.DataFrame,
) -> tuple[dict[int, list[int]], dict[int, int], dict[int, int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    gender_by_pid: dict[int, int] = {}
    total_by_pid: dict[int, int] = {}

    for row in df.itertuples(index=False):
        phq_total = int(row.PHQ8_Total)
        pid = int(row.Participant_ID)
        groups[phq_total].append(pid)
        gender_by_pid[pid] = int(row.Gender)
        total_by_pid[pid] = phq_total

    return groups, gender_by_pid, total_by_pid


def _initial_allocate(
    groups: dict[int, list[int]],
    *,
    rng: random.Random,
) -> _SplitAllocation:
    allocation = _SplitAllocation()

    for pids in groups.values():
        shuffled = pids.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        if n == 1:
            pid = shuffled[0]
            allocation.fixed.train.add(pid)
            allocation.all.train.add(pid)
            continue

        if n == 2:
            pid_val, pid_test = shuffled
            allocation.fixed.val.add(pid_val)
            allocation.fixed.test.add(pid_test)
            allocation.all.val.add(pid_val)
            allocation.all.test.add(pid_test)
            continue

        # Initial proportional allocation. We'll reconcile to exact targets later.
        n_train = max(0, round(n * 0.41))
        n_val = max(0, round(n * 0.30))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)

        allocation.flexible.train.update(shuffled[:n_train])
        allocation.flexible.val.update(shuffled[n_train : n_train + n_val])
        allocation.flexible.test.update(shuffled[n_train + n_val :])

    allocation.all.train.update(allocation.flexible.train)
    allocation.all.val.update(allocation.flexible.val)
    allocation.all.test.update(allocation.flexible.test)
    return allocation


def _move_random(
    source_flex: set[int],
    dest_flex: set[int],
    *,
    source_all: set[int],
    dest_all: set[int],
    rng: random.Random,
) -> None:
    pid = rng.choice(tuple(source_flex))
    source_flex.remove(pid)
    dest_flex.add(pid)
    source_all.remove(pid)
    dest_all.add(pid)


def _rebalance_to_targets(
    allocation: _SplitAllocation,
    *,
    target_train: int,
    target_val: int,
    target_test: int,
    rng: random.Random,
) -> None:
    """Rebalance to exact paper target sizes using only flexible IDs."""
    train_ids = allocation.all.train
    val_ids = allocation.all.val
    test_ids = allocation.all.test

    flex_train = allocation.flexible.train
    flex_val = allocation.flexible.val
    flex_test = allocation.flexible.test

    while len(test_ids) < target_test:
        if len(val_ids) > target_val and flex_val:
            _move_random(flex_val, flex_test, source_all=val_ids, dest_all=test_ids, rng=rng)
            continue
        if len(train_ids) > target_train and flex_train:
            _move_random(flex_train, flex_test, source_all=train_ids, dest_all=test_ids, rng=rng)
            continue
        raise RuntimeError("Unable to rebalance to target test size without moving fixed IDs")

    while len(train_ids) > target_train:
        if not flex_train:
            raise RuntimeError("Train split too large but no flexible IDs to move")
        _move_random(flex_train, flex_val, source_all=train_ids, dest_all=val_ids, rng=rng)

    while len(val_ids) > target_val:
        if not flex_val:
            raise RuntimeError("Validation split too large but no flexible IDs to move")
        _move_random(flex_val, flex_train, source_all=val_ids, dest_all=train_ids, rng=rng)

    while len(test_ids) > target_test:
        if not flex_test:
            raise RuntimeError("Test split too large but no flexible IDs to move")
        if len(train_ids) < target_train:
            _move_random(flex_test, flex_train, source_all=test_ids, dest_all=train_ids, rng=rng)
            continue
        if len(val_ids) < target_val:
            _move_random(flex_test, flex_val, source_all=test_ids, dest_all=val_ids, rng=rng)
            continue
        raise RuntimeError(
            "Test split too large but train/val already at target sizes without moving fixed IDs"
        )


def _male_count(pids: set[int], gender_by_pid: dict[int, int]) -> int:
    return sum(1 for pid in pids if gender_by_pid[pid] == 0)


def _swap_best(
    *,
    a_set: set[int],
    b_set: set[int],
    a_flex: set[int],
    b_flex: set[int],
    a_want_gender: int,
    b_want_gender: int,
    gender_by_pid: dict[int, int],
    total_by_pid: dict[int, int],
) -> bool:
    """Swap one flexible PID between sets to improve gender balance.

    Chooses a swap that minimally perturbs PHQ8_Total distribution by preferring
    close total-score pairs.
    """
    a_candidates = sorted(pid for pid in a_flex if gender_by_pid[pid] == a_want_gender)
    b_candidates = sorted(pid for pid in b_flex if gender_by_pid[pid] == b_want_gender)
    if not a_candidates or not b_candidates:
        return False

    best_pair: tuple[int, int] | None = None
    best_delta: int | None = None
    for a_pid in a_candidates:
        a_total = total_by_pid[a_pid]
        for b_pid in b_candidates:
            delta = abs(a_total - total_by_pid[b_pid])
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_pair = (a_pid, b_pid)
                if best_delta == 0:
                    break
        if best_delta == 0:
            break

    if best_pair is None:
        return False

    a_pid, b_pid = best_pair
    a_set.remove(a_pid)
    b_set.remove(b_pid)
    a_set.add(b_pid)
    b_set.add(a_pid)

    a_flex.remove(a_pid)
    b_flex.remove(b_pid)
    a_flex.add(b_pid)
    b_flex.add(a_pid)
    return True


def _balance_gender(
    allocation: _SplitAllocation,
    *,
    gender_by_pid: dict[int, int],
    total_by_pid: dict[int, int],
    target_train: int,
    target_val: int,
    max_iterations: int,
) -> None:
    """Improve gender balance using swaps within flexible IDs."""
    train_ids = allocation.all.train
    val_ids = allocation.all.val
    test_ids = allocation.all.test

    flex_train = allocation.flexible.train
    flex_val = allocation.flexible.val
    flex_test = allocation.flexible.test

    total_male = sum(1 for g in gender_by_pid.values() if g == 0)
    target_train_male = round(total_male * target_train / len(gender_by_pid))
    target_val_male = round(total_male * target_val / len(gender_by_pid))
    target_test_male = total_male - target_train_male - target_val_male

    for _ in range(max_iterations):
        train_male = _male_count(train_ids, gender_by_pid)
        val_male = _male_count(val_ids, gender_by_pid)
        test_male = _male_count(test_ids, gender_by_pid)

        attempts: list[tuple[bool, set[int], set[int], set[int], set[int], int, int]] = [
            (
                train_male > target_train_male and val_male < target_val_male,
                train_ids,
                val_ids,
                flex_train,
                flex_val,
                0,
                1,
            ),
            (
                train_male < target_train_male and val_male > target_val_male,
                train_ids,
                val_ids,
                flex_train,
                flex_val,
                1,
                0,
            ),
            (
                train_male > target_train_male and test_male < target_test_male,
                train_ids,
                test_ids,
                flex_train,
                flex_test,
                0,
                1,
            ),
            (
                train_male < target_train_male and test_male > target_test_male,
                train_ids,
                test_ids,
                flex_train,
                flex_test,
                1,
                0,
            ),
            (
                val_male > target_val_male and test_male < target_test_male,
                val_ids,
                test_ids,
                flex_val,
                flex_test,
                0,
                1,
            ),
            (
                val_male < target_val_male and test_male > target_test_male,
                val_ids,
                test_ids,
                flex_val,
                flex_test,
                1,
                0,
            ),
        ]

        for cond, a_set, b_set, a_flex, b_flex, a_gender, b_gender in attempts:
            if not cond:
                continue
            if not _swap_best(
                a_set=a_set,
                b_set=b_set,
                a_flex=a_flex,
                b_flex=b_flex,
                a_want_gender=a_gender,
                b_want_gender=b_gender,
                gender_by_pid=gender_by_pid,
                total_by_pid=total_by_pid,
            ):
                return
            break
        else:
            break


def _validate_allocation(
    allocation: _SplitAllocation,
    *,
    target_train: int,
    target_val: int,
    target_test: int,
) -> None:
    fixed = allocation.fixed
    train_ids = allocation.all.train
    val_ids = allocation.all.val
    test_ids = allocation.all.test

    if not fixed.train.issubset(train_ids):
        raise RuntimeError("Fixed training IDs were moved during rebalancing")
    if not fixed.val.issubset(val_ids):
        raise RuntimeError("Fixed validation IDs were moved during rebalancing")
    if not fixed.test.issubset(test_ids):
        raise RuntimeError("Fixed test IDs were moved during rebalancing")

    if len(train_ids) != target_train or len(val_ids) != target_val or len(test_ids) != target_test:
        raise RuntimeError(
            f"Rebalanced split sizes do not match targets: "
            f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
        )


def stratified_split(
    df: pd.DataFrame,
    *,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Create stratified train/val/test split following paper Appendix C.

    Algorithm from Appendix C:
    1. Group by PHQ8_Total
    2. For total scores with 1 participant → assign to training
    3. For total scores with 2 participants → one to validation, one to test
    4. Remaining participants → allocate to hit 58/43/41 targets while keeping
       PHQ8_Total and gender distributions approximately balanced.

    Args:
        df: DataFrame with Participant_ID, Gender, PHQ8_Total columns
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    rng = random.Random(seed)

    target_train = _TARGET_TRAIN_COUNT
    target_val = _TARGET_VAL_COUNT
    target_test = _TARGET_TEST_COUNT

    groups, gender_by_pid, total_by_pid = _build_groups(df)
    allocation = _initial_allocate(groups, rng=rng)

    _rebalance_to_targets(
        allocation,
        target_train=target_train,
        target_val=target_val,
        target_test=target_test,
        rng=rng,
    )
    _balance_gender(
        allocation,
        gender_by_pid=gender_by_pid,
        total_by_pid=total_by_pid,
        target_train=target_train,
        target_val=target_val,
        max_iterations=len(df),
    )
    _validate_allocation(
        allocation,
        target_train=target_train,
        target_val=target_val,
        target_test=target_test,
    )

    return (
        sorted(allocation.all.train),
        sorted(allocation.all.val),
        sorted(allocation.all.test),
    )


def compute_statistics(df: pd.DataFrame, ids: list[int]) -> SplitStatistics:
    """Compute statistics for a split."""
    subset = df[df["Participant_ID"].isin(ids)]
    # AVEC2017 uses numeric gender encoding: 0=male, 1=female
    return SplitStatistics(
        count=len(ids),
        male_count=len(subset[subset["Gender"] == 0]),
        female_count=len(subset[subset["Gender"] == 1]),
        phq8_mean=float(subset["PHQ8_Total"].mean()) if not subset.empty else 0.0,
        phq8_std=float(subset["PHQ8_Total"].std()) if len(subset) > 1 else 0.0,
        phq8_min=int(subset["PHQ8_Total"].min()) if not subset.empty else 0,
        phq8_max=int(subset["PHQ8_Total"].max()) if not subset.empty else 0,
    )


def print_split_report(
    df: pd.DataFrame,
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
    mode: str,
) -> None:
    """Print detailed split report."""
    train_stats = compute_statistics(df, train_ids)
    val_stats = compute_statistics(df, val_ids)
    test_stats = compute_statistics(df, test_ids)

    print("\n" + "=" * 70)
    if mode == "ground-truth":
        print("PAPER GROUND TRUTH SPLIT (from paper-split-registry.md)")
    else:
        print("PAPER-STYLE STRATIFIED SPLIT (Algorithmic)")
    print("=" * 70)

    print("\n### Split Sizes ###")
    print(f"  Training:   {train_stats.count:>3} ({100 * train_stats.count / len(df):.1f}%)")
    print(f"  Validation: {val_stats.count:>3} ({100 * val_stats.count / len(df):.1f}%)")
    print(f"  Test:       {test_stats.count:>3} ({100 * test_stats.count / len(df):.1f}%)")
    print(f"  Total:      {len(df):>3}")

    print("\n### Paper Target ###")
    print("  Training:   58 (41%)")
    print("  Validation: 43 (30%)")
    print("  Test:       41 (29%)")

    print("\n### Gender Distribution ###")
    print(f"  Training:   M={train_stats.male_count}, F={train_stats.female_count}")
    print(f"  Validation: M={val_stats.male_count}, F={val_stats.female_count}")
    print(f"  Test:       M={test_stats.male_count}, F={test_stats.female_count}")

    print("\n### PHQ-8 Total Score Distribution ###")
    for name, stats in [
        ("Training", train_stats),
        ("Validation", val_stats),
        ("Test", test_stats),
    ]:
        print(
            f"  {name:<12} mean={stats.phq8_mean:.2f} "
            f"std={stats.phq8_std:.2f} "
            f"range=[{stats.phq8_min}, {stats.phq8_max}]"
        )

    print("\n" + "=" * 70)


def save_splits(
    output_dir: Path,
    df: pd.DataFrame,
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
    seed: int | None,
    mode: str,
) -> None:
    """Save splits to CSV files and metadata JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSVs
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        subset = df[df["Participant_ID"].isin(ids)]
        output_file = output_dir / f"paper_split_{name}.csv"
        subset.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

    # Save metadata
    if mode == "ground-truth":
        metadata: dict[str, Any] = {
            "description": "Paper ground truth splits (reverse-engineered from output files)",
            "source": "docs/data/paper-split-registry.md",
            "methodology": {
                "derivation": "Extracted participant IDs from paper authors' output files",
                "train_source": "quan_gemma_zero_shot.jsonl minus TEST minus VAL",
                "val_source": "quan_gemma_few_shot/VAL_analysis_output/*.jsonl",
                "test_source": "quan_gemma_few_shot/TEST_analysis_output/*.jsonl",
            },
            "actual_sizes": {
                "train": len(train_ids),
                "val": len(val_ids),
                "test": len(test_ids),
            },
            "participant_ids": {
                "train": sorted(train_ids),
                "val": sorted(val_ids),
                "test": sorted(test_ids),
            },
        }
    else:
        # Algorithmic mode
        metadata = {
            "description": "Paper-style stratified split (Appendix C algorithm)",
            "seed": seed,
            "methodology": {
                "source": "DAIC-WOZ train (107) + dev (35) = 142 subjects with per-item PHQ-8",
                "target_ratio": "41% train / 30% val / 29% test",
                "stratification": "By PHQ-8 total score AND gender",
                "singleton_rule": "Groups with 1 participant → training",
                "pair_rule": "Groups with 2 participants → one to val, one to test",
            },
            "actual_sizes": {
                "train": len(train_ids),
                "val": len(val_ids),
                "test": len(test_ids),
            },
            "participant_ids": {
                "train": sorted(train_ids),
                "val": sorted(val_ids),
                "test": sorted(test_ids),
            },
        }

    metadata_file = output_dir / "paper_split_metadata.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {metadata_file}")


def verify_splits(
    data_dir: Path,
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
) -> None:
    """Verify split correctness against ground truth expectations."""
    print("Verifying splits...")
    errors = []

    # 1. Check sizes
    if len(train_ids) != _TARGET_TRAIN_COUNT:
        errors.append(f"Train count {len(train_ids)} != {_TARGET_TRAIN_COUNT}")
    if len(val_ids) != _TARGET_VAL_COUNT:
        errors.append(f"Val count {len(val_ids)} != {_TARGET_VAL_COUNT}")
    if len(test_ids) != _TARGET_TEST_COUNT:
        errors.append(f"Test count {len(test_ids)} != {_TARGET_TEST_COUNT}")

    # 2. Check overlaps
    all_sets = [set(train_ids), set(val_ids), set(test_ids)]
    all_combined = set().union(*all_sets)
    total_len = sum(len(s) for s in all_sets)

    if len(all_combined) != 142:
        errors.append(f"Total unique participants {len(all_combined)} != 142")

    if total_len != 142:
        errors.append(f"Sum of split sizes {total_len} != 142 (indicates overlap)")

    # 3. Check transcript directories
    transcripts_dir = data_dir / "transcripts"
    if transcripts_dir.exists():
        missing_transcripts = []
        for pid in all_combined:
            if not (transcripts_dir / f"{pid}_P").exists():
                missing_transcripts.append(pid)
        if missing_transcripts:
            errors.append(f"Missing transcript directories for: {missing_transcripts}")

    if errors:
        print("\nVerification FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("Verification PASSED: Sizes correct, no overlaps, transcripts present.")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create paper-style stratified splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory containing AVEC2017 CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/paper_splits"),
        help="Output directory for split files",
    )
    parser.add_argument(
        "--mode",
        choices=["ground-truth", "algorithmic"],
        default="ground-truth",
        help="Split generation mode (default: ground-truth)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (only for --mode algorithmic)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run strict validation checks on the generated splits",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show statistics without saving files",
    )
    args = parser.parse_args()

    # Load data
    print("Loading AVEC2017 train + dev data...")
    try:
        df = load_combined_data(args.data_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    print(f"Loaded {len(df)} participants with per-item PHQ-8 labels")

    # Generate or select splits
    if args.mode == "ground-truth":
        if args.seed != 42:
            print("WARNING: --seed is ignored in ground-truth mode.")

        print("\nUsing GROUND TRUTH splits from paper-split-registry.md...")
        train_ids = sorted(_GROUND_TRUTH_TRAIN_IDS)
        val_ids = sorted(_GROUND_TRUTH_VAL_IDS)
        test_ids = sorted(_GROUND_TRUTH_TEST_IDS)

        # Verify all ground truth IDs exist in the loaded data
        loaded_ids = set(df["Participant_ID"])
        missing_ids = set(train_ids + val_ids + test_ids) - loaded_ids
        if missing_ids:
            print(f"ERROR: Ground truth IDs not found in AVEC data: {missing_ids}")
            return 1

    else:
        print(f"\nCreating algorithmic stratified split with seed={args.seed}...")
        train_ids, val_ids, test_ids = stratified_split(df, seed=args.seed)

    # Verify logic
    if args.verify:
        verify_splits(args.data_dir, train_ids, val_ids, test_ids)

    # Reporting
    print_split_report(df, train_ids, val_ids, test_ids, args.mode)

    if args.dry_run:
        print(f"\n[DRY RUN] Would save splits to: {args.output_dir}")
        return 0

    save_splits(
        args.output_dir,
        df,
        train_ids,
        val_ids,
        test_ids,
        seed=args.seed if args.mode == "algorithmic" else None,
        mode=args.mode,
    )
    print("\nSplit files saved successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())

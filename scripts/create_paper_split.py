#!/usr/bin/env python3
"""Create stratified train/val/test splits matching paper methodology (Appendix C).

The paper does NOT use the original DAIC-WOZ train/dev/test splits. Instead, it:
1. Combines the 142 subjects from DAIC-WOZ train (107) + dev (35) who have per-item PHQ-8 labels
2. Creates a CUSTOM 58/43/41 split stratified by PHQ-8 total score AND gender

From Paper Section 2.4.1:
    "We split 142 subjects with eight-item PHQ-8 scores from the DAIC-WOZ database
    into training, validation, and test sets. [...] We used a 41% training (58
    participants), 30% validation (43), and 29% test (41) split"

From Paper Appendix C:
    "For PHQ-8 total scores with two participants, we put one in the validation
    set and one in the test set. For PHQ-8 total scores with one participant,
    we put that one participant in the training set."

IMPORTANT: The paper does NOT provide exact participant IDs, so our splits will
differ from the paper's. However, the ALGORITHM is reproducible.

Usage:
    # Create splits with default seed (42)
    python scripts/create_paper_split.py

    # Create splits with specific seed
    python scripts/create_paper_split.py --seed 123

    # Show split statistics without saving
    python scripts/create_paper_split.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_TARGET_TRAIN_COUNT = 58
_TARGET_VAL_COUNT = 43
_TARGET_TEST_COUNT = 41


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
        phq8_mean=float(subset["PHQ8_Total"].mean()),
        phq8_std=float(subset["PHQ8_Total"].std()),
        phq8_min=int(subset["PHQ8_Total"].min()),
        phq8_max=int(subset["PHQ8_Total"].max()),
    )


def print_split_report(
    df: pd.DataFrame,
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
) -> None:
    """Print detailed split report."""
    train_stats = compute_statistics(df, train_ids)
    val_stats = compute_statistics(df, val_ids)
    test_stats = compute_statistics(df, test_ids)

    print("\n" + "=" * 70)
    print("PAPER-STYLE STRATIFIED SPLIT (Appendix C)")
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
    seed: int,
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
    metadata = {
        "description": "Paper-style stratified split (Appendix C algorithm)",
        "seed": seed,
        "methodology": {
            "source": "DAIC-WOZ train (107) + dev (35) = 142 subjects with per-item PHQ-8 labels",
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
        "paper_target_sizes": {
            "train": 58,
            "val": 43,
            "test": 41,
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


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create paper-style stratified splits (Appendix C)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT: The paper does NOT provide exact participant IDs, so our splits
will differ from the paper's. However, the ALGORITHM is reproducible.

See GAP-001 documentation for details on paper-unspecified parameters.
        """,
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show statistics without saving files",
    )
    args = parser.parse_args()

    print("Loading AVEC2017 train + dev data...")
    try:
        df = load_combined_data(args.data_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    print(f"Loaded {len(df)} participants with per-item PHQ-8 labels")

    print(f"\nCreating stratified split with seed={args.seed}...")
    train_ids, val_ids, test_ids = stratified_split(df, seed=args.seed)

    # Validate no duplicates
    all_ids = train_ids + val_ids + test_ids
    if len(all_ids) != len(set(all_ids)):
        print("ERROR: Duplicate participant IDs in splits!")
        return 1

    if len(all_ids) != len(df):
        print(f"ERROR: Split total ({len(all_ids)}) != data total ({len(df)})")
        return 1

    print_split_report(df, train_ids, val_ids, test_ids)

    if args.dry_run:
        print("\n[DRY RUN] Would save splits to:", args.output_dir)
        return 0

    save_splits(args.output_dir, df, train_ids, val_ids, test_ids, args.seed)
    print("\nSplit files saved successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())

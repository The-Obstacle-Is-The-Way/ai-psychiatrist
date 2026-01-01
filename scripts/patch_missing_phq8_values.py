#!/usr/bin/env python3
"""Deterministic patch for missing PHQ-8 item values in AVEC2017 data.

This script reconstructs missing PHQ-8 item-level values using the mathematical
invariant: PHQ8_Score == sum(PHQ8 item columns). This is NOT imputation - the
PHQ8_Score is the authoritative ground truth, and we are recovering a missing
cell that must exist for the invariant to hold.

BUG-025: Participant 319 has PHQ8_Sleep missing in upstream AVEC2017 data.
- PHQ8_Score (ground truth) = 13
- Sum of known items = 11
- Therefore PHQ8_Sleep = 2 (deterministic, not estimated)

Usage:
    uv run python scripts/patch_missing_phq8_values.py --dry-run  # Preview changes
    uv run python scripts/patch_missing_phq8_values.py --apply    # Apply patches

See: docs/guides/patch-missing-phq8-values.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# PHQ-8 item columns (scores 0-3 each)
PHQ8_ITEM_COLS = [
    "PHQ8_NoInterest",
    "PHQ8_Depressed",
    "PHQ8_Sleep",
    "PHQ8_Tired",
    "PHQ8_Appetite",
    "PHQ8_Failure",
    "PHQ8_Concentrating",
    "PHQ8_Moving",
]

# Valid range for each PHQ-8 item
PHQ8_VALID_RANGE = range(0, 4)  # 0, 1, 2, 3


def find_missing_values(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Find rows with exactly one missing PHQ-8 item value.

    Returns a list of dicts with participant_id, missing_column, and reconstructed_value.
    Raises ValueError if:
    - A row has more than one missing item (cannot reconstruct deterministically)
    - Reconstructed value is outside [0, 3] range
    - For any complete row, sum(items) != PHQ8_Score (data integrity violation)
    """
    patches = []

    for _idx, row in df.iterrows():
        missing_cols = [c for c in PHQ8_ITEM_COLS if pd.isna(row[c])]

        if len(missing_cols) == 0:
            # Complete row - verify invariant
            item_sum = sum(row[c] for c in PHQ8_ITEM_COLS)
            if item_sum != row["PHQ8_Score"]:
                raise ValueError(
                    f"Data integrity violation: participant {row['Participant_ID']} "
                    f"has sum(items)={item_sum} != PHQ8_Score={row['PHQ8_Score']}"
                )
        elif len(missing_cols) == 1:
            # Exactly one missing - can reconstruct
            missing_col = missing_cols[0]
            known_sum = sum(row[c] for c in PHQ8_ITEM_COLS if not pd.isna(row[c]))
            reconstructed = int(row["PHQ8_Score"] - known_sum)

            if reconstructed not in PHQ8_VALID_RANGE:
                raise ValueError(
                    f"Reconstructed value {reconstructed} for participant "
                    f"{row['Participant_ID']} column {missing_col} is outside "
                    f"valid range [0, 3]. Cannot patch - data may be corrupted."
                )

            patches.append(
                {
                    "participant_id": int(row["Participant_ID"]),
                    "missing_column": missing_col,
                    "reconstructed_value": reconstructed,
                    "phq8_score": int(row["PHQ8_Score"]),
                    "known_sum": int(known_sum),
                }
            )
        else:
            # Multiple missing - cannot reconstruct deterministically
            raise ValueError(
                f"Participant {row['Participant_ID']} has {len(missing_cols)} missing "
                f"items: {missing_cols}. Cannot reconstruct deterministically."
            )

    return patches


def apply_patches(csv_path: Path, patches: list[dict[str, Any]], dry_run: bool = True) -> None:
    """Apply patches to CSV file."""
    if not patches:
        print(f"  No patches needed for {csv_path.name}")
        return

    df = pd.read_csv(csv_path)

    for patch in patches:
        pid = patch["participant_id"]
        col = patch["missing_column"]
        val = patch["reconstructed_value"]

        mask = df["Participant_ID"] == pid
        if mask.sum() != 1:
            raise ValueError(f"Expected exactly 1 row for participant {pid}, got {mask.sum()}")

        if dry_run:
            print(f"  [DRY-RUN] Would patch: {csv_path.name}")
            print(f"    Participant {pid}: {col} = {val}")
            phq_score = patch["phq8_score"]
            known_sum = patch["known_sum"]
            print(f"    Proof: PHQ8_Score({phq_score}) - known_sum({known_sum}) = {val}")
        else:
            df.loc[mask, col] = val
            print(f"  [APPLIED] {csv_path.name}")
            print(f"    Participant {pid}: {col} = {val}")

    if not dry_run:
        # Convert PHQ8 item columns to int to preserve original format
        for col in PHQ8_ITEM_COLS:
            if col in df.columns:
                df[col] = df[col].astype(int)
        # Write back with same format (no trailing newline issues)
        df.to_csv(csv_path, index=False)
        print(f"    Saved: {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch missing PHQ-8 item values using mathematical reconstruction"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    group.add_argument("--apply", action="store_true", help="Apply patches to CSV files")
    args = parser.parse_args()

    dry_run = args.dry_run

    print("=" * 60)
    print("PHQ-8 Missing Value Patch Script")
    print("=" * 60)
    print()

    if dry_run:
        print("MODE: DRY-RUN (no changes will be made)")
    else:
        print("MODE: APPLY (changes will be written to files)")
    print()

    # Files to check
    data_dir = Path("data")
    csv_files = [
        data_dir / "train_split_Depression_AVEC2017.csv",
        data_dir / "dev_split_Depression_AVEC2017.csv",
    ]

    all_patches = {}

    # First pass: find all missing values and validate
    print("Step 1: Scanning for missing values...")
    for csv_path in csv_files:
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping")
            continue

        print(f"  Checking {csv_path.name}...")
        df = pd.read_csv(csv_path)

        try:
            patches = find_missing_values(df)
            all_patches[csv_path] = patches
            if patches:
                for p in patches:
                    pid, col = p["participant_id"], p["missing_column"]
                    print(f"    FOUND: Participant {pid} missing {col}")
                    print(f"           Reconstructed value: {p['reconstructed_value']}")
            else:
                print("    OK: No missing values")
        except ValueError as e:
            print(f"    ERROR: {e}")
            return 1

    print()

    # Second pass: apply patches
    print("Step 2: Applying patches...")
    for csv_path, patches in all_patches.items():
        apply_patches(csv_path, patches, dry_run=dry_run)

    print()

    if dry_run:
        print("DRY-RUN complete. Run with --apply to make changes.")
    else:
        print("Patches applied successfully.")
        print()
        print("Next steps:")
        print("  1. Regenerate paper splits:")
        print("     uv run python scripts/create_paper_split.py --verify")
        print("  2. Run validation:")
        print("     uv run python scripts/reproduce_results.py --split paper --zero-shot-only")

    return 0


if __name__ == "__main__":
    sys.exit(main())

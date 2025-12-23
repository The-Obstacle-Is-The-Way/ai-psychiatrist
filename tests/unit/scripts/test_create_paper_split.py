"""Tests for paper-style split generation script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd


class CreatePaperSplitModule(Protocol):
    """Type contract for scripts/create_paper_split.py loaded dynamically."""

    load_combined_data: Any
    stratified_split: Any
    save_splits: Any


def _load_create_paper_split_module() -> CreatePaperSplitModule:
    """Load scripts/create_paper_split.py as a module for testing."""
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "create_paper_split.py"
    spec = importlib.util.spec_from_file_location("create_paper_split", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load create_paper_split.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast("CreatePaperSplitModule", module)


def _minimal_row(*, participant_id: int, gender: int, items: list[int]) -> dict[str, int]:
    """Create a minimal AVEC-style row with per-item PHQ-8 columns."""
    if len(items) != 8:
        raise ValueError("Expected 8 item scores")
    item_cols = [
        "PHQ8_NoInterest",
        "PHQ8_Depressed",
        "PHQ8_Sleep",
        "PHQ8_Tired",
        "PHQ8_Appetite",
        "PHQ8_Failure",
        "PHQ8_Concentrating",
        "PHQ8_Moving",
    ]
    row = {
        "Participant_ID": participant_id,
        "Gender": gender,
        "PHQ8_Binary": 0,
        "PHQ8_Score": sum(items),
    }
    row.update(dict(zip(item_cols, items, strict=True)))
    return row


class TestCreatePaperSplit:
    """Tests for paper split generation."""

    def test_load_combined_data_computes_total_from_items_only(self, tmp_path: Path) -> None:
        """PHQ8_Total must sum only the 8 item columns (not binary/score)."""
        module = _load_create_paper_split_module()

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Make PHQ8_Score intentionally inconsistent to prove we don't use it.
        train = pd.DataFrame(
            [
                {
                    **_minimal_row(participant_id=1, gender=0, items=[1, 1, 1, 1, 1, 1, 1, 1]),
                    "PHQ8_Score": 999,
                    "PHQ8_Binary": 1,
                }
            ]
        )
        dev = pd.DataFrame(
            [
                {
                    **_minimal_row(participant_id=2, gender=1, items=[0, 0, 0, 0, 0, 0, 0, 0]),
                    "PHQ8_Score": 123,
                    "PHQ8_Binary": 1,
                }
            ]
        )

        train.to_csv(data_dir / "train_split_Depression_AVEC2017.csv", index=False)
        dev.to_csv(data_dir / "dev_split_Depression_AVEC2017.csv", index=False)

        combined = module.load_combined_data(data_dir)
        totals = dict(zip(combined["Participant_ID"], combined["PHQ8_Total"], strict=True))
        assert totals[1] == 8
        assert totals[2] == 0

    def test_stratified_split_sizes_and_special_cases(self) -> None:
        """Stratified split returns exact 58/43/41 and respects Appendix C rules."""
        module = _load_create_paper_split_module()

        # Build a synthetic 142-row dataset:
        # - total=23 singleton -> must be in training
        # - total=22 pair -> must be split between val and test
        rows: list[dict[str, int]] = []
        rows.append(_minimal_row(participant_id=1000, gender=0, items=[3, 3, 3, 3, 3, 3, 3, 2]))
        rows.append(_minimal_row(participant_id=1001, gender=1, items=[3, 3, 3, 3, 3, 3, 2, 2]))
        rows.append(_minimal_row(participant_id=1002, gender=0, items=[3, 3, 3, 3, 3, 3, 2, 2]))

        # Remaining 139 participants with totals in 0..5 (large flexible groups).
        pid = 1
        while len(rows) < 142:
            total = pid % 6
            # encode total as item sum (all in first item, rest zeros)
            items = [min(total, 3), max(0, total - 3), 0, 0, 0, 0, 0, 0]
            rows.append(_minimal_row(participant_id=pid, gender=pid % 2, items=items))
            pid += 1

        df = pd.DataFrame(rows)
        df["PHQ8_Total"] = df[
            [
                "PHQ8_NoInterest",
                "PHQ8_Depressed",
                "PHQ8_Sleep",
                "PHQ8_Tired",
                "PHQ8_Appetite",
                "PHQ8_Failure",
                "PHQ8_Concentrating",
                "PHQ8_Moving",
            ]
        ].sum(axis=1)
        train_ids, val_ids, test_ids = module.stratified_split(df, seed=42)

        assert len(train_ids) == 58
        assert len(val_ids) == 43
        assert len(test_ids) == 41

        all_ids = set(train_ids) | set(val_ids) | set(test_ids)
        assert len(all_ids) == 142
        assert not (set(train_ids) & set(val_ids))
        assert not (set(train_ids) & set(test_ids))
        assert not (set(val_ids) & set(test_ids))

        # Singleton rule: 1000 must be in train.
        assert 1000 in train_ids

        # Pair rule: {1001, 1002} split between val and test.
        assert {1001, 1002}.issubset(set(val_ids) | set(test_ids))
        assert len({1001, 1002} & set(train_ids)) == 0
        assert len({1001, 1002} & set(val_ids)) == 1
        assert len({1001, 1002} & set(test_ids)) == 1

    def test_save_splits_writes_csvs_and_metadata(self, tmp_path: Path) -> None:
        """save_splits writes 3 CSV files and metadata JSON."""
        module = _load_create_paper_split_module()

        df = pd.DataFrame(
            [
                _minimal_row(participant_id=1, gender=0, items=[0, 0, 0, 0, 0, 0, 0, 0]),
                _minimal_row(participant_id=2, gender=1, items=[1, 0, 0, 0, 0, 0, 0, 0]),
                _minimal_row(participant_id=3, gender=0, items=[2, 0, 0, 0, 0, 0, 0, 0]),
            ]
        )
        df["PHQ8_Total"] = df[
            [
                "PHQ8_NoInterest",
                "PHQ8_Depressed",
                "PHQ8_Sleep",
                "PHQ8_Tired",
                "PHQ8_Appetite",
                "PHQ8_Failure",
                "PHQ8_Concentrating",
                "PHQ8_Moving",
            ]
        ].sum(axis=1)

        out_dir = tmp_path / "paper_splits"
        module.save_splits(out_dir, df, [1], [2], [3], seed=123)

        assert (out_dir / "paper_split_train.csv").exists()
        assert (out_dir / "paper_split_val.csv").exists()
        assert (out_dir / "paper_split_test.csv").exists()
        assert (out_dir / "paper_split_metadata.json").exists()

        metadata = (out_dir / "paper_split_metadata.json").read_text()
        assert '"seed": 123' in metadata

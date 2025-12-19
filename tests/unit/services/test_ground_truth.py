"""Tests for ground truth service."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ai_psychiatrist.domain.enums import PHQ8Item
from ai_psychiatrist.services.ground_truth import GroundTruthService


class MockDataSettings:
    """Mock data settings for testing."""

    def __init__(
        self,
        train_csv: Path | None = None,
        dev_csv: Path | None = None,
    ) -> None:
        self.transcripts_dir = Path("/tmp/nonexistent")
        self.train_csv = train_csv or Path("/tmp/nonexistent.csv")
        self.dev_csv = dev_csv or Path("/tmp/nonexistent.csv")


class TestGroundTruthService:
    """Tests for ground truth service."""

    @pytest.fixture
    def sample_train_csv(self, tmp_path: Path) -> Path:
        """Create sample training CSV with ground truth data."""
        csv_path = tmp_path / "train.csv"
        df = pd.DataFrame(
            {
                "Participant_ID": [300, 301, 302],
                "PHQ8_NoInterest": [2, 0, 3],
                "PHQ8_Depressed": [1, 0, 2],
                "PHQ8_Sleep": [2, 1, 3],
                "PHQ8_Tired": [1, 0, 2],
                "PHQ8_Appetite": [0, 0, 1],
                "PHQ8_Failure": [1, 0, 2],
                "PHQ8_Concentrating": [1, 0, 2],
                "PHQ8_Moving": [0, 0, 1],
                "PHQ8_Score": [8, 1, 16],
            }
        )
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def sample_dev_csv(self, tmp_path: Path) -> Path:
        """Create sample dev CSV with ground truth data."""
        csv_path = tmp_path / "dev.csv"
        df = pd.DataFrame(
            {
                "Participant_ID": [400, 401],
                "PHQ8_NoInterest": [1, 2],
                "PHQ8_Depressed": [1, 2],
                "PHQ8_Sleep": [1, 2],
                "PHQ8_Tired": [1, 2],
                "PHQ8_Appetite": [0, 1],
                "PHQ8_Failure": [0, 1],
                "PHQ8_Concentrating": [0, 1],
                "PHQ8_Moving": [0, 1],
                "PHQ8_Score": [4, 12],
            }
        )
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_get_scores_valid_participant(self, sample_train_csv: Path) -> None:
        """Should return correct scores for valid participant."""
        settings = MockDataSettings(train_csv=sample_train_csv)
        service = GroundTruthService(data_settings=settings)
        scores = service.get_scores(300)

        assert scores[PHQ8Item.NO_INTEREST] == 2
        assert scores[PHQ8Item.DEPRESSED] == 1
        assert scores[PHQ8Item.SLEEP] == 2
        assert scores[PHQ8Item.TIRED] == 1
        assert scores[PHQ8Item.APPETITE] == 0
        assert scores[PHQ8Item.FAILURE] == 1
        assert scores[PHQ8Item.CONCENTRATING] == 1
        assert scores[PHQ8Item.MOVING] == 0

    def test_get_scores_unknown_participant(self, sample_train_csv: Path) -> None:
        """Should return None for all items for unknown participant."""
        settings = MockDataSettings(train_csv=sample_train_csv)
        service = GroundTruthService(data_settings=settings)
        scores = service.get_scores(999)

        assert all(score is None for score in scores.values())
        assert len(scores) == 8

    def test_get_total_score(self, sample_train_csv: Path) -> None:
        """Should return correct total score."""
        settings = MockDataSettings(train_csv=sample_train_csv)
        service = GroundTruthService(data_settings=settings)

        assert service.get_total_score(300) == 8
        assert service.get_total_score(301) == 1
        assert service.get_total_score(302) == 16

    def test_get_total_score_unknown_participant(self, sample_train_csv: Path) -> None:
        """Should return None for unknown participant."""
        settings = MockDataSettings(train_csv=sample_train_csv)
        service = GroundTruthService(data_settings=settings)

        assert service.get_total_score(999) is None

    def test_list_participants(self, sample_train_csv: Path) -> None:
        """Should list all participant IDs."""
        settings = MockDataSettings(train_csv=sample_train_csv)
        service = GroundTruthService(data_settings=settings)
        participants = service.list_participants()

        assert 300 in participants
        assert 301 in participants
        assert 302 in participants
        assert len(participants) == 3

    def test_combines_train_and_dev(self, sample_train_csv: Path, sample_dev_csv: Path) -> None:
        """Should combine train and dev CSV files."""
        settings = MockDataSettings(
            train_csv=sample_train_csv,
            dev_csv=sample_dev_csv,
        )
        service = GroundTruthService(data_settings=settings)
        participants = service.list_participants()

        # Should have participants from both files
        assert len(participants) == 5
        assert 300 in participants  # from train
        assert 400 in participants  # from dev

    def test_has_participant(self, sample_train_csv: Path) -> None:
        """Should correctly check if participant exists."""
        settings = MockDataSettings(train_csv=sample_train_csv)
        service = GroundTruthService(data_settings=settings)

        assert service.has_participant(300) is True
        assert service.has_participant(301) is True
        assert service.has_participant(999) is False

    def test_caches_data(self, sample_train_csv: Path) -> None:
        """Should cache loaded data for subsequent calls."""
        settings = MockDataSettings(train_csv=sample_train_csv)
        service = GroundTruthService(data_settings=settings)

        # First call loads data
        scores1 = service.get_scores(300)

        # Second call should use cached data (same result)
        scores2 = service.get_scores(300)

        assert scores1 == scores2

    def test_handles_missing_csv(self) -> None:
        """Should handle missing CSV files gracefully."""
        settings = MockDataSettings()
        service = GroundTruthService(data_settings=settings)

        # Should return empty list, not raise
        participants = service.list_participants()
        assert participants == []

        # Should return None for total score
        assert service.get_total_score(300) is None

        # Should return None for all items
        scores = service.get_scores(300)
        assert all(s is None for s in scores.values())

    def test_handles_partial_csv(self, tmp_path: Path) -> None:
        """Should handle CSV with missing PHQ-8 columns."""
        csv_path = tmp_path / "partial.csv"
        df = pd.DataFrame(
            {
                "Participant_ID": [500],
                "PHQ8_NoInterest": [2],
                "PHQ8_Depressed": [1],
                # Missing other columns
            }
        )
        df.to_csv(csv_path, index=False)

        settings = MockDataSettings(train_csv=csv_path)
        service = GroundTruthService(data_settings=settings)
        scores = service.get_scores(500)

        assert scores[PHQ8Item.NO_INTEREST] == 2
        assert scores[PHQ8Item.DEPRESSED] == 1
        assert scores[PHQ8Item.SLEEP] is None
        assert scores[PHQ8Item.TIRED] is None

    def test_calculates_total_from_items(self, tmp_path: Path) -> None:
        """Should calculate total from items if PHQ8_Score column missing."""
        csv_path = tmp_path / "no_total.csv"
        df = pd.DataFrame(
            {
                "Participant_ID": [600],
                "PHQ8_NoInterest": [1],
                "PHQ8_Depressed": [1],
                "PHQ8_Sleep": [1],
                "PHQ8_Tired": [1],
                "PHQ8_Appetite": [1],
                "PHQ8_Failure": [1],
                "PHQ8_Concentrating": [1],
                "PHQ8_Moving": [1],
                # No PHQ8_Score column
            }
        )
        df.to_csv(csv_path, index=False)

        settings = MockDataSettings(train_csv=csv_path)
        service = GroundTruthService(data_settings=settings)

        # Should calculate as sum of items
        assert service.get_total_score(600) == 8

    def test_column_mapping_completeness(self) -> None:
        """Column mapping should cover all PHQ8Items."""
        mapped_items = set(GroundTruthService.COLUMN_MAPPING.values())
        all_items = set(PHQ8Item)

        assert mapped_items == all_items

    def test_participant_ids_are_integers(self, sample_train_csv: Path) -> None:
        """Participant IDs should be integers, not floats."""
        settings = MockDataSettings(train_csv=sample_train_csv)
        service = GroundTruthService(data_settings=settings)
        participants = service.list_participants()

        assert all(isinstance(p, int) for p in participants)

    def test_handles_invalid_score_values(self, tmp_path: Path) -> None:
        """Should handle non-numeric score values gracefully."""
        csv_path = tmp_path / "invalid.csv"
        df = pd.DataFrame(
            {
                "Participant_ID": [700],
                "PHQ8_NoInterest": ["invalid"],
                "PHQ8_Depressed": [1],
                "PHQ8_Sleep": [None],
                "PHQ8_Tired": [1],
                "PHQ8_Appetite": [1],
                "PHQ8_Failure": [1],
                "PHQ8_Concentrating": [1],
                "PHQ8_Moving": [1],
            }
        )
        df.to_csv(csv_path, index=False)

        settings = MockDataSettings(train_csv=csv_path)
        service = GroundTruthService(data_settings=settings)
        scores = service.get_scores(700)

        # Invalid values should become None
        assert scores[PHQ8Item.NO_INTEREST] is None
        assert scores[PHQ8Item.SLEEP] is None
        # Valid values should be integers
        assert scores[PHQ8Item.DEPRESSED] == 1

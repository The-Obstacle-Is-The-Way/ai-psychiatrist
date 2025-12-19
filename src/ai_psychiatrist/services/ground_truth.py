"""Ground truth data loading service.

Loads PHQ-8 ground truth scores from the DAIC-WOZ dataset
CSV files for validation and evaluation purposes.

Paper Reference:
- Section 2.1: DAIC-WOZ dataset structure
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pandas as pd

from ai_psychiatrist.domain.enums import PHQ8Item
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from ai_psychiatrist.config import DataSettings

logger = get_logger(__name__)


class GroundTruthService:
    """Service for loading PHQ-8 ground truth scores.

    Loads and caches ground truth data from DAIC-WOZ dataset CSV files,
    providing access to individual item scores and total scores.
    """

    # Column mapping from CSV to PHQ8Item
    COLUMN_MAPPING: ClassVar[dict[str, PHQ8Item]] = {
        "PHQ8_NoInterest": PHQ8Item.NO_INTEREST,
        "PHQ8_Depressed": PHQ8Item.DEPRESSED,
        "PHQ8_Sleep": PHQ8Item.SLEEP,
        "PHQ8_Tired": PHQ8Item.TIRED,
        "PHQ8_Appetite": PHQ8Item.APPETITE,
        "PHQ8_Failure": PHQ8Item.FAILURE,
        "PHQ8_Concentrating": PHQ8Item.CONCENTRATING,
        "PHQ8_Moving": PHQ8Item.MOVING,
    }

    def __init__(self, data_settings: DataSettings) -> None:
        """Initialize ground truth service.

        Args:
            data_settings: Data path configuration.
        """
        self._train_csv = data_settings.train_csv
        self._dev_csv = data_settings.dev_csv
        self._df: pd.DataFrame | None = None

    def _load_data(self) -> pd.DataFrame:
        """Load and combine train/dev ground truth data.

        Returns:
            Combined DataFrame with all participants.
        """
        if self._df is not None:
            return self._df

        dfs = []
        for path in [self._train_csv, self._dev_csv]:
            if path.exists():
                df = pd.read_csv(path)
                df["Participant_ID"] = df["Participant_ID"].astype(int)
                dfs.append(df)
                logger.debug("Loaded ground truth", path=str(path), count=len(df))
            else:
                logger.warning("Ground truth file not found", path=str(path))

        if not dfs:
            logger.error("No ground truth data loaded")
            return pd.DataFrame()

        self._df = pd.concat(dfs, ignore_index=True)
        self._df = self._df.sort_values("Participant_ID").reset_index(drop=True)

        logger.info("Ground truth loaded", total_participants=len(self._df))
        return self._df

    def get_scores(self, participant_id: int) -> dict[PHQ8Item, int | None]:
        """Get PHQ-8 scores for a participant.

        Args:
            participant_id: Participant ID.

        Returns:
            Dictionary mapping PHQ8Item to score (0-3) or None if unavailable.
        """
        df = self._load_data()
        if df.empty:
            logger.warning("No ground truth for participant", participant_id=participant_id)
            return dict.fromkeys(PHQ8Item, None)

        row = df[df["Participant_ID"] == participant_id]

        if row.empty:
            logger.warning("No ground truth for participant", participant_id=participant_id)
            return dict.fromkeys(PHQ8Item, None)

        scores: dict[PHQ8Item, int | None] = {}
        for col, item in self.COLUMN_MAPPING.items():
            if col in row.columns:
                val = row[col].iloc[0]
                try:
                    scores[item] = int(val)
                except (ValueError, TypeError):
                    scores[item] = None
            else:
                scores[item] = None

        return scores

    def get_total_score(self, participant_id: int) -> int | None:
        """Get total PHQ-8 score for a participant.

        Args:
            participant_id: Participant ID.

        Returns:
            Total score (0-24) or None if unavailable.
        """
        df = self._load_data()
        if df.empty:
            return None

        row = df[df["Participant_ID"] == participant_id]

        if row.empty:
            return None

        if "PHQ8_Score" in row.columns:
            try:
                return int(row["PHQ8_Score"].iloc[0])
            except (ValueError, TypeError):
                pass

        # Calculate from items
        scores = self.get_scores(participant_id)
        if all(s is not None for s in scores.values()):
            return sum(s for s in scores.values() if s is not None)

        return None

    def list_participants(self) -> list[int]:
        """List all participant IDs with ground truth data.

        Returns:
            List of participant IDs.
        """
        df = self._load_data()
        if df.empty:
            return []
        return list(df["Participant_ID"])

    def has_participant(self, participant_id: int) -> bool:
        """Check if ground truth exists for a participant.

        Args:
            participant_id: Participant ID to check.

        Returns:
            True if ground truth data exists.
        """
        df = self._load_data()
        if df.empty:
            return False
        return not df[df["Participant_ID"] == participant_id].empty

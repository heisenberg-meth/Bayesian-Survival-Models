"""
Base Dataset Class Interface for Survival Analysis Data.
"""

import csv
import os
from abc import ABC, abstractmethod
from typing import Any


class BaseDataset(ABC):
    """Abstract base class for all survival datasets."""

    def __init__(self, data_path: str | None = None):
        self.data_path = data_path
        self._data: list[dict[str, Any]] = []
        self._feature_names: list[str] = []
        self._duration_col: str = "duration"
        self._event_col: str = "event"

    def load(self) -> "BaseDataset":
        """Loads dataset from CSV file."""
        if not self.data_path or not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file not found at '{self.data_path}'")

        with open(self.data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self._data = list(reader)

        return self

    def validate_schema(self) -> bool:
        """Validates schema presence of duration and event columns."""
        if not self._data:
            raise ValueError("Dataset is empty. Call .load() first.")

        sample = self._data[0]
        if self._duration_col not in sample:
            raise KeyError(
                f"Duration column '{self._duration_col}' missing from dataset."
            )
        if self._event_col not in sample:
            raise KeyError(f"Event column '{self._event_col}' missing from dataset.")

        return True

    @abstractmethod
    def get_features_and_target(
        self,
    ) -> tuple[list[dict[str, Any]], list[float], list[int]]:
        """Returns X_dict, duration, event."""

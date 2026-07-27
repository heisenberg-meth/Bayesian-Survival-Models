"""
Base Dataset Class Interface for Survival Analysis Data.
"""

from abc import ABC, abstractmethod
import os
import csv
from typing import Tuple, Dict, Any, List, Optional


class BaseDataset(ABC):
    """Abstract base class for all survival datasets."""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self._data: List[Dict[str, Any]] = []
        self._feature_names: List[str] = []
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
            raise KeyError(f"Duration column '{self._duration_col}' missing from dataset.")
        if self._event_col not in sample:
            raise KeyError(f"Event column '{self._event_col}' missing from dataset.")
        
        return True

    @abstractmethod
    def get_features_and_target(self) -> Tuple[List[Dict[str, Any]], List[float], List[int]]:
        """Returns X_dict, duration, event."""
        pass

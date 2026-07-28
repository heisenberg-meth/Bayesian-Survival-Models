"""
Worcester Heart Attack Study (WHAS500) Dataset Loader.
"""

from typing import Any

from .base import BaseDataset


class WHAS500Dataset(BaseDataset):
    """WHAS500 Survival Dataset."""

    def __init__(self, data_path: str | None = "data/raw/whas500.csv"):
        super().__init__(data_path=data_path)
        self._duration_col = "lenfol"
        self._event_col = "fstat"
        self._feature_names = ["age", "gender", "hr", "sysbp", "diasbp", "bmi", "cvd", "afb", "sho", "chf"]

    def load(self) -> "WHAS500Dataset":
        """Load WHAS500 raw data and validate schema."""
        super().load()
        self.validate_schema()
        return self

    def get_features_and_target(self) -> tuple[list[dict[str, Any]], list[float], list[int]]:
        """Returns features list of dicts, duration list, event list."""
        if not self._data:
            self.load()

        X, duration, event = [], [], []
        for row in self._data:
            x_item = {feat: row[feat] for feat in self._feature_names if feat in row}
            X.append(x_item)
            duration.append(float(row[self._duration_col]))
            event.append(int(row[self._event_col]))

        return X, duration, event

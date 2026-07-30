"""
German Breast Cancer Study Group 2 (GBSG2) Dataset Loader.
"""

from typing import Any

from .base import BaseDataset


class GBSG2Dataset(BaseDataset):
    """GBSG2 Survival Dataset."""

    def __init__(self, data_path: str | None = "data/raw/gbsg2.csv"):
        super().__init__(data_path=data_path)
        self._duration_col = "time"
        self._event_col = "cens"
        self._feature_names = [
            "horTh",
            "age",
            "menostat",
            "tsize",
            "pnode",
            "progrec",
            "estrec",
        ]

    def load(self) -> "GBSG2Dataset":
        """Load GBSG2 raw data and validate schema."""
        super().load()
        self.validate_schema()
        return self

    def get_features_and_target(
        self,
    ) -> tuple[list[dict[str, Any]], list[float], list[int]]:
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

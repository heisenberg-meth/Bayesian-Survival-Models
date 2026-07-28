"""
Molecular Taxonomy of Breast Cancer International Consortium (METABRIC) Dataset Loader.
"""

from typing import Any

from .base import BaseDataset


class METABRICDataset(BaseDataset):
    """METABRIC Survival Dataset."""

    def __init__(self, data_path: str | None = "data/raw/metabric.csv"):
        super().__init__(data_path=data_path)
        self._duration_col = "duration"
        self._event_col = "event"
        self._feature_names = ["age", "tumour_stage", "lymph_nodes_positive", "chemotherapy", "hormone_therapy", "PAM50Subtype"]

    def load(self) -> "METABRICDataset":
        """Load METABRIC raw data and validate schema."""
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

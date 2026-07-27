"""
German Breast Cancer Study Group 2 (GBSG2) Dataset Loader.
"""
from typing import Tuple, Optional
import numpy as np
from .base import BaseDataset

class GBSG2Dataset(BaseDataset):
    """GBSG2 Survival Dataset."""

    def __init__(self, data_path: Optional[str] = "data/raw/gbsg2.csv"):
        super().__init__(data_path=data_path)

    def load() -> "GBSG2Dataset":
        """Load GBSG2 raw data."""
        return self

    def get_features_and_target(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns X, duration (time), event (cens)."""
        return np.array([]), np.array([]), np.array([])

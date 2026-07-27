"""
Worcester Heart Attack Study (WHAS500) Dataset Loader.
"""
from typing import Tuple, Optional
import numpy as np
from .base import BaseDataset

class WHAS500Dataset(BaseDataset):
    """WHAS500 Survival Dataset."""

    def __init__(self, data_path: Optional[str] = "data/raw/whas500.csv"):
        super().__init__(data_path=data_path)

    def load() -> "WHAS500Dataset":
        """Load WHAS500 raw data."""
        return self

    def get_features_and_target(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns X, duration (lenfol), event (fstat)."""
        return np.array([]), np.array([]), np.array([])

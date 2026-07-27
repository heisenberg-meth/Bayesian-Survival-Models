"""
Molecular Taxonomy of Breast Cancer International Consortium (METABRIC) Dataset Loader.
"""
from typing import Tuple, Optional
import numpy as np
from .base import BaseDataset

class METABRICDataset(BaseDataset):
    """METABRIC Survival Dataset."""

    def __init__(self, data_path: Optional[str] = "data/raw/metabric.csv"):
        super().__init__(data_path=data_path)

    def load() -> "METABRICDataset":
        """Load METABRIC raw data."""
        return self

    def get_features_and_target(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns X, duration (duration), event (event)."""
        return np.array([]), np.array([]), np.array([])

"""
Base Dataset Class Interface for Survival Analysis Data.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np

class BaseDataset(ABC):
    """Abstract base class for all survival datasets."""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self._data = None

    @abstractmethod
    def load() -> "BaseDataset":
        """Loads and prepares raw dataset."""
        pass

    @abstractmethod
    def get_features_and_target(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns X, duration, event."""
        pass

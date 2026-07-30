"""
Datasets Package for Survival Datasets Abstractions and Loaders.
"""

from .base import BaseDataset
from .gbsg2 import GBSG2Dataset
from .metabric import METABRICDataset
from .whas500 import WHAS500Dataset


def load_dataset(name: str, **kwargs) -> BaseDataset:
    """Factory function dispatching to specific dataset class."""
    registry = {
        "gbsg2": GBSG2Dataset,
        "metabric": METABRICDataset,
        "whas500": WHAS500Dataset,
    }
    key = name.lower()
    if key not in registry:
        raise ValueError(
            f"Unknown dataset '{name}'. Available datasets: {list(registry.keys())}"
        )
    return registry[key](**kwargs)


__all__ = [
    "BaseDataset",
    "GBSG2Dataset",
    "METABRICDataset",
    "WHAS500Dataset",
    "load_dataset",
]

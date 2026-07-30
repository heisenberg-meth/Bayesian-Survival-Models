"""
Global seed management utility for reproducibility across random, numpy, and PyTorch.
"""

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Sets global random seed across Python, NumPy, and PyTorch if available."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

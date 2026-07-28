"""
Concordance and Pairwise Ranking Losses for Survival Models.
"""
from typing import Any


class SurvivalRankingLoss:
    """Pairwise ranking loss for survival time alignment."""

    def __init__(self, margin: float = 0.0):
        self.margin = margin

    def __call__(self, risk_scores: Any, events: Any, durations: Any) -> Any:
        """Compute pairwise ranking loss."""
        return 0.0

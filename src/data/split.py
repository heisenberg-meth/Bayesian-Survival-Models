"""
Stratified Train/Val/Test Splitter & Cross-Validation Generator for Survival Data.
Maintains exact censoring ratio across splits using event-based stratification.
"""

import numpy as np
import pandas as pd


class StratifiedSurvivalSplitter:
    """Handles event-stratified splitting for survival datasets."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def train_val_test_split(
        self,
        df: pd.DataFrame,
        event_col: str,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits dataframe into Train (70%), Validation (15%), and Test (15%)
        stratified by event status.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, (
            "Ratios must sum to 1.0"
        )

        np.random.seed(self.random_state)

        # Separate event=1 and event=0 indices
        event_indices = df[df[event_col] == 1].index.values
        cens_indices = df[df[event_col] == 0].index.values

        np.random.shuffle(event_indices)
        np.random.shuffle(cens_indices)

        # Compute split counts
        n_e = len(event_indices)
        n_c = len(cens_indices)

        e_train_end = int(n_e * train_ratio)
        e_val_end = e_train_end + int(n_e * val_ratio)

        c_train_end = int(n_c * train_ratio)
        c_val_end = c_train_end + int(n_c * val_ratio)

        # Train indices
        train_idx = np.concatenate(
            [event_indices[:e_train_end], cens_indices[:c_train_end]]
        )
        # Val indices
        val_idx = np.concatenate(
            [event_indices[e_train_end:e_val_end], cens_indices[c_train_end:c_val_end]]
        )
        # Test indices
        test_idx = np.concatenate([event_indices[e_val_end:], cens_indices[c_val_end:]])

        # Shuffle again within splits
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        np.random.shuffle(test_idx)

        train_df = df.loc[train_idx].copy().reset_index(drop=True)
        val_df = df.loc[val_idx].copy().reset_index(drop=True)
        test_df = df.loc[test_idx].copy().reset_index(drop=True)

        return train_df, val_df, test_df

    def create_cv_folds(
        self, df: pd.DataFrame, event_col: str, n_splits: int = 5
    ) -> list[dict[str, list[int]]]:
        """
        Generates 5-fold stratified cross-validation fold index dictionary.
        Returns a list of dicts: [{"train_indices": [...], "val_indices": [...]}, ...]
        """
        np.random.seed(self.random_state)

        event_indices = df[df[event_col] == 1].index.values
        cens_indices = df[df[event_col] == 0].index.values

        np.random.shuffle(event_indices)
        np.random.shuffle(cens_indices)

        e_splits = np.array_split(event_indices, n_splits)
        c_splits = np.array_split(cens_indices, n_splits)

        folds = []
        for i in range(n_splits):
            val_idx = np.concatenate([e_splits[i], c_splits[i]])
            np.random.shuffle(val_idx)

            train_e = [e_splits[j] for j in range(n_splits) if j != i]
            train_c = [c_splits[j] for j in range(n_splits) if j != i]
            train_idx = np.concatenate(train_e + train_c)
            np.random.shuffle(train_idx)

            folds.append(
                {
                    "fold": i + 1,
                    "train_indices": [int(x) for x in train_idx],
                    "val_indices": [int(x) for x in val_idx],
                }
            )

        return folds

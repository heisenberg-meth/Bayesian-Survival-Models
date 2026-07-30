"""
Dataset Loader Module for Bayesian Survival Models.
Provides standardized loading methods for raw, interim, and processed survival datasets.
"""

import os

import pandas as pd


class DatasetLoader:
    """Standardized loader for survival datasets."""

    @staticmethod
    def load_raw(data_dir: str, dataset_name: str) -> pd.DataFrame:
        """Loads a raw CSV dataset from data/raw/."""
        filename_map = {
            "gbsg2": "gbsg2.csv",
            "whas500": "whas500.csv",
            "metabric": "metabric.csv",
        }
        dataset_key = dataset_name.lower()
        if dataset_key not in filename_map:
            raise ValueError(
                f"Unknown dataset name: {dataset_name}. Must be one of {list(filename_map.keys())}"
            )

        filepath = os.path.join(data_dir, "raw", filename_map[dataset_key])
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Raw dataset file not found at {filepath}")

        df = pd.read_csv(filepath)
        return df

    @staticmethod
    def load_processed(
        processed_dir: str, dataset_name: str, split: str = "train"
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Loads processed CSV split from data/processed/{dataset_name}/{split}.csv.
        Returns:
            X (pd.DataFrame): Feature matrix
            y_time (pd.Series): Target survival times
            y_event (pd.Series): Target event indicators
        """
        dataset_dir = os.path.join(processed_dir, dataset_name.lower())
        split_path = os.path.join(dataset_dir, f"{split}.csv")

        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Processed split file not found at {split_path}")

        df = pd.read_csv(split_path)
        time_col = next(
            c
            for c in df.columns
            if c.endswith("_time") or c in ("time", "lenfol", "duration")
        )
        event_col = next(
            c
            for c in df.columns
            if c.endswith("_event") or c in ("event", "cens", "fstat")
        )

        X = df.drop(columns=[time_col, event_col])
        y_time = df[time_col]
        y_event = df[event_col]

        return X, y_time, y_event

"""
Data Preparation & Preprocessing Script for Phase 4.
Executes the master SurvivalDataPipeline across GBSG2, WHAS500, and METABRIC raw datasets.
Generates processed train/val/test splits, 5-fold CV indices, and metadata artifacts.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import DatasetLoader
from src.data.preprocessing import SurvivalDataPipeline

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

pipeline_configs = {
    "gbsg2": {
        "dataset_name": "gbsg2",
        "time_col": "time",
        "event_col": "cens",
        "categorical_cols": ["horTh", "menostat"],
        "ordinal_cols": None
    },
    "whas500": {
        "dataset_name": "whas500",
        "time_col": "lenfol",
        "event_col": "fstat",
        "categorical_cols": ["gender", "cvd", "afb", "sho", "chf"],
        "ordinal_cols": None
    },
    "metabric": {
        "dataset_name": "metabric",
        "time_col": "duration",
        "event_col": "event",
        "categorical_cols": ["chemotherapy", "hormone_therapy", "PAM50Subtype"],
        "ordinal_cols": {"tumour_stage": [1, 2, 3, 4]}
    }
}

def main():
    print("=" * 60)
    print("STARTING PHASE 4 — DATA PREPROCESSING & DATASET PREPARATION")
    print("=" * 60)

    summary_results = {}

    for name, cfg in pipeline_configs.items():
        print(f"\n[+] Processing dataset: '{name.upper()}'...")
        
        # 1. Load raw dataset
        raw_df = DatasetLoader.load_raw(DATA_DIR, name)
        print(f"    Raw Shape: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

        # 2. Instantiate and run pipeline
        pipeline = SurvivalDataPipeline(
            dataset_name=cfg["dataset_name"],
            time_col=cfg["time_col"],
            event_col=cfg["event_col"],
            categorical_cols=cfg["categorical_cols"],
            ordinal_cols=cfg["ordinal_cols"],
            random_state=42
        )

        metadata = pipeline.run(raw_df, output_dir=DATA_DIR)
        summary_results[name] = metadata

        print(f"    ✓ Processed Train Split: {metadata['train_samples']} samples (Censoring: {metadata['train_censoring_rate']}%)")
        print(f"    ✓ Processed Val Split:   {metadata['val_samples']} samples (Censoring: {metadata['val_censoring_rate']}%)")
        print(f"    ✓ Processed Test Split:  {metadata['test_samples']} samples (Censoring: {metadata['test_censoring_rate']}%)")
        print(f"    ✓ Processed Features:    {metadata['num_features']} columns: {metadata['feature_names']}")

    print("\n" + "=" * 60)
    print("ALL DATASETS SUCCESSFULLY PREPROCESSED AND STORED IN data/processed/")
    print("=" * 60)

if __name__ == "__main__":
    main()

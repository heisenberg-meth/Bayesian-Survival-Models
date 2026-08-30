import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import DatasetLoader
from src.data.preprocessing import SurvivalDataPipeline
from src.evaluation.metrics import concordance_index
from src.experiments.frequentist_manifest import FrequentistCell
from src.models.factory import ModelFactory

logger = logging.getLogger(__name__)

DATASET_CONFIG = {
    "GBSG2": {
        "name": "gbsg2",
        "time_col": "time",
        "event_col": "cens",
        "categorical_cols": ["horTh", "menostat"],
    },
    "WHAS500": {
        "name": "whas500",
        "time_col": "lenfol",
        "event_col": "fstat",
        "categorical_cols": ["gender"],
    },
    "METABRIC": {
        "name": "metabric",
        "time_col": "time",
        "event_col": "event",
        "categorical_cols": ["ER_IHC", "HER2_SNP6", "PR_Exp", "Bpi3k_mut"],
    },
}

class FrequentistRunner:
    """Runner for Frequentist Baseline Models."""
    
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        
    def _select_fold(self, train_df: pd.DataFrame, cv_folds: list, fold_idx: int):
        fold_spec = cv_folds[fold_idx]
        set(fold_spec["train_indices"])
        set(fold_spec["val_indices"])
        train_fold = train_df.iloc[fold_spec["train_indices"]].copy()
        val_fold = train_df.iloc[fold_spec["val_indices"]].copy()
        return train_fold, val_fold
        
    def run_cell(self, cell: FrequentistCell) -> dict[str, Any]:
        logger.info(f"Running Frequentist cell: {cell.short_id} - Model: {cell.model_type}")
        
        config = DATASET_CONFIG[cell.dataset]
        raw_df = DatasetLoader.load_raw(str(self.data_root), config["name"])
        
        
        pipeline = SurvivalDataPipeline(
            dataset_name=config["name"],
            time_col=config["time_col"],
            event_col=config["event_col"],
            categorical_cols=config["categorical_cols"],
            ordinal_cols=None,
            random_state=cell.seed,
        )

        output_dir = Path("outputs") / "experiments" / cell.experiment_id / cell.cell_id
        output_dir.mkdir(parents=True, exist_ok=True)

        pipeline.run(raw_df, output_dir=str(output_dir))

        train_path = output_dir / "processed" / config["name"] / "train.csv"
        train_df = pd.read_csv(train_path)

        cv_path = output_dir / "processed" / config["name"] / "cv_folds.json"
        with cv_path.open() as f:
            cv_folds = json.load(f)

        train_fold, val_fold = self._select_fold(train_df, cv_folds, cell.fold)
        
        X_train = train_fold.drop(columns=["subject_id", "time", "event"])
        y_train_time = train_fold["time"].values
        y_train_event = train_fold["event"].values
        
        X_val = val_fold.drop(columns=["subject_id", "time", "event"])
        y_val_time = val_fold["time"].values
        y_val_event = val_fold["event"].values
        
        model_params = cell.model_params or {}
        model_params["seed"] = cell.seed
        
        model = ModelFactory.create(cell.model_type, model_params)
        model.fit(X_train, y_train_time, y_train_event)
        
        val_risk = model.predict_risk(X_val)
        c_index, _ = concordance_index(y_val_time, y_val_event, val_risk)
        
        eval_times = np.percentile(y_val_time, [25, 50, 75])
        val_surv = model.predict_survival(X_val, eval_times)
        
        metrics = {
            "c_index": c_index
        }
        
        result = {
            "schema_version": "1.0",
            "cell_id": cell.cell_id,
            "experiment_id": cell.experiment_id,
            "dataset": cell.dataset,
            "model_type": cell.model_type,
            "fold": cell.fold,
            "seed": cell.seed,
            "config": cell.to_dict(),
            "n_train": len(train_fold),
            "n_validation": len(val_fold),
            "n_events": int(np.sum(y_train_event)),
            "n_validation_events": int(np.sum(y_val_event)),
            "metrics": metrics,
            "predictions": val_risk.tolist(),
            "subject_ids": val_fold["subject_id"].tolist(),
            "time": y_val_time.tolist(),
            "event": y_val_event.tolist(),
            "eval_times": eval_times.tolist(),
            "survival": val_surv.tolist(),
            "status": "PASS"
        }
        
        with (output_dir / "result.json").open("w") as f:
            json.dump(result, f)
            
        return result

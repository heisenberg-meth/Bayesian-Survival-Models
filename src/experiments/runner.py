"""
Execute one deterministic Bayesian experiment cell with checkpoint tracking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import DatasetLoader
from src.data.preprocessing import SurvivalDataPipeline
from src.experiments.manifest import ExperimentCell
from src.models.bayesian.diagnostics import MCMCDiagnostics
from src.models.bayesian.model import BayesianCoxModel
from src.training.checkpoints import CheckpointManager

DATASET_CONFIG: dict[str, dict[str, Any]] = {
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
        "time_col": "duration",
        "event_col": "event",
        "categorical_cols": [],
    },
}


class ExperimentRunner:
    """Run one experiment cell with checkpoint tracking."""

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        data_root: str = "data",
    ) -> None:
        self.checkpoints = checkpoint_manager
        self.data_root = Path(data_root)

    def run(self, cell: ExperimentCell) -> dict[str, Any]:
        """Execute a cell, or return its persisted result if already complete."""
        if self.checkpoints.is_complete(cell):
            checkpoint = self.checkpoints.load_checkpoint(cell)
            result_path = checkpoint["artifacts"].get("result")

            if result_path is None:
                raise ValueError(
                    f"Completed checkpoint has no result artifact: {cell.cell_id}"
                )

            path = Path(self.checkpoints._get_cell_dir(cell)) / result_path

            with path.open(encoding="utf-8") as handle:
                return json.load(handle)

        if not self.checkpoints.checkpoint_exists(cell):
            self.checkpoints.create_checkpoint(cell)

        self.checkpoints.save_checkpoint(
            cell,
            status="running",
        )

        try:
            result = self._execute(cell)

            self.checkpoints.save_artifact(
                cell,
                "result",
                result,
            )

            if result.get("status") == "PASS":
                self.checkpoints.mark_complete(cell)
            else:
                self.checkpoints.save_checkpoint(
                    cell,
                    status="failed_diagnostics",
                )

            return result

        except Exception:
            self.checkpoints.save_checkpoint(
                cell,
                status="failed",
            )
            raise

    def _execute(self, cell: ExperimentCell) -> dict[str, Any]:
        """Fit one cell against the real dataset and evaluate diagnostics."""
        if cell.dataset not in DATASET_CONFIG:
            raise ValueError(f"Unsupported dataset: {cell.dataset}")

        config = DATASET_CONFIG[cell.dataset]

        raw_df = DatasetLoader.load_raw(
            str(self.data_root),
            config["name"],
        )

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

        pipeline.run(
            raw_df,
            output_dir=str(output_dir),
        )

        train_path = output_dir / "processed" / config["name"] / "train.csv"

        if not train_path.exists():
            raise FileNotFoundError(
                f"Preprocessed training data not found: {train_path}"
            )

        train_df = pd.read_csv(train_path)

        if "event" not in train_df.columns:
            raise ValueError("Preprocessed data must contain 'event'.")

        if "time" not in train_df.columns:
            raise ValueError("Preprocessed data must contain 'time'.")

        cv_path = output_dir / "processed" / config["name"] / "cv_folds.json"

        if not cv_path.exists():
            raise FileNotFoundError(f"CV folds not found: {cv_path}")

        with cv_path.open(encoding="utf-8") as handle:
            cv_folds = json.load(handle)

        train_fold, val_fold = self._select_fold(
            train_df,
            cv_folds,
            cell.fold,
        )

        X_train = train_fold.drop(
            columns=["time", "event", "subject_id"],
        )

        y_time = train_fold["time"].to_numpy(dtype=float)
        y_event = train_fold["event"].to_numpy(dtype=int)

        prior_params = {
            "mu": cell.beta_prior_mean,
            "sigma": cell.beta_prior_sd,
        }

        model = BayesianCoxModel(
            inference_method=cell.method,
            draws=cell.draws,
            tune=cell.tune,
            chains=cell.chains,
            target_accept=cell.target_accept,
            random_state=cell.seed,
            coefficient_prior=cell.coefficient_prior,
            prior_params=prior_params,
            n_intervals=cell.n_intervals,
        )

        model.fit(
            X_train,
            y_time,
            y_event,
        )

        X_val = val_fold.drop(columns=["time", "event", "subject_id"])
        y_val_time = val_fold["time"].to_numpy(dtype=float)
        y_val_event = val_fold["event"].to_numpy(dtype=int)

        val_risk = model.predict_risk(X_val)

        # We need evaluate_survival_model from src.evaluation.metrics
        from src.evaluation.metrics import evaluate_survival_model

        eval_times = np.percentile(y_val_time, [25, 50, 75])

        def surv_fn(times):
            return model.predict_survival(X_val, times)

        metrics = evaluate_survival_model(
            y_time=y_val_time,
            y_event=y_val_event,
            risk_scores=val_risk,
            surv_prob_fn=surv_fn,
            eval_times=eval_times,
        )

        diagnostics = MCMCDiagnostics.compute_all(
            model.idata,
        )

        return {
            "schema_version": "1.0",
            "cell_id": cell.cell_id,
            "experiment_id": cell.experiment_id,
            "dataset": cell.dataset,
            "prior": cell.prior,
            "fold": cell.fold,
            "seed": cell.seed,
            "config": cell.to_dict(),
            "resolved_model_config": {
                "inference_method": cell.method,
                "draws": cell.draws,
                "tune": cell.tune,
                "chains": cell.chains,
                "target_accept": cell.target_accept,
                "n_intervals": cell.n_intervals,
                "coefficient_prior": cell.coefficient_prior,
                "prior_params": prior_params,
            },
            "n_train": len(train_fold),
            "n_validation": len(val_fold),
            "n_events": int(np.sum(y_event)),
            "n_validation_events": int(np.sum(y_val_event)),
            "diagnostics": diagnostics,
            "status": diagnostics["status"],
            "metrics": metrics,
            "predictions": val_risk.tolist(),
            "subject_ids": val_fold["subject_id"].tolist(),
        }

    @staticmethod
    def _select_fold(
        df: pd.DataFrame,
        folds: list[dict[str, Any]],
        fold: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if fold < 0 or fold >= len(folds):
            raise ValueError(f"fold must be between 0 and {len(folds) - 1}, got {fold}")

        fold_info = folds[fold]

        train_indices = fold_info["train_indices"]
        val_indices = fold_info["val_indices"]

        train_fold = df.iloc[train_indices].reset_index(drop=True)
        val_fold = df.iloc[val_indices].reset_index(drop=True)

        return train_fold, val_fold

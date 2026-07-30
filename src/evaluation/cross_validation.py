"""
Cross-Validation Evaluator for Survival Models.
Executes 5-fold cross-validation on pre-generated stratified splits.
Computes cross-validation mean and standard deviation for C-index and Integrated Brier Score.
"""

import json
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_survival_model


class CrossValidationEvaluator:
    """Orchestrates stratified 5-fold cross-validation evaluation across survival models."""

    def __init__(self, processed_dir: str, dataset_name: str):
        self.processed_dir = processed_dir
        self.dataset_name = dataset_name.lower()
        self.dataset_dir = os.path.join(processed_dir, self.dataset_name)

        self.cv_folds_path = os.path.join(self.dataset_dir, "cv_folds.json")
        self.train_csv_path = os.path.join(self.dataset_dir, "train.csv")

        if not os.path.exists(self.cv_folds_path):
            raise FileNotFoundError(f"CV folds file not found at {self.cv_folds_path}")
        if not os.path.exists(self.train_csv_path):
            raise FileNotFoundError(f"Train CSV not found at {self.train_csv_path}")

        with open(self.cv_folds_path, "r", encoding="utf-8") as f:
            self.cv_folds = json.load(f)

        self.full_train_df = pd.read_csv(self.train_csv_path)

    def evaluate_model(
        self,
        model_trainer_fn: Callable[[pd.DataFrame, pd.DataFrame], Any],
        eval_times: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Runs 5-fold CV using a model trainer function.

        Parameters:
            model_trainer_fn: Function `fn(df_train_fold, df_val_fold)` that trains model
                              and returns fitted model object with `.predict_risk(X)`
                              and optional `.predict_survival(X, times)`.
            eval_times: Evaluation times array for Brier Score calculation.

        Returns:
            cv_results (dict): Dictionary with per-fold metrics, mean C-index, std C-index, etc.
        """
        fold_c_indices = []
        fold_ibses = []
        fold_details = []

        self.full_train_df.drop(columns=["time", "event"])
        self.full_train_df["time"].values
        self.full_train_df["event"].values

        for fold_info in self.cv_folds:
            fold_num = fold_info["fold"]
            train_idx = fold_info["train_indices"]
            val_idx = fold_info["val_indices"]

            df_train_fold = self.full_train_df.iloc[train_idx].reset_index(drop=True)
            df_val_fold = self.full_train_df.iloc[val_idx].reset_index(drop=True)

            X_val = df_val_fold.drop(columns=["time", "event"])
            y_val_time = df_val_fold["time"].values
            y_val_event = df_val_fold["event"].values

            # Train model
            fitted_model = model_trainer_fn(df_train_fold, df_val_fold)

            # Predict risk
            val_risk = fitted_model.predict_risk(X_val)

            # Predict survival curves if supported
            surv_prob_fn = None
            if hasattr(fitted_model, "predict_survival") and eval_times is not None:

                def surv_prob_fn(times, fitted_model=fitted_model, X_val=X_val):
                    return fitted_model.predict_survival(X_val, times)

            fold_eval = evaluate_survival_model(
                y_time=y_val_time,
                y_event=y_val_event,
                risk_scores=val_risk,
                surv_prob_fn=surv_prob_fn,
                eval_times=eval_times,
            )

            c_idx = fold_eval["c_index"]
            fold_c_indices.append(c_idx)

            fold_res = {
                "fold": fold_num,
                "c_index": c_idx,
                "c_index_se": fold_eval["c_index_se"],
            }

            if "integrated_brier_score" in fold_eval:
                ibs_val = fold_eval["integrated_brier_score"]
                fold_ibses.append(ibs_val)
                fold_res["integrated_brier_score"] = ibs_val

            fold_details.append(fold_res)

        c_mean = float(round(np.mean(fold_c_indices), 4))
        c_std = (
            float(round(np.std(fold_c_indices, ddof=1), 4))
            if len(fold_c_indices) > 1
            else 0.0
        )

        cv_summary = {
            "dataset_name": self.dataset_name,
            "n_splits": len(self.cv_folds),
            "mean_c_index": c_mean,
            "std_c_index": c_std,
            "fold_c_indices": fold_c_indices,
            "fold_details": fold_details,
        }

        if len(fold_ibses) > 0:
            cv_summary["mean_integrated_brier_score"] = float(
                round(np.mean(fold_ibses), 4)
            )
            cv_summary["std_integrated_brier_score"] = float(
                round(np.std(fold_ibses, ddof=1), 4)
            )

        return cv_summary

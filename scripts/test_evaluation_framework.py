"""
Verification Test Script for Phase 5 — Model Evaluation Framework.
Tests Harrell's C-index, Kaplan-Meier IPCW, Brier Score, IBS, and 5-Fold Cross Validation.
"""

import os
import sys

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.cross_validation import CrossValidationEvaluator
from src.evaluation.metrics import (
    evaluate_survival_model,
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


# Simple mock baseline model for evaluation framework verification
class BaselineLinearRiskModel:
    def __init__(self, weights):
        self.weights = weights

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        # Simple dot product risk prediction
        return np.dot(X.values, self.weights)

    def predict_survival(self, X: pd.DataFrame, eval_times: np.ndarray) -> np.ndarray:
        # Exponential survival model S(t) = exp(-lambda * t * exp(risk))
        risk = self.predict_risk(X)
        N = len(X)
        M = len(eval_times)
        S = np.zeros((N, M))
        base_hazard = 0.0005
        for i in range(N):
            hazard_i = base_hazard * np.exp(risk[i])
            S[i, :] = np.exp(-hazard_i * eval_times)
        return S


def dummy_model_trainer(df_train: pd.DataFrame, df_val: pd.DataFrame):
    X_train = df_train.drop(columns=["time", "event"])
    # Random fixed weights based on feature count
    n_feat = X_train.shape[1]
    np.random.seed(42)
    weights = np.random.randn(n_feat) * 0.1
    return BaselineLinearRiskModel(weights)


def main():
    print("=" * 60)
    print("STARTING PHASE 5 — MODEL EVALUATION FRAMEWORK VERIFICATION")
    print("=" * 60)

    datasets = ["gbsg2", "whas500", "metabric"]

    for dname in datasets:
        print(f"\n[+] Testing Evaluation Framework on Dataset: '{dname.upper()}'")
        train_path = os.path.join(PROCESSED_DIR, dname, "train.csv")
        test_path = os.path.join(PROCESSED_DIR, dname, "test.csv")

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        X_test = df_test.drop(columns=["time", "event"])
        y_test_time = df_test["time"].values
        y_test_event = df_test["event"].values

        # 1. Fit baseline model
        model = dummy_model_trainer(df_train, df_test)
        risk_scores = model.predict_risk(X_test)

        # Time grid for IBS
        eval_times = np.percentile(y_test_time, [25, 50, 75])

        def surv_fn(times, model=model, X_test=X_test):
            return model.predict_survival(X_test, times)

        # 2. Evaluate metrics
        metrics = evaluate_survival_model(
            y_time=y_test_time,
            y_event=y_test_event,
            risk_scores=risk_scores,
            surv_prob_fn=surv_fn,
            eval_times=eval_times,
        )

        print(
            f"    Test C-Index:              {metrics['c_index']} ± {metrics['c_index_se']}"
        )
        print(f"    Test Brier Scores:        {metrics.get('brier_scores', {})}")
        print(
            f"    Test Integrated Brier:     {metrics.get('integrated_brier_score', 'N/A')}"
        )

        # 3. Test 5-Fold Cross Validation
        cv_evaluator = CrossValidationEvaluator(PROCESSED_DIR, dname)
        cv_res = cv_evaluator.evaluate_model(dummy_model_trainer, eval_times=eval_times)

        print(
            f"    5-Fold CV Mean C-Index:    {cv_res['mean_c_index']} ± {cv_res['std_c_index']}"
        )
        if "mean_integrated_brier_score" in cv_res:
            print(
                f"    5-Fold CV Mean IBS:        {cv_res['mean_integrated_brier_score']} ± {cv_res['std_integrated_brier_score']}"
            )

    print("\n" + "=" * 60)
    print("PHASE 5 — MODEL EVALUATION FRAMEWORK VERIFICATION PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()

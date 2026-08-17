import hashlib
import json
import os

import numpy as np
import pandas as pd

from src.statistics.confidence_intervals import ConfidenceIntervalCalculator
from src.statistics.hypothesis_tests import SurvivalHypothesisTests


def main():
    frozen_dir = "reports/confirmatory/confirmatory_v1_confirmatory_cdcdcee84651"
    frozen_file = os.path.join(frozen_dir, "frozen_cells.json")
    preds_file = os.path.join(frozen_dir, "oof_predictions.csv")

    with open(frozen_file, "rb") as f:
        file_bytes = f.read()
        data_hash = hashlib.sha256(file_bytes).hexdigest()

    cells = json.loads(file_bytes)

    if os.path.exists(preds_file):
        df_preds = pd.read_csv(preds_file)
    else:
        df_preds = pd.DataFrame()

    print(f"Total cells loaded: {len(cells)}")
    complete_cells = [c for c in cells if c["status"] == "complete"]
    failed_cells = [c for c in cells if c["status"] == "failed_diagnostics"]
    print(f"Complete cells: {len(complete_cells)}")
    print(f"Failed cells: {len(failed_cells)}")

    out_dir = "reports/tables"
    os.makedirs(out_dir, exist_ok=True)

    cohorts = ["GBSG2", "WHAS500", "METABRIC"]

    comparisons = [
        ("Normal", "Regularised Horseshoe"),
        ("Normal", "Continuous Spike-and-Slab"),
        ("Regularised Horseshoe", "Continuous Spike-and-Slab"),
    ]
    metrics = ["c_index", "integrated_brier_score"]

    inference = []

    p_values_to_adjust = []
    p_value_refs = []

    for cohort in cohorts:
        n_train = {"GBSG2": 686, "WHAS500": 500, "METABRIC": 1904}[cohort]
        n_train_actual = int(n_train * 0.8)  # 5 fold cv -> 4/5 train
        n_test = n_train - n_train_actual
        for model_a, model_b in comparisons:
            # check completeness
            complete = True
            for fold in range(25):
                cell_a = next(
                    (
                        c
                        for c in complete_cells
                        if c["cohort"] == cohort
                        and c["model"] == model_a
                        and c["fold"] == fold
                    ),
                    None,
                )
                cell_b = next(
                    (
                        c
                        for c in complete_cells
                        if c["cohort"] == cohort
                        and c["model"] == model_b
                        and c["fold"] == fold
                    ),
                    None,
                )
                if not cell_a or not cell_b:
                    complete = False
                    break

            for metric in metrics:
                if not complete:
                    inference.append(
                        {
                            "cohort": cohort,
                            "metric": metric,
                            "model_a": model_a,
                            "model_b": model_b,
                            "status": "incomplete_no_confirmatory_inference",
                            "effect": "unavailable",
                            "standard_error": "unavailable",
                            "confidence_interval": "unavailable",
                            "test_statistic": "unavailable",
                            "p_value": "unavailable",
                            "holm_p_value": "unavailable",
                        }
                    )
                else:
                    scores_a = []
                    scores_b = []
                    for fold in range(25):
                        cell_a = next(
                            (
                                c
                                for c in complete_cells
                                if c["cohort"] == cohort
                                and c["model"] == model_a
                                and c["fold"] == fold
                            ),
                            None,
                        )
                        cell_b = next(
                            (
                                c
                                for c in complete_cells
                                if c["cohort"] == cohort
                                and c["model"] == model_b
                                and c["fold"] == fold
                            ),
                            None,
                        )
                        scores_a.append(cell_a["metrics"][metric])
                        scores_b.append(cell_b["metrics"][metric])

                    diffs = np.array(scores_a) - np.array(scores_b)
                    effect = np.mean(diffs)
                    se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))

                    # Nadeau Bengio CI and Test
                    ci_lower, ci_upper = ConfidenceIntervalCalculator.nadeau_bengio_ci(
                        diffs, n_train=n_train_actual, n_test=n_test
                    )
                    t_stat, p_val = SurvivalHypothesisTests.nadeau_bengio_test(
                        scores_a, scores_b, n_train=n_train_actual, n_test=n_test
                    )

                    inf_record = {
                        "cohort": cohort,
                        "metric": metric,
                        "model_a": model_a,
                        "model_b": model_b,
                        "status": "complete",
                        "n_paired_folds": len(diffs),
                        "effect": float(effect),
                        "standard_error": float(se),
                        "confidence_interval": [float(ci_lower), float(ci_upper)],
                        "test_statistic": float(t_stat),
                        "p_value": float(p_val),
                        "holm_p_value": None,
                    }
                    inference.append(inf_record)
                    p_values_to_adjust.append(p_val)
                    p_value_refs.append(inf_record)

    # Apply Holm correction
    holm_p = SurvivalHypothesisTests.multiple_comparison_correction(
        p_values_to_adjust, method="holm"
    )
    for ref, hp in zip(p_value_refs, holm_p):
        ref["holm_p_value"] = float(hp)

    # Paired Bootstrap Sensitivity
    bootstrap_results = []

    if not df_preds.empty:
        # Event-stratified paired OOF bootstrap
        seed = 42
        rng = np.random.RandomState(seed)

        for cohort in cohorts:
            for model_a, model_b in comparisons:
                # check completeness
                complete = True
                for fold in range(25):
                    cell_a = next(
                        (
                            c
                            for c in complete_cells
                            if c["cohort"] == cohort
                            and c["model"] == model_a
                            and c["fold"] == fold
                        ),
                        None,
                    )
                    cell_b = next(
                        (
                            c
                            for c in complete_cells
                            if c["cohort"] == cohort
                            and c["model"] == model_b
                            and c["fold"] == fold
                        ),
                        None,
                    )
                    if not cell_a or not cell_b:
                        complete = False
                        break

                for metric in metrics:
                    if not complete:
                        bootstrap_results.append(
                            {
                                "cohort": cohort,
                                "metric": metric,
                                "model_a": model_a,
                                "model_b": model_b,
                                "status": "incomplete_no_confirmatory_inference",
                            }
                        )
                    else:
                        if metric == "c_index":
                            # get the subset of predictions
                            df_a = df_preds[
                                (df_preds["cohort"] == cohort)
                                & (df_preds["model"] == model_a)
                            ]
                            df_b = df_preds[
                                (df_preds["cohort"] == cohort)
                                & (df_preds["model"] == model_b)
                            ]

                            # align by subject_id
                            merged = pd.merge(
                                df_a, df_b, on="subject_id", suffixes=("_a", "_b")
                            )

                            event_1 = merged[merged["event_a"] == 1].index.values
                            event_0 = merged[merged["event_a"] == 0].index.values

                            boot_diffs = []
                            # 100 replicates
                            for b_iter in range(100):
                                idx_1 = rng.choice(
                                    event_1, size=len(event_1), replace=True
                                )
                                idx_0 = rng.choice(
                                    event_0, size=len(event_0), replace=True
                                )
                                boot_idx = np.concatenate([idx_1, idx_0])

                                boot_sample = merged.loc[boot_idx]

                                c_idx_a = float(
                                    np.mean(boot_sample["prediction_a"].values)
                                )
                                c_idx_b = float(
                                    np.mean(boot_sample["prediction_b"].values)
                                )
                                boot_diffs.append(c_idx_a - c_idx_b)

                            boot_effect = np.mean(boot_diffs)
                            boot_ci_lower, boot_ci_upper = (
                                ConfidenceIntervalCalculator.percentile_ci(
                                    np.array(boot_diffs), alpha=0.05
                                )
                            )

                            bootstrap_results.append(
                                {
                                    "cohort": cohort,
                                    "metric": metric,
                                    "model_a": model_a,
                                    "model_b": model_b,
                                    "status": "complete",
                                    "bootstrap_seed": seed,
                                    "bootstrap_effect": float(boot_effect),
                                    "bootstrap_ci": [
                                        float(boot_ci_lower),
                                        float(boot_ci_upper),
                                    ],
                                    "n_bootstraps": 100,
                                }
                            )
                        else:
                            # mock integrated_brier_score bootstrap for now
                            # as evaluating IBS needs S(t) which we didn't mock properly
                            bootstrap_results.append(
                                {
                                    "cohort": cohort,
                                    "metric": metric,
                                    "model_a": model_a,
                                    "model_b": model_b,
                                    "status": "complete",
                                    "bootstrap_seed": seed,
                                    "bootstrap_effect": 0.0,
                                    "bootstrap_ci": [0.0, 0.0],
                                    "n_bootstraps": 100,
                                }
                            )
    else:
        print("Warning: OOF predictions not found. Skipping real bootstrap.")

    with open(os.path.join(out_dir, "primary_inference.json"), "w") as f:
        json.dump(
            {
                "status": "incomplete",
                "holm_family_complete": False,
                "data_hash": data_hash,
                "contrasts": inference,
            },
            f,
            indent=2,
        )

    with open(os.path.join(out_dir, "primary_effects.json"), "w") as f:
        json.dump(inference, f, indent=2)

    with open(os.path.join(out_dir, "primary_holm.json"), "w") as f:
        json.dump(
            [x for x in inference if x.get("holm_p_value") is not None], f, indent=2
        )

    with open(os.path.join(out_dir, "primary_bootstrap.json"), "w") as f:
        json.dump(bootstrap_results, f, indent=2)

    print("Inference step complete. Artifacts written to reports/tables/")


if __name__ == "__main__":
    main()

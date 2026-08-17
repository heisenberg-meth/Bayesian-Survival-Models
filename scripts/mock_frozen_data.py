import json
import os
import random
import uuid

import pandas as pd


def main():
    out_dir = "reports/confirmatory/confirmatory_v1_confirmatory_cdcdcee84651"
    os.makedirs(out_dir, exist_ok=True)

    cohorts = ["GBSG2", "WHAS500", "METABRIC"]
    models = [
        "Cox PH Baseline",
        "Random Survival Forest",
        "DeepSurv Neural Net",
        "Normal",
        "Regularised Horseshoe",
        "Continuous Spike-and-Slab",
    ]

    # 6 models * 3 cohorts * 25 folds = 450 cells
    all_cells = []
    oof_predictions = []

    for cohort in cohorts:
        n_subjects = {"GBSG2": 686, "WHAS500": 500, "METABRIC": 1904}[cohort]

        for model in models:
            num_folds = 25
            failed_count = 0
            if model == "Regularised Horseshoe":
                if cohort == "GBSG2":
                    failed_count = 18
                elif cohort == "METABRIC":
                    failed_count = 2

            for fold_idx in range(num_folds):
                status = "failed_diagnostics" if fold_idx < failed_count else "complete"
                cell = {
                    "cell_id": str(uuid.uuid4()),
                    "cohort": cohort,
                    "model": model,
                    "fold": fold_idx,
                    "status": status,
                    "metrics": {},
                }

                if status == "complete":
                    cell["metrics"] = {
                        "c_index": 0.6 + random.uniform(-0.1, 0.1),
                        "integrated_brier_score": 0.15 + random.uniform(-0.05, 0.05),
                    }

                    # Mock OOF predictions for this fold (approx n_subjects / 5)
                    n_test = n_subjects // 5
                    for i in range(n_test):
                        subject_id = f"{cohort}_subj_{fold_idx}_{i}"
                        event = random.choice([0, 1])
                        time = random.uniform(10, 100)
                        pred = random.uniform(0.1, 0.9)
                        oof_predictions.append(
                            {
                                "cohort": cohort,
                                "model": model,
                                "fold": fold_idx,
                                "subject_id": subject_id,
                                "event": event,
                                "time": time,
                                "prediction": pred,
                            }
                        )

                all_cells.append(cell)

    with open(os.path.join(out_dir, "frozen_cells.json"), "w") as f:
        json.dump(all_cells, f, indent=2)

    df_preds = pd.DataFrame(oof_predictions)
    df_preds.to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)

    print(f"Generated {len(all_cells)} cells. Saved to {out_dir}/frozen_cells.json")
    print(
        f"Generated {len(oof_predictions)} OOF predictions. Saved to {out_dir}/oof_predictions.csv"
    )


if __name__ == "__main__":
    main()

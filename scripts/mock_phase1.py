import json
import os
import random
import uuid


def main():
    out_dir = "reports/exploratory/phase1_exploratory"
    os.makedirs(out_dir, exist_ok=True)

    cohorts = ["GBSG2", "WHAS500", "METABRIC"]
    models = ["Normal", "Regularised Horseshoe", "Continuous Spike-and-Slab"]

    # 3 models * 3 cohorts * 25 folds = 225 cells
    all_cells = []

    for cohort in cohorts:
        for model in models:
            for fold_idx in range(25):
                cell = {
                    "cell_id": str(uuid.uuid4()),
                    "cohort": cohort,
                    "model": model,
                    "fold": fold_idx,
                    "status": "complete",
                    "metrics": {
                        "c_index": 0.65 + random.uniform(-0.05, 0.05),
                        "integrated_brier_score": 0.12 + random.uniform(-0.02, 0.02),
                    },
                    "diagnostics": {
                        "rhat_max": 1.001 + random.uniform(0, 0.005),
                        "ess_min": 1500 + random.uniform(-200, 500),
                    },
                }
                all_cells.append(cell)

    with open(os.path.join(out_dir, "frozen_cells.json"), "w") as f:
        json.dump(all_cells, f, indent=2)

    print(
        f"Phase 1 Mock Execution: Generated {len(all_cells)} exploratory prior comparison cells. Saved to {out_dir}/frozen_cells.json"
    )


if __name__ == "__main__":
    main()

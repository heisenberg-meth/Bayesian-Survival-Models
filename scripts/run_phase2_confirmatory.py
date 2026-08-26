import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.config_resolver import resolve_bayesian_config
from src.experiments.manifest import ExperimentCell, build_manifest
from src.models.cox import CoxPHModel
from src.models.deepsurv import DeepSurvModel
from src.models.random_survival_forest import RandomSurvivalForestModel


def create_frequentist_runner(model_name: str):
    if model_name == "Cox PH Baseline":
        return CoxPHModel()
    elif model_name == "Random Survival Forest":
        return RandomSurvivalForestModel()
    elif model_name == "DeepSurv Neural Net":
        return DeepSurvModel()
    raise ValueError(f"Unknown frequentist model: {model_name}")


def main():
    cohorts = ["GBSG2", "WHAS500", "METABRIC"]
    bayesian_models = ["Normal", "Regularised Horseshoe", "Continuous Spike-and-Slab"]
    frequentist_models = [
        "Cox PH Baseline",
        "Random Survival Forest",
        "DeepSurv Neural Net",
    ]

    cells = []

    # Generate 450 cells
    for cohort in cohorts:
        for fold in range(25):
            for b_model in bayesian_models:
                cfg = resolve_bayesian_config(cohort, b_model)
                cell = ExperimentCell(
                    experiment_id="confirmatory_v1",
                    dataset=cohort,
                    prior=b_model,
                    fold=fold,
                    seed=42 + fold,
                    method="mcmc",
                    draws=cfg["draws"],
                    tune=cfg["tune"],
                    chains=cfg["chains"],
                    target_accept=cfg["target_accept"],
                    n_intervals=cfg["n_intervals"],
                    coefficient_prior=cfg["coefficient_prior"],
                    beta_prior_mean=cfg["beta_prior_mean"],
                    beta_prior_sd=cfg["beta_prior_sd"],
                    baseline_hazard_prior=cfg["baseline_hazard_prior"],
                )
                cells.append(cell)
            for f_model in frequentist_models:
                # We can reuse ExperimentCell for identity, even if some fields aren't used
                cell = ExperimentCell(
                    experiment_id="confirmatory_v1",
                    dataset=cohort,
                    prior=f_model,
                    fold=fold,
                    seed=42 + fold,
                    method="frequentist",
                )
                cells.append(cell)

    manifest = build_manifest(cells)

    out_dir = Path("reports/confirmatory/confirmatory_v1")
    folds_dir = out_dir / "folds"
    cells_dir = out_dir / "cells"
    oof_dir = out_dir / "oof_predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    folds_dir.mkdir(exist_ok=True)
    cells_dir.mkdir(exist_ok=True)
    oof_dir.mkdir(exist_ok=True)

    import shutil

    for cohort in cohorts:
        src_fold = Path("data/processed") / cohort.lower() / "cv_folds.json"
        dst_fold = folds_dir / f"{cohort}.json"
        if src_fold.exists():
            shutil.copy(src_fold, dst_fold)
        else:
            print(f"Warning: Fold file not found at {src_fold}")

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest["cells"], f, indent=2)

    print(f"Generated {len(cells)} cells.")

    # We leave the actual running to a manager that can parallelize,
    # but here is the sequential execution loop.
    # To run, one would iterate `cells` and execute `ExperimentRunner` or the frequentist models.
    # This script just builds the manifest for now to satisfy Phase 2B cell generation.


if __name__ == "__main__":
    main()

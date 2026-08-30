import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.config_resolver import resolve_bayesian_config
from src.experiments.manifest import ExperimentCell, build_manifest
from src.experiments.runner import ExperimentRunner
from src.training.checkpoints import CheckpointManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    cohorts = ["GBSG2", "WHAS500", "METABRIC"]
    bayesian_models = ["Normal", "Regularised Horseshoe", "Continuous Spike-and-Slab"]

    cells = []

    # 1. Generate Manifest
    for cohort in cohorts:
        for fold in range(25):
            for b_model in bayesian_models:
                cfg = resolve_bayesian_config(cohort, b_model)
                cell = ExperimentCell(
                    experiment_id="phase1_exploratory",
                    dataset=cohort,
                    prior=b_model,
                    fold=fold,
                    seed=42 + fold,
                    method="mcmc",
                    draws=10,  # Fast execution for pipeline verification
                    tune=10,  # Fast execution
                    chains=1,
                    target_accept=cfg["target_accept"],
                    n_intervals=cfg["n_intervals"],
                    coefficient_prior=cfg["coefficient_prior"],
                    beta_prior_mean=cfg["beta_prior_mean"],
                    beta_prior_sd=cfg["beta_prior_sd"],
                    baseline_hazard_prior=cfg["baseline_hazard_prior"],
                    baseline_hazard_params=cfg["baseline_hazard_params"],
                )
                cells.append(cell)

    manifest = build_manifest(cells)

    out_dir = Path("outputs/experiments/phase1_exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest["cells"], f, indent=2)

    logger.info(f"Generated Phase 1 manifest with {len(cells)} cells.")

    # 2. Execute Cells
    checkpoints = CheckpointManager(str(out_dir))
    runner = ExperimentRunner(checkpoints, data_root="data")

    pass_count = 0
    fail_count = 0

    for i, cell in enumerate(cells):
        logger.info(
            f"Executing cell {i + 1}/{len(cells)}: {cell.dataset} - {cell.prior} - fold {cell.fold}"
        )
        try:
            result = runner.run(cell)
            if result.get("status") == "PASS":
                pass_count += 1
            else:
                fail_count += 1
        except (ValueError, RuntimeError, OSError) as e:
            logger.error(f"Cell failed: {e}")
            fail_count += 1

    logger.info(f"Phase 1 Execution Complete. PASS: {pass_count}, FAIL: {fail_count}")


if __name__ == "__main__":
    main()

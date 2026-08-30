import shutil
from pathlib import Path

from src.experiments.manifest import ExperimentCell
from src.experiments.runner import ExperimentRunner
from src.training.checkpoints import CheckpointManager


def main():
    root = Path("outputs/metabric_horseshoe_gamma_convergence")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    cell = ExperimentCell(
        experiment_id="metabric_horseshoe_gamma_convergence",
        dataset="METABRIC",
        prior="Regularised Horseshoe",
        fold=0,
        seed=42,
        method="mcmc",
        draws=500,
        tune=1000,
        chains=4,
        target_accept=0.995,
        n_intervals=6,
        coefficient_prior="regularized_horseshoe",
        beta_prior_mean=0.0,
        beta_prior_sd=2.5,
        baseline_hazard_prior="gamma",
        baseline_hazard_params={
            "alpha": 1.0,
        },
    )

    runner = ExperimentRunner(
        CheckpointManager(str(root)),
        data_root="data",
    )

    result = runner.run(cell)

    print("\n==================================")
    print("METABRIC HORSESHOE GAMMA PILOT (0.995)")
    print("==================================")
    print("STATUS:", result["status"])
    print("CELL:", result["cell_id"])
    print("\nMETRICS:")
    print(result["metrics"])
    print("\nDIAGNOSTICS:")
    print(result["diagnostics"])

if __name__ == "__main__":
    main()

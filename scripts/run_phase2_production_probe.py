import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.manifest import ExperimentCell
from src.experiments.runner import ExperimentRunner
from src.training.checkpoints import CheckpointManager


def main():
    cell = ExperimentCell(
        experiment_id="phase2_validation_v1",
        dataset="GBSG2",
        prior="Normal",
        fold=0,
        seed=42,
        method="mcmc",
        draws=1000,
        tune=1000,
        chains=4,
    )

    manager = CheckpointManager("outputs/checkpoints")
    runner = ExperimentRunner(manager)

    print("=== PRODUCTION MCMC PROBE ===")
    print("Cell ID:", cell.cell_id)
    print("Draws:", cell.draws)
    print("Tune:", cell.tune)
    print("Chains:", cell.chains)

    result = runner.run(cell)

    diagnostics = result["diagnostics"]

    print("\n=== DIAGNOSTICS ===")
    print("R-hat:", diagnostics["rhat_max"])
    print("Bulk ESS:", diagnostics["ess_bulk_min"])
    print("Tail ESS:", diagnostics["ess_tail_min"])
    print("Divergences:", diagnostics["divergences"])
    print("BFMI:", diagnostics["bfmi_min"])
    print("Tree depth:", diagnostics["tree_depth_max"])
    print("Gates:", diagnostics["gates"])
    print("Status:", result["status"])

    checkpoint = manager.load_checkpoint(cell)

    print("\nCheckpoint status:", checkpoint["status"])


if __name__ == "__main__":
    main()

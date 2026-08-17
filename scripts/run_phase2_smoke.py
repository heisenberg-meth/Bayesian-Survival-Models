from pathlib import Path

from src.experiments.manifest import ExperimentCell
from src.experiments.runner import ExperimentRunner
from src.training.checkpoints import CheckpointManager


def main():
    cell = ExperimentCell(
        experiment_id="phase2_smoke_v1",
        dataset="GBSG2",
        prior="Normal",
        fold=0,
        seed=42,
        method="mcmc",
        draws=200,
        tune=200,
        chains=2,
    )

    checkpoint_root = Path("outputs/checkpoints")

    manager = CheckpointManager(str(checkpoint_root))
    runner = ExperimentRunner(manager)

    print("=== PHASE 2.10 REAL CELL ===")
    print("Cell ID:", cell.cell_id)
    print("Dataset:", cell.dataset)
    print("Prior:", cell.prior)
    print("Fold:", cell.fold)
    print("Seed:", cell.seed)
    print("Draws:", cell.draws)
    print("Tune:", cell.tune)
    print("Chains:", cell.chains)

    result = runner.run(cell)

    print("\n=== RESULT ===")
    print("Status:", result["status"])
    print("Cell ID:", result["cell_id"])
    print("Diagnostics:", result["diagnostics"])

    checkpoint = manager.load_checkpoint(cell)

    print("\n=== CHECKPOINT ===")
    print("Status:", checkpoint["status"])
    print("Artifacts:", checkpoint["artifacts"])

    assert result["cell_id"] == cell.cell_id
    assert checkpoint["cell_id"] == cell.cell_id
    assert checkpoint["status"] == "complete"
    assert "result" in checkpoint["artifacts"]

    print("\nPHASE 2.10 SMOKE CHECK: PASS")


if __name__ == "__main__":
    main()

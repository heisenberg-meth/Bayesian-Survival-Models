import json
import os
import tempfile
from datetime import datetime, timezone

from src.experiments.manifest import ExperimentCell


class CheckpointValidationError(Exception):
    pass


class CheckpointExistsError(Exception):
    pass


class CheckpointNotFoundError(Exception):
    pass


class CheckpointManager:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def _get_cell_dir(self, cell: ExperimentCell) -> str:
        return os.path.join(
            self.base_dir,
            cell.experiment_id,
            cell.cell_id,
        )

    def _get_checkpoint_path(self, cell: ExperimentCell) -> str:
        return os.path.join(self._get_cell_dir(cell), "checkpoint.json")

    def _write_atomic(self, path: str, data: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.rename(temp_path, path)

    def create_checkpoint(self, cell: ExperimentCell, status: str = "pending"):
        if status not in (
            "pending",
            "running",
            "failed",
            "failed_diagnostics",
            "complete",
        ):
            raise ValueError(f"Invalid status: {status}")

        if self.is_complete(cell):
            raise CheckpointExistsError("Completed checkpoint cannot be overwritten")

        now = datetime.now(timezone.utc).isoformat()

        metadata = {
            "schema_version": "1.0",
            "cell_id": cell.cell_id,
            "experiment_id": cell.experiment_id,
            "dataset": cell.dataset,
            "prior": cell.prior,
            "fold": cell.fold,
            "seed": cell.seed,
            "status": status,
            "created_at": now,
            "updated_at": now,
            "completed_at": now if status == "complete" else None,
            "config": cell.to_dict(),
            "artifacts": {},
        }

        self._write_atomic(self._get_checkpoint_path(cell), metadata)

    def save_checkpoint(
        self, cell: ExperimentCell, status: str, artifacts: dict[str, str] | None = None
    ):
        if status not in (
            "pending",
            "running",
            "failed",
            "failed_diagnostics",
            "complete",
        ):
            raise ValueError(f"Invalid status: {status}")

        if not self.checkpoint_exists(cell):
            raise CheckpointNotFoundError("Checkpoint does not exist")

        metadata = self.load_checkpoint(cell)

        if metadata["status"] == "complete":
            raise CheckpointExistsError("Completed checkpoint cannot be overwritten")

        now = datetime.now(timezone.utc).isoformat()
        metadata["status"] = status
        metadata["updated_at"] = now
        if status == "complete":
            metadata["completed_at"] = now

        if artifacts:
            metadata["artifacts"].update(artifacts)

        self._write_atomic(self._get_checkpoint_path(cell), metadata)

    def load_checkpoint(self, cell: ExperimentCell) -> dict:
        path = self._get_checkpoint_path(cell)
        if not os.path.exists(path):
            raise CheckpointNotFoundError("Checkpoint does not exist")

        with open(path) as f:
            try:
                metadata = json.load(f)
            except json.JSONDecodeError:
                raise ValueError("Corrupt JSON checkpoint")

        self.validate_checkpoint(cell, metadata)
        return metadata

    def checkpoint_exists(self, cell: ExperimentCell) -> bool:
        return os.path.exists(self._get_checkpoint_path(cell))

    def is_complete(self, cell: ExperimentCell) -> bool:
        try:
            metadata = self.load_checkpoint(cell)
            return metadata["status"] == "complete"
        except CheckpointNotFoundError:
            return False
        except ValueError:
            return False
        except CheckpointValidationError:
            return False

    def validate_checkpoint(self, expected: ExperimentCell, checkpoint: dict):
        fields_to_check = [
            "cell_id",
            "experiment_id",
            "dataset",
            "prior",
            "fold",
            "seed",
        ]
        for field in fields_to_check:
            if checkpoint.get(field) != getattr(expected, field):
                raise CheckpointValidationError(
                    f"Mismatch in {field}: expected {getattr(expected, field)}, "
                    f"got {checkpoint.get(field)}"
                )

        # Validate artifacts exist if registered
        artifacts = checkpoint.get("artifacts", {})
        cell_dir = self._get_cell_dir(expected)
        for artifact_path in artifacts.values():
            if not os.path.exists(os.path.join(cell_dir, artifact_path)):
                raise CheckpointValidationError(f"Artifact missing: {artifact_path}")

    def mark_complete(self, cell: ExperimentCell):
        self.save_checkpoint(cell, status="complete")

    def save_artifact(self, cell: ExperimentCell, artifact_name: str, data: dict):
        """Saves a JSON artifact (e.g. metrics.json or diagnostics.json) to the cell directory."""
        if self.is_complete(cell):
            raise CheckpointExistsError("Completed checkpoint cannot be overwritten")

        artifact_file = f"{artifact_name}.json"
        path = os.path.join(self._get_cell_dir(cell), artifact_file)
        self._write_atomic(path, data)

        # Update the checkpoint to register this artifact
        self.save_checkpoint(
            cell,
            status=self.load_checkpoint(cell)["status"],
            artifacts={artifact_name: artifact_file},
        )

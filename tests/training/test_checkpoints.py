import os
import tempfile
import unittest

from src.experiments.manifest import ExperimentCell
from src.training.checkpoints import (
    CheckpointExistsError,
    CheckpointManager,
    CheckpointNotFoundError,
    CheckpointValidationError,
)


class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.manager = CheckpointManager(base_dir=self.temp_dir.name)
        self.cell = ExperimentCell(
            experiment_id="test_exp",
            dataset="GBSG2",
            prior="Normal",
            fold=0,
            seed=42,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_create_checkpoint(self):
        self.manager.create_checkpoint(self.cell)
        path = self.manager._get_checkpoint_path(self.cell)
        self.assertTrue(os.path.exists(path))

        data = self.manager.load_checkpoint(self.cell)
        self.assertEqual(data["status"], "pending")
        self.assertEqual(data["cell_id"], self.cell.cell_id)

    def test_02_save_running_checkpoint(self):
        self.manager.create_checkpoint(self.cell)
        self.manager.save_checkpoint(self.cell, status="running")

        data = self.manager.load_checkpoint(self.cell)
        self.assertEqual(data["status"], "running")

    def test_03_complete_checkpoint(self):
        self.manager.create_checkpoint(self.cell)
        self.manager.mark_complete(self.cell)

        self.assertTrue(self.manager.is_complete(self.cell))
        data = self.manager.load_checkpoint(self.cell)
        self.assertEqual(data["status"], "complete")

    def test_04_wrong_cell_rejected(self):
        self.manager.create_checkpoint(self.cell)

        wrong_cell = ExperimentCell(
            experiment_id="test_exp",
            dataset="GBSG2",
            prior="Regularised Horseshoe",
            fold=0,
            seed=42,
        )

        # Manually move the file to the wrong cell's location to simulate a mismatch
        wrong_path = self.manager._get_checkpoint_path(wrong_cell)
        os.makedirs(os.path.dirname(wrong_path), exist_ok=True)
        os.rename(self.manager._get_checkpoint_path(self.cell), wrong_path)

        with self.assertRaises(CheckpointValidationError):
            self.manager.load_checkpoint(wrong_cell)

    def test_05_wrong_seed_rejected(self):
        self.manager.create_checkpoint(self.cell)

        wrong_seed_cell = ExperimentCell(
            experiment_id="test_exp",
            dataset="GBSG2",
            prior="Normal",
            fold=0,
            seed=43,
        )
        # Manually move the file to the wrong cell's location to simulate a mismatch
        wrong_path = self.manager._get_checkpoint_path(wrong_seed_cell)
        os.makedirs(os.path.dirname(wrong_path), exist_ok=True)
        os.rename(self.manager._get_checkpoint_path(self.cell), wrong_path)

        with self.assertRaises(CheckpointValidationError):
            self.manager.load_checkpoint(wrong_seed_cell)

    def test_06_completed_checkpoint_cannot_be_overwritten(self):
        self.manager.create_checkpoint(self.cell, status="complete")

        with self.assertRaises(CheckpointExistsError):
            self.manager.save_checkpoint(self.cell, status="failed")

        with self.assertRaises(CheckpointExistsError):
            self.manager.create_checkpoint(self.cell, status="pending")

    def test_07_missing_checkpoint(self):
        with self.assertRaises(CheckpointNotFoundError):
            self.manager.load_checkpoint(self.cell)

    def test_08_corrupt_json(self):
        path = self.manager._get_checkpoint_path(self.cell)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("{ invalid json")

        with self.assertRaises(ValueError):
            self.manager.load_checkpoint(self.cell)

    def test_09_artifact_existence(self):
        self.manager.create_checkpoint(self.cell)
        self.manager.save_artifact(self.cell, "metrics", {"c_index": 0.8})

        # Load it fine
        self.manager.load_checkpoint(self.cell)

        # Corrupt the artifact
        artifact_path = os.path.join(
            self.manager._get_cell_dir(self.cell), "metrics.json"
        )
        os.remove(artifact_path)

        with self.assertRaises(CheckpointValidationError):
            self.manager.load_checkpoint(self.cell)

    def test_10_deterministic_checkpoint_identity(self):
        cell1 = ExperimentCell(
            experiment_id="test", dataset="A", prior="B", fold=1, seed=42
        )
        cell2 = ExperimentCell(
            experiment_id="test", dataset="A", prior="B", fold=1, seed=42
        )

        self.assertEqual(cell1.cell_id, cell2.cell_id)
        self.assertEqual(
            self.manager._get_checkpoint_path(cell1),
            self.manager._get_checkpoint_path(cell2),
        )

    def test_11_different_seed_gets_different_checkpoint_path(self):
        cell1 = ExperimentCell(
            experiment_id="test",
            dataset="GBSG2",
            prior="Normal",
            fold=0,
            seed=42,
        )

        cell2 = ExperimentCell(
            experiment_id="test",
            dataset="GBSG2",
            prior="Normal",
            fold=0,
            seed=43,
        )

        self.assertNotEqual(cell1.cell_id, cell2.cell_id)
        self.assertNotEqual(
            self.manager._get_checkpoint_path(cell1),
            self.manager._get_checkpoint_path(cell2),
        )

    def test_12_different_sampling_config_gets_different_checkpoint_path(self):
        cell1 = ExperimentCell(
            experiment_id="test",
            dataset="GBSG2",
            prior="Normal",
            fold=0,
            seed=42,
            draws=1000,
            tune=1000,
            chains=4,
        )

        cell2 = ExperimentCell(
            experiment_id="test",
            dataset="GBSG2",
            prior="Normal",
            fold=0,
            seed=42,
            draws=2000,
            tune=1000,
            chains=4,
        )

        self.assertNotEqual(cell1.cell_id, cell2.cell_id)
        self.assertNotEqual(
            self.manager._get_checkpoint_path(cell1),
            self.manager._get_checkpoint_path(cell2),
        )


if __name__ == "__main__":
    unittest.main()

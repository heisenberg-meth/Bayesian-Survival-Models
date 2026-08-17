import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.loader import DatasetLoader
from src.data.preprocessing import SurvivalDataPipeline
from src.models.bayesian.diagnostics import MCMCDiagnostics
from src.models.bayesian.model import BayesianCoxModel


class TestBayesianRealFit(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_real_gbsg2_fit(self):
        # 1. Load real GBSG2 dataset
        raw_df = DatasetLoader.load_raw("data", "gbsg2")

        # 2. Run existing real preprocessing, redirecting artifacts to a temp dir
        pipeline = SurvivalDataPipeline(
            dataset_name="gbsg2",
            time_col="time",
            event_col="cens",
            categorical_cols=["horTh", "menostat"],
            ordinal_cols=None,
            random_state=42,
        )
        pipeline.run(raw_df, output_dir=self.temp_dir.name)

        # Extract training data created by the pipeline
        train_path = os.path.join(self.temp_dir.name, "processed", "gbsg2", "train.csv")
        train_df = pd.read_csv(train_path)

        X_train = train_df.drop(columns=["time", "event"])
        y_train_time = train_df["time"].values
        y_train_event = train_df["event"].values

        # 3. Instantiate BayesianCoxModel with tiny MCMC settings
        model = BayesianCoxModel(
            inference_method="mcmc",
            draws=10,
            tune=10,
            chains=2,
            random_state=42,
            n_intervals=2,
        )

        # 4. Fit the model
        model.fit(X_train, y_train_time, y_train_event)

        # 5. Confirm idata exists and is an ArviZ InferenceData
        self.assertIsNotNone(model.idata)
        self.assertTrue(hasattr(model.idata, "posterior"))
        self.assertTrue(hasattr(model.idata, "sample_stats"))

        # 6. Compute diagnostics
        diags = MCMCDiagnostics.compute_all(model.idata)

        # Confirm diagnostics are computed and not just hardcoded placeholders
        self.assertIn("rhat_max", diags)
        self.assertIn("ess_bulk_min", diags)
        self.assertIn("divergences", diags)

        self.assertTrue(np.isfinite(diags["rhat_max"]))
        self.assertTrue(np.isfinite(diags["ess_bulk_min"]))
        self.assertTrue(np.isfinite(diags["bfmi_min"]))
        self.assertIsInstance(diags["divergences"], int)
        self.assertIsInstance(diags["tree_depth_max"], int)

        # Check values are computed from the real posterior (not just 1.01 or 1500)
        # Note: with draws=10, these values will be wild, which proves they are real.
        self.assertNotEqual(diags["rhat_max"], 1.01)
        self.assertNotEqual(diags["ess_bulk_min"], 1500.0)


if __name__ == "__main__":
    unittest.main()

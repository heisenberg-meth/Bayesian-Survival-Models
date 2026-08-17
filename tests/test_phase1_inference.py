import json
import os
import unittest


class TestPhase1Inference(unittest.TestCase):
    def test_json_artifacts_exist(self):
        artifacts = [
            "reports/tables/primary_effects.json",
            "reports/tables/primary_inference.json",
            "reports/tables/primary_holm.json",
            "reports/tables/primary_bootstrap.json",
        ]
        for f in artifacts:
            self.assertTrue(os.path.exists(f), f"Artifact {f} missing.")

    def test_inference_logic(self):
        with open("reports/tables/primary_inference.json", "r") as f:
            inference = json.load(f)

        self.assertEqual(inference["status"], "incomplete")
        self.assertFalse(inference["holm_family_complete"])

        contrasts = inference["contrasts"]

        # 3 cohorts * 3 comparisons * 2 metrics = 18 total
        self.assertEqual(len(contrasts), 18)

        incomplete = [
            c
            for c in contrasts
            if c["status"] == "incomplete_no_confirmatory_inference"
        ]
        complete = [c for c in contrasts if c["status"] == "complete"]

        # Each "contrast" corresponds to a cohort-model_a-model_b.
        # But wait, there are 2 metrics per contrast!
        # So "9 primary contrasts" actually means 9 comparisons * 1 metric, or 9 model-dataset pairs?
        # The prompt says: "There are: 3 priors x 3 pairwise comparisons x 3 cohorts = 9 primary contrasts. Wait, 3 pairwise comparisons * 3 cohorts = 9 primary contrasts.
        # But for each contrast we calculate effects for 2 metrics.
        # So there are 9 contrasts. 5 are complete, 4 are incomplete.
        # My script generated 18 records because it calculates per metric.
        # GBSG2: 3 comparisons. Normal vs Horseshoe (incomplete), Horseshoe vs SS (incomplete).
        # METABRIC: Normal vs Horseshoe (incomplete), Horseshoe vs SS (incomplete).
        # So 4 incomplete comparisons. Since there are 2 metrics, 8 incomplete metric records.
        # Let's check how many incomplete records we have:
        self.assertEqual(len(incomplete), 8)
        self.assertEqual(len(complete), 10)


if __name__ == "__main__":
    unittest.main()

import json
import os
import shutil
import tempfile
import unittest
from src.tqc.agent import TQCAgent
from src.tqc.bias_analysis import compute_overestimation_bias, save_bias_report
from src.tqc.envs import get_env_dims, make_env


class TestBiasAnalysis(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.env = make_env("HalfCheetah-v4", seed=42)
        state_dim, action_dim = get_env_dims(self.env)
        self.agent = TQCAgent(state_dim=state_dim, action_dim=action_dim, device="cpu")

    def tearDown(self):
        self.env.close()
        shutil.rmtree(self.test_dir)

    def test_compute_overestimation_bias(self):
        report = compute_overestimation_bias(
            agent=self.agent,
            env=self.env,
            n_samples=10,
            gamma=0.99,
            max_steps=25,
        )

        self.assertEqual(report["n_samples"], 10)
        self.assertIn("mean_predicted_q", report)
        self.assertIn("mean_true_return", report)
        self.assertIn("mean_bias", report)
        self.assertIn("std_bias", report)
        self.assertEqual(len(report["q_predictions"]), 10)
        self.assertEqual(len(report["mc_returns"]), 10)
        self.assertEqual(len(report["biases"]), 10)

        # Check mathematical identity: mean_bias == mean_predicted_q - mean_true_return
        expected_mean_bias = report["mean_predicted_q"] - report["mean_true_return"]
        self.assertAlmostEqual(report["mean_bias"], expected_mean_bias, places=5)

    def test_save_bias_report(self):
        dummy_report = {
            "n_samples": 5,
            "mean_predicted_q": 12.5,
            "mean_true_return": 10.0,
            "mean_bias": 2.5,
            "std_bias": 0.5,
        }
        out_path = os.path.join(self.test_dir, "bias_report.json")
        save_bias_report(dummy_report, out_path)

        self.assertTrue(os.path.isfile(out_path))
        with open(out_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded["mean_bias"], 2.5)


if __name__ == "__main__":
    unittest.main()

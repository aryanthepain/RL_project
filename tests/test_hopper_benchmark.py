import glob
import os
import shutil
import tempfile
import unittest

import numpy as np
import torch
import yaml

from scripts.run_hopper_benchmark import (
    DEFAULT_DROP_TOP,
    ENV_ID,
    generate_hopper_reproduction_plots,
    run_hopper_bias_analysis,
    run_hopper_training,
)
from src.tqc.envs import get_env_dims, get_env_metadata, make_env


class TestHopperBenchmark(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.test_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_hopper_env_specs(self):
        """Verify Hopper-v4 environment instantiation, observation, and action bounds."""
        env = make_env(ENV_ID, seed=42)
        state_dim, action_dim = get_env_dims(env)
        obs, _ = env.reset(seed=42)
        env.close()

        self.assertEqual(state_dim, 11)
        self.assertEqual(action_dim, 3)
        self.assertEqual(obs.shape, (11,))

    def test_hopper_hyperparameter_metadata(self):
        """Verify Hopper-v4 metadata conforms to Kuznetsov et al. (ICML 2020)."""
        meta = get_env_metadata(ENV_ID)
        self.assertEqual(meta["drop_top"], 5)
        self.assertEqual(meta["total_timesteps"], 1_000_000)
        self.assertEqual(DEFAULT_DROP_TOP, 5)

    def test_hopper_presets_in_matrix(self):
        """Verify Hopper presets exist in configs/benchmark_matrix.yaml."""
        matrix_path = os.path.join(os.path.dirname(__file__), "..", "configs", "benchmark_matrix.yaml")
        with open(matrix_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        presets = cfg.get("presets", {})
        self.assertIn("hopper_pilot", presets)
        self.assertIn("hopper_phase9", presets)
        self.assertEqual(presets["hopper_pilot"]["env_id"], "Hopper-v4")
        self.assertEqual(presets["hopper_phase9"]["seeds"], [42, 43, 44, 45, 46])

    def test_hopper_training_fast_smoke(self):
        """Execute fast Hopper training smoke run and verify artifacts generation."""
        runs_dir = os.path.join(self.test_dir, "runs")
        completed = run_hopper_training(
            seeds=[42],
            timesteps=30,
            drop_top=5,
            device="cpu",
            log_dir=runs_dir,
            fast_test=True,
        )

        self.assertIn(42, completed)
        run_path = completed[42]
        self.assertTrue(os.path.isdir(run_path))

        # Check required artifacts exist
        metrics_csv = os.path.join(run_path, "metrics.csv")
        metrics_jsonl = os.path.join(run_path, "metrics.jsonl")
        final_model = os.path.join(run_path, "final_model.pt")

        self.assertTrue(os.path.isfile(metrics_csv))
        self.assertTrue(os.path.isfile(metrics_jsonl))
        self.assertTrue(os.path.isfile(final_model))

        # Check bias analysis
        bias_res = run_hopper_bias_analysis(
            run_dirs=[run_path],
            results_dir=self.results_dir,
            n_samples=5,
            device="cpu",
        )
        self.assertIn("mean_predicted_q", bias_res)
        self.assertIn("mean_true_return", bias_res)
        self.assertIn("mean_bias", bias_res)

        # Check plot and table generation
        generate_hopper_reproduction_plots(
            log_dir=runs_dir,
            results_dir=self.results_dir,
            bias_report=bias_res,
        )

        self.assertTrue(
            os.path.isfile(os.path.join(self.results_dir, "hopper_reproduction_learning_curves.png"))
        )
        self.assertTrue(
            os.path.isfile(os.path.join(self.results_dir, "hopper_bias_diagnostics.png"))
        )
        self.assertTrue(
            os.path.isfile(os.path.join(self.results_dir, "hopper_results_table.md"))
        )


if __name__ == "__main__":
    unittest.main()

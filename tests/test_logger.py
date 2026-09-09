import csv
import json
import os
import shutil
import tempfile
import unittest
from src.tqc.agent import TQCAgent
from src.tqc.logger import MetricsLogger, create_agent


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_metrics_logger_files_and_writes(self):
        logger = MetricsLogger(log_dir=self.test_dir, exp_name="test_run")
        self.assertTrue(os.path.isdir(logger.exp_dir))

        metrics1 = {
            "eval_return_mean": 100.5,
            "eval_return_std": 5.2,
            "critic_loss": 0.42,
            "actor_loss": -12.3,
            "alpha": 0.2,
        }
        logger.log_metrics(step=1000, metrics=metrics1, print_to_console=False)

        metrics2 = {
            "eval_return_mean": 150.0,
            "eval_return_std": 4.1,
            "critic_loss": 0.35,
            "actor_loss": -15.1,
            "alpha": 0.18,
        }
        logger.log_metrics(step=2000, metrics=metrics2, print_to_console=False)
        logger.close()

        # Verify CSV
        self.assertTrue(os.path.isfile(logger.csv_path))
        with open(logger.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertEqual(float(rows[0]["step"]), 1000)
            self.assertEqual(float(rows[0]["eval_return_mean"]), 100.5)
            self.assertEqual(float(rows[1]["step"]), 2000)
            self.assertEqual(float(rows[1]["eval_return_mean"]), 150.0)

        # Verify JSONL
        self.assertTrue(os.path.isfile(logger.jsonl_path))
        with open(logger.jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["step"], 1000)
            self.assertEqual(lines[1]["step"], 2000)

    def test_create_agent_tqc_and_sac(self):
        # TQC default
        agent_tqc = create_agent("tqc", state_dim=17, action_dim=6, env_id="HalfCheetah-v4")
        self.assertIsInstance(agent_tqc, TQCAgent)
        self.assertEqual(agent_tqc.n_critics, 5)
        self.assertEqual(agent_tqc.n_quantiles, 25)
        self.assertEqual(agent_tqc.drop_top, 5)

        # Humanoid TQC
        agent_hum = create_agent("tqc", state_dim=376, action_dim=17, env_id="Humanoid-v4")
        self.assertEqual(agent_hum.drop_top, 2)

        # SAC baseline
        agent_sac = create_agent("sac", state_dim=17, action_dim=6)
        self.assertIsInstance(agent_sac, TQCAgent)
        self.assertEqual(agent_sac.n_critics, 2)
        self.assertEqual(agent_sac.n_quantiles, 1)
        self.assertEqual(agent_sac.drop_top, 0)

        with self.assertRaises(ValueError):
            create_agent("unknown_algo", state_dim=10, action_dim=2)


if __name__ == "__main__":
    unittest.main()

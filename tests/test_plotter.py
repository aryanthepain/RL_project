import csv
import os
import shutil
import tempfile
import unittest
from src.tqc.plotter import load_run_data, plot_bias_comparison, plot_learning_curves


class TestPlotter(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_mock_run(self, run_dir, steps, returns):
        os.makedirs(run_dir, exist_ok=True)
        csv_path = os.path.join(run_dir, "metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "eval_return_mean", "eval_return_std"])
            for s, r in zip(steps, returns):
                writer.writerow([s, r, 5.0])

    def test_load_run_data(self):
        run_dir = os.path.join(self.test_dir, "run_0")
        self._create_mock_run(run_dir, [10, 20, 30], [100.0, 150.0, 200.0])

        data = load_run_data(run_dir)
        self.assertIsNotNone(data)
        self.assertEqual(len(data["steps"]), 3)
        self.assertEqual(len(data["returns"]), 3)
        self.assertEqual(data["returns"][-1], 200.0)

    def test_plot_learning_curves(self):
        tqc_dir1 = os.path.join(self.test_dir, "tqc_s0")
        tqc_dir2 = os.path.join(self.test_dir, "tqc_s1")
        sac_dir1 = os.path.join(self.test_dir, "sac_s0")

        self._create_mock_run(tqc_dir1, [100, 200, 300], [50, 100, 150])
        self._create_mock_run(tqc_dir2, [100, 200, 300], [60, 110, 160])
        self._create_mock_run(sac_dir1, [100, 200, 300], [40, 80, 110])

        run_dirs = {
            "TQC": [tqc_dir1, tqc_dir2],
            "SAC": [sac_dir1],
        }
        out_png = os.path.join(self.test_dir, "learning_curves.png")
        plot_learning_curves(run_dirs, out_png, env_name="HalfCheetah-v4")

        self.assertTrue(os.path.isfile(out_png))
        self.assertGreater(os.path.getsize(out_png), 1000)

    def test_plot_bias_comparison(self):
        mock_bias = {
            "TQC": {
                "mean_bias": 1.2,
                "std_bias": 0.4,
                "q_predictions": [10.0, 12.0, 15.0],
                "mc_returns": [9.0, 11.0, 13.5],
            },
            "SAC": {
                "mean_bias": 8.5,
                "std_bias": 1.8,
                "q_predictions": [20.0, 25.0, 28.0],
                "mc_returns": [10.0, 15.0, 20.0],
            },
        }
        out_png = os.path.join(self.test_dir, "bias_comparison.png")
        plot_bias_comparison(mock_bias, out_png, env_name="HalfCheetah-v4")

        self.assertTrue(os.path.isfile(out_png))
        self.assertGreater(os.path.getsize(out_png), 1000)


if __name__ == "__main__":
    unittest.main()

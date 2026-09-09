import os
import shutil
import tempfile
import unittest
from src.tqc.benchmark import run_benchmark


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_run_benchmark_fast_test(self):
        # Run 1 env, 1 algo, 1 seed in fast_test mode
        results = run_benchmark(
            env_ids=["HalfCheetah-v4"],
            algos=["tqc"],
            seeds=[0],
            fast_test=True,
            device="cpu",
            log_dir=self.test_dir,
        )

        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res["env_id"], "HalfCheetah-v4")
        self.assertEqual(res["algo"], "tqc")
        self.assertEqual(res["seed"], 0)
        self.assertEqual(res["timesteps"], 30)

        # Check directory was created and files exist
        exp_dir = os.path.join(self.test_dir, res["exp_name"])
        self.assertTrue(os.path.isdir(exp_dir))
        self.assertTrue(os.path.isfile(os.path.join(exp_dir, "metrics.csv")))
        self.assertTrue(os.path.isfile(os.path.join(exp_dir, "final_model.pt")))


if __name__ == "__main__":
    unittest.main()

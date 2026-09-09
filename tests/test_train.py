import os
import shutil
import tempfile
import unittest
from src.tqc.agent import TQCAgent
from src.tqc.envs import get_env_dims, make_env
from src.tqc.evaluate import evaluate_policy
from src.tqc.train import train_tqc


class TestTrainPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_evaluate_policy(self):
        env = make_env("HalfCheetah-v4", seed=42)
        state_dim, action_dim = get_env_dims(env)
        agent = TQCAgent(state_dim=state_dim, action_dim=action_dim, device="cpu")

        mean_ret, std_ret = evaluate_policy(agent, env, n_episodes=2, deterministic=True)
        self.assertIsInstance(mean_ret, float)
        self.assertIsInstance(std_ret, float)
        env.close()

    def test_train_tqc_end_to_end_integration(self):
        # Run miniature end-to-end training loop
        exp_name = "test_mini_train"
        agent = train_tqc(
            env_id="HalfCheetah-v4",
            seed=42,
            total_timesteps=30,
            eval_freq=15,
            eval_episodes=1,
            warmup_steps=10,
            batch_size=8,
            buffer_capacity=100,
            drop_top=5,
            algo="tqc",
            device="cpu",
            log_dir=self.test_dir,
            exp_name=exp_name,
            checkpoint_freq=15,
        )

        self.assertIsInstance(agent, TQCAgent)

        exp_dir = os.path.join(self.test_dir, exp_name)
        self.assertTrue(os.path.isdir(exp_dir))

        # Checkpoint files
        self.assertTrue(os.path.isfile(os.path.join(exp_dir, "best_model.pt")))
        self.assertTrue(os.path.isfile(os.path.join(exp_dir, "final_model.pt")))
        self.assertTrue(os.path.isfile(os.path.join(exp_dir, "checkpoint_15.pt")))

        # Metric logs
        self.assertTrue(os.path.isfile(os.path.join(exp_dir, "metrics.csv")))
        self.assertTrue(os.path.isfile(os.path.join(exp_dir, "metrics.jsonl")))

    def test_train_sac_mode_end_to_end(self):
        # Run miniature training in SAC baseline mode
        exp_name = "test_mini_sac"
        agent = train_tqc(
            env_id="HalfCheetah-v4",
            seed=42,
            total_timesteps=20,
            eval_freq=10,
            eval_episodes=1,
            warmup_steps=8,
            batch_size=4,
            buffer_capacity=50,
            algo="sac",
            device="cpu",
            log_dir=self.test_dir,
            exp_name=exp_name,
            checkpoint_freq=0,
        )

        self.assertIsInstance(agent, TQCAgent)
        self.assertEqual(agent.n_critics, 2)
        self.assertEqual(agent.n_quantiles, 1)
        self.assertEqual(agent.drop_top, 0)


if __name__ == "__main__":
    unittest.main()

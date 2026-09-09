import os
import shutil
import tempfile
import unittest
from src.tqc.agent import TQCAgent
from src.tqc.envs import get_env_dims, make_env
from src.tqc.visualize import record_agent_video


class TestVisualize(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        env = make_env("HalfCheetah-v4", seed=42)
        state_dim, action_dim = get_env_dims(env)
        env.close()
        self.agent = TQCAgent(state_dim=state_dim, action_dim=action_dim, device="cpu")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_record_agent_video_gif(self):
        out_gif = os.path.join(self.test_dir, "test_rollout.gif")
        saved_path = record_agent_video(
            agent=self.agent,
            env_id="HalfCheetah-v4",
            output_path=out_gif,
            max_steps=5,  # fast test: 5 frames
            fps=10,
            seed=42,
        )

        self.assertEqual(saved_path, out_gif)
        self.assertTrue(os.path.isfile(out_gif))
        self.assertGreater(os.path.getsize(out_gif), 1000)


if __name__ == "__main__":
    unittest.main()

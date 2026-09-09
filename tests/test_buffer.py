import unittest
import numpy as np
import torch
from src.tqc.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.state_dim = 4
        self.action_dim = 2
        self.capacity = 100
        self.buffer = ReplayBuffer(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            capacity=self.capacity,
            device="cpu",
        )

    def test_buffer_init_and_empty_sample(self):
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(self.buffer.ptr, 0)
        with self.assertRaises(ValueError):
            self.buffer.sample(32)

    def test_buffer_add_and_len(self):
        for i in range(10):
            s = np.ones(self.state_dim) * i
            a = np.ones(self.action_dim) * i
            r = float(i)
            s_next = np.ones(self.state_dim) * (i + 1)
            done = False
            self.buffer.add(s, a, r, s_next, done)

        self.assertEqual(len(self.buffer), 10)
        self.assertEqual(self.buffer.ptr, 10)

    def test_buffer_circular_overwrite(self):
        small_capacity = 5
        buf = ReplayBuffer(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            capacity=small_capacity,
        )

        for i in range(12):
            s = np.array([i, i, i, i], dtype=np.float32)
            a = np.array([i, i], dtype=np.float32)
            r = float(i)
            s_next = s + 1
            done = i % 2 == 0
            buf.add(s, a, r, s_next, done)

        self.assertEqual(len(buf), small_capacity)
        # ptr should be 12 % 5 = 2
        self.assertEqual(buf.ptr, 2)
        # Check that elements at 0 and 1 correspond to items 10 and 11
        np.testing.assert_array_equal(buf.states[0], np.array([10, 10, 10, 10], dtype=np.float32))
        np.testing.assert_array_equal(buf.states[1], np.array([11, 11, 11, 11], dtype=np.float32))

    def test_buffer_sample_shapes_and_types(self):
        batch_size = 16
        for i in range(50):
            s = np.random.randn(self.state_dim)
            a = np.random.randn(self.action_dim)
            r = np.random.randn()
            s_next = np.random.randn(self.state_dim)
            done = (i % 10 == 0)
            self.buffer.add(s, a, r, s_next, done)

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        self.assertIsInstance(states, torch.Tensor)
        self.assertIsInstance(actions, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)

        self.assertEqual(states.shape, (batch_size, self.state_dim))
        self.assertEqual(actions.shape, (batch_size, self.action_dim))
        self.assertEqual(rewards.shape, (batch_size, 1))
        self.assertEqual(next_states.shape, (batch_size, self.state_dim))
        self.assertEqual(dones.shape, (batch_size, 1))

        self.assertEqual(states.dtype, torch.float32)
        self.assertEqual(actions.dtype, torch.float32)
        self.assertEqual(rewards.dtype, torch.float32)
        self.assertEqual(next_states.dtype, torch.float32)
        self.assertEqual(dones.dtype, torch.float32)

    def test_buffer_over_batch_size_error(self):
        for i in range(5):
            self.buffer.add(
                np.zeros(self.state_dim),
                np.zeros(self.action_dim),
                0.0,
                np.zeros(self.state_dim),
                False,
            )

        with self.assertRaises(ValueError):
            self.buffer.sample(10)


if __name__ == "__main__":
    unittest.main()

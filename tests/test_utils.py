import unittest
import numpy as np
import torch
from src.tqc.utils import seed_everything, get_device


class TestUtils(unittest.TestCase):
    def test_seed_everything_reproducibility(self):
        seed_everything(1234)
        t1 = torch.randn(10, 10)
        n1 = np.random.randn(10, 10)

        seed_everything(1234)
        t2 = torch.randn(10, 10)
        n2 = np.random.randn(10, 10)

        self.assertTrue(torch.equal(t1, t2))
        np.testing.assert_array_equal(n1, n2)

    def test_seed_everything_different_seeds(self):
        seed_everything(111)
        t1 = torch.randn(10, 10)

        seed_everything(222)
        t2 = torch.randn(10, 10)

        self.assertFalse(torch.equal(t1, t2))

    def test_get_device(self):
        dev_cpu = get_device("cpu")
        self.assertEqual(dev_cpu.type, "cpu")

        dev_default = get_device()
        self.assertIn(dev_default.type, ["cpu", "cuda"])


if __name__ == "__main__":
    unittest.main()

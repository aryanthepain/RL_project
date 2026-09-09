import unittest
import torch
from src.tqc.truncation import truncate_quantiles, quantile_huber_loss


class TestTruncation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_truncate_quantiles_sorting_and_dropping(self):
        batch_size = 4
        n_critics = 5
        n_quantiles = 25
        drop_top = 5

        # Create random quantiles
        raw = torch.randn(batch_size, n_critics, n_quantiles)
        truncated = truncate_quantiles(raw, drop_top=drop_top)

        expected_retained = n_critics * n_quantiles - drop_top  # 125 - 5 = 120
        self.assertEqual(truncated.shape, (batch_size, expected_retained))

        # Check sorted order
        for b in range(batch_size):
            diffs = truncated[b, 1:] - truncated[b, :-1]
            self.assertTrue((diffs >= 0).all(), "Truncated quantiles are not sorted in ascending order")

        # Verify that the dropped values were indeed the highest values
        flat_raw, _ = torch.sort(raw.reshape(batch_size, -1), dim=-1)
        expected = flat_raw[:, :expected_retained]
        self.assertTrue(torch.allclose(truncated, expected))

    def test_truncate_quantiles_zero_drop(self):
        raw = torch.randn(2, 3, 4)
        out = truncate_quantiles(raw, drop_top=0)
        self.assertEqual(out.shape, (2, 12))

    def test_quantile_huber_loss_symmetry_and_gradient(self):
        batch_size = 8
        n_critics = 5
        n_quantiles = 25
        n_target_quantiles = 120

        pred = torch.randn(batch_size, n_critics, n_quantiles, requires_grad=True)
        target = torch.randn(batch_size, n_target_quantiles)

        loss = quantile_huber_loss(pred, target)
        self.assertTrue(loss.item() > 0)
        self.assertFalse(torch.isnan(loss))

        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertFalse(torch.isnan(pred.grad).any())

    def test_quantile_huber_loss_exact_value(self):
        # 1 sample, 1 quantile at tau = 0.5 (median), predicted 0.0, target 2.0
        # error u = target - pred = 2.0. |u| > 1.0 -> huber = 2.0 - 0.5 = 1.5
        # u >= 0 so indicator = 0. weight = |0.5 - 0| / 1.0 = 0.5
        # expected loss = 0.5 * 1.5 = 0.75
        pred = torch.tensor([[0.0]])  # shape (1, 1)
        target = torch.tensor([[2.0]])  # shape (1, 1)
        loss = quantile_huber_loss(pred, target, kappa=1.0)
        self.assertAlmostEqual(loss.item(), 0.75, places=5)


if __name__ == "__main__":
    unittest.main()

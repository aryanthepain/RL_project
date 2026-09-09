import torch


def truncate_quantiles(target_quantiles: torch.Tensor, drop_top: int = 5) -> torch.Tensor:
    """
    Truncation operator for target quantiles.
    Pools quantiles across all target critics in the ensemble, sorts them in ascending order,
    and drops the top `drop_top` quantiles (outliers with the highest overestimation bias).

    Args:
        target_quantiles: Tensor of shape (batch_size, n_critics, n_quantiles)
        drop_top: Integer number of top quantiles to drop

    Returns:
        retained_quantiles: Tensor of shape (batch_size, n_critics * n_quantiles - drop_top)
    """
    batch_size, n_critics, n_quantiles = target_quantiles.shape
    n_total = n_critics * n_quantiles

    # Flatten all ensemble quantiles into a single pool per batch element: (batch_size, n_total)
    flat_quantiles = target_quantiles.reshape(batch_size, n_total)

    # Sort in ascending order
    sorted_quantiles, _ = torch.sort(flat_quantiles, dim=-1)

    # Truncate top d atoms
    if drop_top > 0:
        k = n_total - drop_top
        retained_quantiles = sorted_quantiles[:, :k]
    else:
        retained_quantiles = sorted_quantiles

    return retained_quantiles


def quantile_huber_loss(
    quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """
    Pairwise Quantile Huber Loss.

    Args:
        quantiles: Predicted quantiles from online critics.
                   Shape (batch_size, n_quantiles) or (batch_size, n_critics, n_quantiles)
        target_quantiles: Shifted target atoms.
                          Shape (batch_size, n_target_quantiles)
        kappa: Huber threshold (default 1.0)

    Returns:
        Scalar loss tensor averaged over batch, critics, and quantile pairs.
    """
    # If quantiles has shape (B, M, N):
    if quantiles.dim() == 3:
        # (B, M, N, 1) vs (B, 1, 1, K)
        n_quantiles = quantiles.shape[-1]
        pairwise_delta = (
            target_quantiles.unsqueeze(1).unsqueeze(2) - quantiles.unsqueeze(-1)
        )
        # tau values for N quantiles: tau_i = (2i - 1) / (2N)
        tau = (
            torch.arange(n_quantiles, device=quantiles.device, dtype=torch.float32)
            + 0.5
        ) / n_quantiles
        # tau shape: (1, 1, N, 1)
        tau = tau.view(1, 1, n_quantiles, 1)
    elif quantiles.dim() == 2:
        # (B, N, 1) vs (B, 1, K)
        n_quantiles = quantiles.shape[-1]
        pairwise_delta = target_quantiles.unsqueeze(1) - quantiles.unsqueeze(-1)
        tau = (
            torch.arange(n_quantiles, device=quantiles.device, dtype=torch.float32)
            + 0.5
        ) / n_quantiles
        tau = tau.view(1, n_quantiles, 1)
    else:
        raise ValueError(f"Unexpected quantiles tensor dimension: {quantiles.dim()}")

    abs_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(
        abs_delta <= kappa,
        0.5 * pairwise_delta.pow(2),
        kappa * (abs_delta - 0.5 * kappa),
    )

    # Indicator condition for asymmetric quantile penalty
    indicator = (pairwise_delta.detach() < 0.0).float()
    weight = torch.abs(tau - indicator) / kappa

    loss = (weight * huber_loss).mean()
    return loss

"""Reusable initial-state samplers."""
import math
from typing import Union
import torch


def sample_x0(
    batch_size: int,
    N: int,
    mean: float,
    std: float,
    device: Union[str, torch.device],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return mean + std * torch.randn(batch_size, N, device=device, dtype=dtype)


def sample_x0_structured(
    batch_size: int,
    xi: torch.Tensor,
    noise_std: float = 0.2,
) -> torch.Tensor:
    """
    xi: (N,)
    Sinusoidal base profile with additive noise.
    """
    base = 1.5 * torch.sin(2 * torch.pi * xi) + 0.8 * torch.cos(4 * torch.pi * xi)
    return base.unsqueeze(0) + noise_std * torch.randn(batch_size, xi.numel(), device=xi.device, dtype=xi.dtype)


def sample_x0_two_clusters(
    batch_size: int,
    xi: torch.Tensor,
    noise_std: float = 0.15,
) -> torch.Tensor:
    """
    xi: (N,)
    Agents with xi < 0.5 start near -1, the rest near +1.
    """
    base = torch.where(xi < 0.5, torch.tensor(-1.0, device=xi.device, dtype=xi.dtype), torch.tensor(1.0, device=xi.device, dtype=xi.dtype))
    return base.unsqueeze(0) + noise_std * torch.randn(batch_size, xi.numel(), device=xi.device, dtype=xi.dtype)


def sample_x0_linear_label(
    batch_size: int,
    xi: torch.Tensor,
    noise_std: float = 0.15,
) -> torch.Tensor:
    """
    xi: (N,)
    Base profile 20*(ξ - 1/2) with additive Gaussian noise.
    Agents with higher labels start at higher states.
    """
    base = 20.0 * (xi - 0.5)
    return base.unsqueeze(0) + noise_std * torch.randn(batch_size, xi.numel(), device=xi.device, dtype=xi.dtype)


def sample_x0_linear_sine(
    batch_size: int,
    xi: torch.Tensor,
    amplitude: float = 7.0,
    n_periods: float = 5.0,
    noise_std: float = 1.0,
) -> torch.Tensor:
    """
    xi: (N,)
    Base profile: 20*(ξ - 1/2) + amplitude * sin(2π * n_periods * ξ)
    with additive Gaussian noise. Combines a linear label-dependent slope
    with multi-oscillation sinusoidal structure.
    """
    base = 20.0 * (xi - 0.5) + amplitude * torch.sin(2 * math.pi * n_periods * xi)
    return base.unsqueeze(0) + noise_std * torch.randn(batch_size, xi.numel(), device=xi.device, dtype=xi.dtype)

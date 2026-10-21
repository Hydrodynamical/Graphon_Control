"""Differentiable finite-agent Euler and Euler–Maruyama dynamics.

W contains quadrature weights: W_ij = wbar_ij/N. Consequently the drift is
sum_j W_ij * phi(x_j - x_i), with no additional factor of 1/N.
"""

import math
from typing import Callable

import torch

from .validation import finite_scalar, n_time_steps, positive_integer


def mean_field_kernel(N: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """The constant graphon w=1, represented by W_ij=1/N."""
    positive_integer(N, "N")
    return torch.full((N, N), 1.0 / N, device=device, dtype=dtype)


class ConstantKernel:
    """Expose a fixed baseline through the controller's scalar/batched API."""

    def __init__(self, W0: torch.Tensor):
        if W0.ndim != 2 or W0.shape[0] != W0.shape[1] or W0.shape[0] == 0:
            raise ValueError("W0 must have nonempty square shape (N, N)")
        if not W0.is_floating_point() or not torch.isfinite(W0).all():
            raise ValueError("W0 must contain finite floating point weights")
        self.W0 = W0

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            return self.W0
        if t.ndim != 1:
            raise ValueError("time must be scalar or one-dimensional")
        return self.W0.unsqueeze(0).expand(t.shape[0], *self.W0.shape)


def _model_dtype(model):
    for name in ("xi", "W0"):
        value = getattr(model, name, None)
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.dtype
    if hasattr(model, "parameters"):
        for parameter in model.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
    return torch.get_default_dtype()


def precompute_Ws(model, NT: int, dt: float, device, chunk: int) -> torch.Tensor:
    """Sample W(t_n) in chunks, preserving its gradient and (NT,N,N) shape."""
    positive_integer(NT, "NT")
    positive_integer(chunk, "chunk")
    finite_scalar(dt, "dt", positive=True)
    times = torch.arange(NT, device=device, dtype=_model_dtype(model)) * dt
    outputs = []
    for start in range(0, NT, chunk):
        batch = times[start:start + chunk]
        W = model(batch)
        # Compatibility with older controllers that squeeze one-element batches.
        if W.ndim == 2 and batch.numel() == 1:
            W = W.unsqueeze(0)
        if W.ndim != 3 or W.shape[0] != batch.numel() or W.shape[-2] != W.shape[-1] or W.shape[-1] == 0:
            raise ValueError("controller must return (batch, N, N) for batched times")
        outputs.append(W)
    return torch.cat(outputs, dim=0)


def rollout_with_Ws(
    Ws: torch.Tensor,
    x0: torch.Tensor,
    dt: float,
    phi: Callable[[torch.Tensor], torch.Tensor],
    sigma: float = 0.0,
    generator: torch.Generator | None = None,
    brownian_increments: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate a batch of states; return shape (NT+1,B,N).

    Optional ``brownian_increments`` has shape (NT,B,N) and contains actual
    Brownian increments ΔB (variance dt), rather than standardized normals.
    It permits paired stochastic comparisons and exact replay of a rollout.
    A generator controls automatic draws and must match the state device.
    """
    dt = finite_scalar(dt, "dt", positive=True)
    sigma = finite_scalar(sigma, "sigma", nonnegative=True)
    if x0.ndim != 2 or min(x0.shape) == 0 or not x0.is_floating_point():
        raise ValueError("x0 must have nonempty floating point shape (B, N)")
    B, N = x0.shape
    if Ws.ndim != 3 or Ws.shape[0] == 0 or Ws.shape[1:] != (N, N):
        raise ValueError("Ws must have shape (NT, N, N) with NT >= 1")
    if Ws.device != x0.device or Ws.dtype != x0.dtype:
        raise ValueError("Ws and x0 must have the same device and dtype")
    if brownian_increments is not None:
        if generator is not None:
            raise ValueError("supply brownian_increments or generator, not both")
        if brownian_increments.shape != (Ws.shape[0], B, N):
            raise ValueError("brownian_increments must have shape (NT, B, N)")
        if brownian_increments.device != x0.device or brownian_increments.dtype != x0.dtype:
            raise ValueError("brownian_increments must match x0 device and dtype")
    x = x0
    xs = [x]
    for n, W in enumerate(Ws):
        # diff[b,i,j] = x_j - x_i: receiver is the row, sender is the column.
        diff = x.unsqueeze(1) - x.unsqueeze(2)
        drift = (W.unsqueeze(0) * phi(diff)).sum(dim=-1)
        x = x + dt * drift
        if sigma:
            dB = (brownian_increments[n] if brownian_increments is not None else
                  math.sqrt(dt) * torch.randn(x.shape, device=x.device, dtype=x.dtype,
                                             generator=generator))
            x = x + sigma * dB
        xs.append(x)
    return torch.stack(xs)


def rollout_dynamics_w1(model, x0, T, dt, sigma, phi, *, generator=None,
                        brownian_increments=None):
    """Compatibility adapter returning lists of states and kernels."""
    NT = n_time_steps(T, dt)
    # Keep the generic API's scalar-time evaluations to support custom models.
    Ws = torch.stack([model(x0.new_tensor(n * dt)) for n in range(NT)])
    xs = rollout_with_Ws(Ws, x0, dt, phi, sigma, generator, brownian_increments)
    return list(xs.unbind(0)), list(Ws.unbind(0))

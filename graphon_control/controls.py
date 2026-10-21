"""Time-and-label neural interaction controls, independent of experiment plotting.

The directed controller implements the manuscript's row-softmax parameterization:
``W[i, j]`` is the weight that receiver i assigns sender j.  The quadrature
factor is already included in W, so the associated graphon density is ``N * W``.
Self-interactions are permitted.  No prescribed receiver-variation bound is
enforced by this module.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite_scalar(name: str, value: Real, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        requirement = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {requirement}")
    return value


class FourierFeatures(nn.Module):
    """Dyadic sine/cosine features, with all sine terms followed by cosine terms."""

    def __init__(self, num_freqs: int):
        super().__init__()
        _positive_int("num_freqs", num_freqs)
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        if not torch.isfinite(freqs).all():
            raise ValueError("num_freqs produces nonfinite Fourier frequencies")
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        return 2 * self.num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map labels of shape ``(...)`` to features of shape ``(..., 2*m)``."""
        angles = 2.0 * math.pi * x.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class MLP(nn.Module):
    """A ReLU multilayer perceptron with a linear output layer."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2):
        super().__init__()
        for name, value in (("in_dim", in_dim), ("hidden_dim", hidden_dim), ("out_dim", out_dim)):
            _positive_int(name, value)
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ValueError("depth must be a nonnegative integer")
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.extend((nn.Linear(d, hidden_dim), nn.ReLU()))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeLabelGraphonControl(nn.Module):
    """Learn an open-loop interaction schedule from time and fixed agent labels.

    ``xi`` is a nonempty, nondecreasing floating tensor in [0, 1].  Model
    parameters and buffers inherit its device and dtype.  A scalar time returns
    ``(N, N)``; a time vector of length B returns ``(B, N, N)``, including B=1.

    Directed mode is row-stochastic, as in manuscript equations (64)--(67).
    Optional ``score_bound=S`` uses ``S*tanh(raw_score)`` and guarantees
    ``N*W[i,j] <= exp(2*S)``.  The default leaves scores unconstrained, preserving
    existing experiment behavior.  Neither setting imposes a prescribed BV bound.

    The legacy symmetric alternative uses one shared embedding, symmetric
    positive weights, and mean-row normalization.  Its individual row sums may
    differ from one, so it lies outside the manuscript's row-stochastic class.
    ``positive_map`` affects only that alternative.

    Module/buffer names and parameter ordering match legacy checkpoints from
    ``model_1.TimeLabelGraphonControl``.  ``score_bound`` is configuration, not a
    learned parameter; restore it alongside other constructor settings.
    """

    def __init__(
        self,
        xi: torch.Tensor,
        num_fourier_freqs: int,
        embed_dim: int,
        hidden_embed: int,
        hidden_score: int,
        time_scale: float = 1.0,
        enforce_symmetry: bool = False,
        positive_map: str = "softplus",
        score_bound: float | None = None,
    ):
        super().__init__()
        if not isinstance(xi, torch.Tensor) or not xi.is_floating_point():
            raise ValueError("xi must be a floating-point tensor")
        if xi.ndim != 1 or xi.numel() == 0:
            raise ValueError("xi must be a nonempty one-dimensional label grid")
        if not torch.isfinite(xi).all() or torch.any((xi < 0) | (xi > 1)):
            raise ValueError("xi must contain finite labels in [0, 1]")
        if torch.any(xi[1:] < xi[:-1]):
            raise ValueError("xi must be sorted in nondecreasing order")
        for name, value in (("num_fourier_freqs", num_fourier_freqs), ("embed_dim", embed_dim),
                            ("hidden_embed", hidden_embed), ("hidden_score", hidden_score)):
            _positive_int(name, value)
        self.time_scale = _finite_scalar("time_scale", time_scale, positive=True)
        if not isinstance(enforce_symmetry, bool):
            raise ValueError("enforce_symmetry must be a boolean")
        if positive_map not in ("softplus", "exp"):
            raise ValueError("positive_map must be 'softplus' or 'exp'")
        if score_bound is not None:
            score_bound = _finite_scalar("score_bound", score_bound)
            if score_bound < 0:
                raise ValueError("score_bound must be nonnegative")
        self.score_bound = score_bound
        self.enforce_symmetry = enforce_symmetry
        self.positive_map = positive_map
        # Labels are fixed model data; clone so callers cannot invalidate cached features.
        self.register_buffer("xi", xi.detach().clone())
        self.N = xi.numel()

        self.ff = FourierFeatures(num_fourier_freqs)
        ff_dim = self.ff.out_dim
        # Preserve legacy construction order for seeded runs and optimizer states.
        self.embed_net = MLP(ff_dim, hidden_embed, embed_dim) if enforce_symmetry else None
        self.receiver_net = None if enforce_symmetry else MLP(ff_dim, hidden_embed, embed_dim)
        self.sender_net = None if enforce_symmetry else MLP(ff_dim, hidden_embed, embed_dim)
        self.score_net = MLP(1 + 2 * embed_dim, hidden_score, 1)

        # Move frequencies before evaluating labels: xi may already be on CUDA/MPS.
        self.to(device=xi.device, dtype=xi.dtype)
        if not torch.isfinite(self.ff.freqs).all():
            raise ValueError("Fourier frequencies are not representable in xi.dtype")
        features = self.ff(self.xi)
        if not torch.isfinite(features).all():
            raise ValueError("Fourier features are not representable in xi.dtype")
        self.register_buffer("psi_xi_cached", features)

    def forward(self, t: torch.Tensor | float) -> torch.Tensor:
        """Evaluate scalar or one-dimensional finite times on the fixed grid."""
        if isinstance(t, torch.Tensor) and t.is_complex():
            raise ValueError("t must contain real finite times")
        t = torch.as_tensor(t, device=self.xi.device, dtype=self.xi.dtype)
        if t.ndim > 1 or t.numel() == 0:
            raise ValueError("t must be a scalar or a nonempty one-dimensional vector")
        if not torch.isfinite(t).all():
            raise ValueError("t must contain finite times")
        scalar = t.ndim == 0
        t = t.reshape(-1)
        B, N = t.numel(), self.N
        t_in = (t / self.time_scale).reshape(B, 1, 1, 1).expand(B, N, N, 1)
        if not torch.isfinite(t_in).all():
            raise ValueError("scaled times must be finite in the controller dtype")

        if self.enforce_symmetry:
            h_receiver = h_sender = self.embed_net(self.psi_xi_cached)
        else:
            h_receiver = self.receiver_net(self.psi_xi_cached)
            h_sender = self.sender_net(self.psi_xi_cached)
        d = h_receiver.shape[-1]
        h_i = h_receiver.view(1, N, 1, d).expand(B, N, N, d)
        h_j = h_sender.view(1, 1, N, d).expand(B, N, N, d)
        scores = self.score_net(torch.cat((t_in, h_i, h_j), dim=-1)).squeeze(-1)
        if self.enforce_symmetry:
            scores = 0.5 * (scores + scores.transpose(-1, -2))
        if self.score_bound is not None:
            scores = self.score_bound * torch.tanh(scores)

        if self.enforce_symmetry:
            if self.positive_map == "exp":
                log_weights = scores.clamp(max=20.0)
                weights = torch.exp(log_weights - log_weights.amax(dim=(-1, -2), keepdim=True))
            else:
                weights = F.softplus(scores)
            # Rescaling first avoids overflow in row sums; tiny floors avoid 0/0
            # when softplus underflows.  Both operations preserve symmetry.
            weights = weights.clamp_min(torch.finfo(weights.dtype).tiny)
            weights = weights / weights.amax(dim=(-1, -2), keepdim=True)
            W = weights / weights.sum(dim=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
        else:
            W = F.softmax(scores, dim=-1)
        return W[0] if scalar else W

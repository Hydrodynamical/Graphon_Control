"""Scientific diagnostics; measured variation is not an enforced BV budget."""

import math

import torch


def _check_kernel(W: torch.Tensor) -> None:
    if W.ndim < 2 or W.shape[-2] != W.shape[-1] or W.shape[-1] == 0:
        raise ValueError("kernel must have shape (..., N, N), N >= 1")


def vxi_discrete(W: torch.Tensor) -> torch.Tensor:
    """Receiver variation of graphon values wbar=N*W, preserving batch axes.

    V_xi = sum_i sqrt((1/N) sum_j (wbar[i+1,j]-wbar[i,j])²).
    The receiver interval has no periodic closing edge.
    """
    _check_kernel(W)
    N = W.shape[-1]
    differences = N * (W[..., 1:, :] - W[..., :-1, :])
    return torch.linalg.vector_norm(differences, dim=-1).sum(dim=-1) / math.sqrt(N)


def kernel_constraint_report(W: torch.Tensor) -> dict[str, torch.Tensor]:
    """Report constraints and variation without asserting admissibility."""
    _check_kernel(W)
    return {
        "min_weight": W.amin(dim=(-2, -1)),
        "max_row_sum_error": (W.sum(-1) - 1).abs().amax(dim=-1),
        "max_symmetry_error": (W - W.transpose(-1, -2)).abs().amax(dim=(-2, -1)),
        "receiver_variation": vxi_discrete(W),
    }

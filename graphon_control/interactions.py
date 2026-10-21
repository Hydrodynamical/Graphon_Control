"""Reusable interaction laws."""
import math
from typing import Union
import torch


def phi_default(y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Default saturating interaction φ(y) = alpha * tanh(y).

    The rollout evaluates φ at x_j - x_i (paper convention, eqn FN definition
    in main5.tex), so alpha > 0 yields attractive, consensus-seeking drift.
    """
    return alpha * torch.tanh(y)


def phi_linear(y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Linear interaction φ(y) = alpha * y.

    The rollout evaluates φ at x_j - x_i (paper convention, eqn FN definition
    in main5.tex), so alpha > 0 yields attractive, consensus-seeking drift.
    """
    return alpha * y

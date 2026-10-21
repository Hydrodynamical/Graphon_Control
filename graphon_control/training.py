"""Shared optimization loop with finite loss and gradient checks."""

import math
from collections.abc import Callable, Mapping
from numbers import Real

import torch
from tqdm.auto import tqdm

from .validation import finite_scalar, positive_integer


def _metric_value(value, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"objective component {name!r} must be scalar")
        value = float(value.detach())
    if not isinstance(value, Real) or not math.isfinite(value):
        raise FloatingPointError(f"objective component {name!r} is not finite: {value!r}")
    return float(value)


def train_controller(model, objective: Callable[[], Mapping], num_steps: int, lr: float,
                     grad_clip: float | None = 1.0, description: str = "training"):
    """Optimize a fresh stochastic objective each iteration with Adam.

    ``objective`` returns a mapping whose ``total`` is a differentiable scalar
    tensor. Other values may be scalar tensors or numeric metrics. History
    records ``total`` under ``loss`` and preserves the remaining component keys.
    Nonfinite values raise before an optimizer update can corrupt parameters.
    """
    positive_integer(num_steps, "num_steps")
    finite_scalar(lr, "lr", positive=True)
    if grad_clip is not None:
        finite_scalar(grad_clip, "grad_clip", positive=True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model must have trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=lr)
    history = {"loss": []}
    expected_keys = None
    last_parts = {}
    for step in tqdm(range(num_steps), desc=description, unit="step"):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        parts = objective()
        if not isinstance(parts, Mapping) or "total" not in parts:
            raise ValueError("objective must return a mapping containing 'total'")
        total = parts["total"]
        if not isinstance(total, torch.Tensor) or total.ndim != 0 or not total.requires_grad:
            raise ValueError("objective 'total' must be a differentiable scalar tensor")
        if "loss" in parts:
            raise ValueError("objective component 'loss' is reserved for total history")
        if expected_keys is None:
            expected_keys = set(parts)
            history.update({key: [] for key in parts if key != "total"})
        elif set(parts) != expected_keys:
            raise ValueError("objective component keys changed during training")
        last_parts = {key: _metric_value(value, key) for key, value in parts.items()}
        total.backward()
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients:
            raise ValueError("objective is disconnected from the controller parameters")
        if any(not torch.isfinite(gradient).all() for gradient in gradients):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"nonfinite controller gradient at optimization step {step}")
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in parameters):
            raise FloatingPointError(f"nonfinite controller parameter at optimization step {step}")
        history["loss"].append(last_parts["total"])
        for key in parts:
            if key != "total":
                history[key].append(last_parts[key])
    return history, last_parts

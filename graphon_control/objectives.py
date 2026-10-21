"""One discretized objective shared by generic and paper experiment APIs.

The paper coefficient is beta * ||w||², without an additional 1/2. Here
W=wbar/N, so ||w||² is represented by sum_ij W_ij². The optional uniform
regularizer retains its separate legacy convention eta_uniform/2.
"""

from collections.abc import Sequence

import torch

from .dynamics import mean_field_kernel, rollout_dynamics_w1
from .validation import finite_scalar, n_time_steps


def empirical_mean(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=-1)


def empirical_variance(x: torch.Tensor) -> torch.Tensor:
    """Population variance along the agent axis, using a stable centered sum."""
    return (x - x.mean(dim=-1, keepdim=True)).square().mean(dim=-1)


def resolve_target_schedule(T, dt, gamma_target, target_times, target_values,
                            target_weights) -> list[dict[str, float]]:
    """Snap checkpoint targets to the nearest rollout time, including endpoints."""
    NT = n_time_steps(T, dt)
    finite_scalar(gamma_target, "gamma_target", nonnegative=True)
    if len(target_times) != len(target_values):
        raise ValueError("target_times and target_values must have the same length")
    if len(target_weights) not in (0, len(target_times)):
        raise ValueError("target_weights must be empty or match target_times")
    if len(target_times) == 0:
        return []
    if len(target_weights) == 0 and gamma_target == 0:
        raise ValueError("scheduled targets require target_weights or nonzero gamma_target")
    weights = target_weights if len(target_weights) else [gamma_target] * len(target_times)
    schedule = []
    for time, value, weight in zip(target_times, target_values, weights):
        time = finite_scalar(time, "target time")
        value = finite_scalar(value, "target value")
        weight = finite_scalar(weight, "target weight", nonnegative=True)
        if not 0 <= time <= T:
            raise ValueError(f"target time {time} must lie in [0, T={T}]")
        index = min(NT, max(0, round(time / dt)))
        schedule.append({"index": float(index), "time": index * dt,
                         "value": value, "weight": weight})
    return schedule


def graphon_l2_penalty(W: torch.Tensor) -> torch.Tensor:
    """Approximate ||w||²; batched inputs retain their leading axes."""
    if W.ndim < 2 or W.shape[-2] != W.shape[-1] or W.shape[-1] == 0:
        raise ValueError("W must have shape (..., N, N), N >= 1")
    return W.square().sum(dim=(-2, -1))


def uniform_graphon_target(N, device=None, dtype=torch.float32):
    return mean_field_kernel(N, device, dtype)


def graphon_uniform_penalty(W: torch.Tensor) -> torch.Tensor:
    """Approximate ||w-1||² in the quadrature-weight representation."""
    # Validate shape before constructing the uniform target.
    graphon_l2_penalty(W)
    return graphon_l2_penalty(W - uniform_graphon_target(W.shape[-1], W.device, W.dtype))


def trajectory_cost(
    xs: torch.Tensor,
    Ws: torch.Tensor,
    *,
    dt: float,
    beta: float,
    lambda_terminal: float = 0.0,
    gamma_target: float = 0.0,
    x_target: float = 0.0,
    target_schedule: Sequence[dict] | None = None,
    eta_uniform: float = 0.0,
    regularizer_mode: str = "l2",
    weighted_running_cost: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute tensor-valued components for xs=(NT+1,B,N), Ws=(NT,N,N).

    Running terms use the left endpoint at each Euler step. The optional
    running-state ramp is (n+1)/NT, matching the former generic trainer.
    ``terminal_var`` is the unweighted diagnostic; its weighted contribution
    is ``terminal_var_cost``.
    """
    finite_scalar(dt, "dt", positive=True)
    for name, value in (("beta", beta), ("lambda_terminal", lambda_terminal),
                        ("gamma_target", gamma_target), ("eta_uniform", eta_uniform)):
        finite_scalar(value, name, nonnegative=True)
    finite_scalar(x_target, "x_target")
    if regularizer_mode not in ("l2", "uniform", "l2_plus_uniform"):
        raise ValueError("regularizer_mode must be 'l2', 'uniform', or 'l2_plus_uniform'")
    if xs.ndim != 3 or min(xs.shape) == 0 or xs.shape[0] < 2:
        raise ValueError("xs must have shape (NT+1, B, N), with NT >= 1")
    NT, N = xs.shape[0] - 1, xs.shape[-1]
    if Ws.shape != (NT, N, N):
        raise ValueError("Ws must have shape (NT, N, N) matching xs")
    if Ws.device != xs.device or Ws.dtype != xs.dtype:
        raise ValueError("Ws and xs must have the same device and dtype")
    zero = xs.new_zeros(())
    variance = empirical_variance(xs).mean(dim=-1)
    running_variance = variance[:-1]
    if weighted_running_cost:
        running_variance = running_variance * torch.arange(1, NT + 1, device=xs.device,
                                                           dtype=xs.dtype) / NT
    state_cost = dt * running_variance.sum()
    l2_cost = (dt * beta * graphon_l2_penalty(Ws).sum()
               if regularizer_mode in ("l2", "l2_plus_uniform") else zero)
    uniform_cost = (dt * eta_uniform / 2 * graphon_uniform_penalty(Ws).sum()
                   if regularizer_mode in ("uniform", "l2_plus_uniform") else zero)
    means_T = empirical_mean(xs[-1])
    target_tracking = (means_T - x_target).square().mean()
    terminal_var_cost = lambda_terminal * variance[-1]
    terminal_target_cost = gamma_target * target_tracking
    scheduled_cost = zero
    for target in target_schedule or ():
        index = target["index"]
        if int(index) != index or not 0 <= index <= NT:
            raise ValueError("target index must be an integer in [0, NT]")
        weight = finite_scalar(target["weight"], "target weight", nonnegative=True)
        value = finite_scalar(target["value"], "target value")
        tracking = (empirical_mean(xs[int(index)]) - value).square().mean()
        scheduled_cost = scheduled_cost + weight * tracking
    terminal_cost = terminal_var_cost + terminal_target_cost
    total = state_cost + l2_cost + uniform_cost + terminal_cost + scheduled_cost
    return {
        "total": total,
        "state_cost": state_cost,
        "l2_reg_cost": l2_cost,
        "uniform_cost": uniform_cost,
        "scheduled_target_cost": scheduled_cost,
        "terminal_var": variance[-1],
        "terminal_mean": means_T.mean(),
        "target_tracking": target_tracking,
        "terminal_var_cost": terminal_var_cost,
        "terminal_target_cost": terminal_target_cost,
        "terminal_target": terminal_target_cost,
        "terminal_cost": terminal_cost,
    }


def paper_cost(xs, Ws, cfg) -> dict[str, torch.Tensor]:
    """Shared paper objective with the experiment scripts' established keys."""
    parts = trajectory_cost(xs, Ws, dt=cfg.dt, beta=cfg.beta,
                            lambda_terminal=cfg.lambda_terminal,
                            gamma_target=cfg.gamma_target, x_target=cfg.x_target)
    return {key: parts[key] for key in ("total", "state_cost", "l2_reg_cost",
                                       "terminal_var", "terminal_target", "terminal_mean")}


def sampled_cost_JNdt(
    model, x0, T, dt, sigma, beta, lambda_terminal, phi,
    gamma_target=0.0, x_target=0.0, target_times=None, target_values=None,
    target_weights=None, eta_uniform=0.0, regularizer_mode="l2",
    weighted_running_cost=False,
):
    """Legacy generic adapter: return a differentiable loss and float metrics.

    The beta convention now agrees with the manuscript and both experiments.
    To retain the old generic regularization strength, divide beta by two.
    RNG streams can differ because zero-noise rollouts no longer draw noise.
    """
    schedule = resolve_target_schedule(T, dt, gamma_target, target_times or [],
                                       target_values or [], target_weights or [])
    xs, Ws = rollout_dynamics_w1(model, x0, T, dt, sigma, phi)
    parts = trajectory_cost(
        torch.stack(xs), torch.stack(Ws), dt=dt, beta=beta,
        lambda_terminal=lambda_terminal, gamma_target=gamma_target, x_target=x_target,
        target_schedule=schedule, eta_uniform=eta_uniform, regularizer_mode=regularizer_mode,
        weighted_running_cost=weighted_running_cost,
    )
    aux = {("total_cost" if key == "total" else key): float(value.detach())
           for key, value in parts.items() if key != "terminal_target"}
    return parts["total"], aux

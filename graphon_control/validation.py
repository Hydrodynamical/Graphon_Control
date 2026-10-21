"""Validation shared by the generic trainer and paper experiments."""

import math
from numbers import Integral, Real


def positive_integer(value, name: str) -> int:
    """Require an integer count, rejecting booleans and silent truncation."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return int(value)


def finite_scalar(value, name: str, *, positive=False, nonnegative=False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive; got {value!r}")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative; got {value!r}")
    return float(value)


def n_time_steps(T: float, dt: float) -> int:
    """Return T/dt only when it is a positive integer (up to roundoff).

    A partial final step is not implemented. Rejecting it prevents a rollout
    from silently stopping before or after its advertised horizon.
    """
    T = finite_scalar(T, "T", positive=True)
    dt = finite_scalar(dt, "dt", positive=True)
    ratio = T / dt
    if not math.isfinite(ratio):
        raise ValueError("T/dt must be a finite positive integer")
    NT = round(ratio)
    if NT < 1 or not math.isclose(ratio, NT, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"T/dt must be a positive integer; got T={T}, dt={dt}")
    return NT


def validate_common_config(cfg) -> None:
    """Validate common dataclass fields without coupling to one experiment."""
    n_time_steps(cfg.T, cfg.dt)
    for name in ("N", "batch_size", "num_steps", "num_fourier_freqs", "embed_dim",
                 "hidden_embed", "hidden_score", "time_chunk"):
        if hasattr(cfg, name):
            positive_integer(getattr(cfg, name), name)
    for name in ("lr", "time_scale"):
        if hasattr(cfg, name):
            if name == "time_scale" and cfg.time_scale is None:
                continue  # None explicitly selects the experiment horizon T.
            finite_scalar(getattr(cfg, name), name, positive=True)
    for name in ("sigma", "beta", "lambda_terminal", "gamma_target", "eta_uniform", "jitter"):
        if hasattr(cfg, name):
            finite_scalar(getattr(cfg, name), name, nonnegative=True)
    if getattr(cfg, "score_bound", None) is not None:
        finite_scalar(cfg.score_bound, "score_bound", nonnegative=True)
    if hasattr(cfg, "interaction_alpha"):
        finite_scalar(cfg.interaction_alpha, "interaction_alpha")
    if getattr(cfg, "grad_clip", None) is not None:
        finite_scalar(cfg.grad_clip, "grad_clip", positive=True)
    if hasattr(cfg, "x_target"):
        finite_scalar(cfg.x_target, "x_target")
    for name in ("seed", "evaluation_seed"):
        if hasattr(cfg, name):
            seed = getattr(cfg, name)
            if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
    if hasattr(cfg, "regularizer_mode") and cfg.regularizer_mode not in (
        "l2", "uniform", "l2_plus_uniform"
    ):
        raise ValueError("regularizer_mode must be 'l2', 'uniform', or 'l2_plus_uniform'")
    if hasattr(cfg, "positive_map") and cfg.positive_map not in ("softplus", "exp"):
        raise ValueError("positive_map must be 'softplus' or 'exp'")
    if any(hasattr(cfg, key) for key in ("target_times", "target_values", "target_weights")):
        # Local import avoids a module import cycle.
        from .objectives import resolve_target_schedule

        resolve_target_schedule(
            cfg.T, cfg.dt, getattr(cfg, "gamma_target", 0.0),
            getattr(cfg, "target_times", []), getattr(cfg, "target_values", []),
            getattr(cfg, "target_weights", []),
        )

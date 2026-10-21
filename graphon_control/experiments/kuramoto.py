"""Kuramoto untwisting experiment using the shared paper conventions.

See docs/NUMERICS.md for equations and docs/DEVELOPMENT.md for reproducible runs.
Model-specific rendering is in kuramoto_plots.py.
"""
import argparse
import datetime
import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from ..controls import TimeLabelGraphonControl
from ..artifacts import create_run_directory, write_run_provenance, save_checkpoint
from ..dynamics import ConstantKernel, mean_field_kernel, precompute_Ws, rollout_with_Ws
from ..objectives import paper_cost
from ..diagnostics import vxi_discrete
from ..training import train_controller
from ..validation import n_time_steps, validate_common_config
from ..cli import add_common_arguments, configuration_from_args
from pathlib import Path


# ============================================================
# Configuration (paper protocol, §5.1)
# ============================================================

@dataclass
class E1Config:
    # experiment
    k: int = 1                 # winding number of the twisted state
    jitter: float = 0.1        # i.i.d. jitter on the training initial data (resampled each batch)
    ring_range: float = 0.1    # range r of the smooth ring baseline
    x0_level: int = 0          # if > 0: use the coarse datum P_M(twist) with M = x0_level
                               # (the common datum of lem:AN-monotone; requires M | N)

    # discretization
    N: int = 64
    T: float = 5.0
    dt: float = 0.05
    sigma: float = 0.0         # deterministic dynamics (theory of §3 is deterministic)

    # cost — PAPER convention (main5.tex, eqn: JN definition / eqn:J-cost-1):
    #   J = Σ_n Δt [ Var(x^n) + β ||w(t_n)||²_{L²(I²)} ] + γ (mean(x^T) - x_target)²
    # (β, not β/2, and NO terminal-variance term; lambda_terminal kept as an
    #  optional knob but defaults to 0 to match the paper)
    beta: float = 10.0
    lambda_terminal: float = 0.0
    gamma_target: float = 100.0
    x_target: float = 0.0

    # optimization
    batch_size: int = 16
    num_steps: int = 2000
    lr: float = 1e-3
    grad_clip: float = 1.0

    # architecture (model_1.py defaults)
    num_fourier_freqs: int = 8
    embed_dim: int = 64
    hidden_embed: int = 128
    hidden_score: int = 128

    # bookkeeping
    evaluation_seed: int = 1729
    score_bound: Optional[float] = None
    seed: int = 0
    device: str = "cpu"
    time_chunk: int = 50       # chunk size for the batched W(t) precomputation

    def __post_init__(self):
        validate_common_config(self)
        if not isinstance(self.k, int) or isinstance(self.k, bool) or self.k < 1 or 2 * self.k >= self.N:
            raise ValueError("k must be positive and satisfy 2*k < N to resolve the twist")
        if isinstance(self.x0_level, bool) or not isinstance(self.x0_level, int):
            raise ValueError("x0_level must be an integer")
        if self.x0_level and (self.x0_level < 2 or self.N % self.x0_level):
            raise ValueError("x0_level must be zero or a divisor of N >= 2")
        if not math.isfinite(self.ring_range) or self.ring_range <= 0:
            raise ValueError("ring_range must be finite and positive")

    @property
    def NT(self) -> int:
        return n_time_steps(self.T, self.dt)


# ============================================================
# E1 ingredients
# ============================================================

def phi_sin(y: torch.Tensor) -> torch.Tensor:
    """Kuramoto interaction φ(z) = sin z (bounded, Lipschitz, odd; φ(0)=0)."""
    return torch.sin(y)


def twisted_state(xi: torch.Tensor, k: int) -> torch.Tensor:
    """k-twisted state x(0, ξ) = 2πk(ξ - 1/2)."""
    return 2.0 * math.pi * k * (xi - 0.5)


def coarse_twisted_state(xi: torch.Tensor, k: int, M: int) -> torch.Tensor:
    """
    P_M projection of the k-twisted state: block average over M cells.
    Since the twist is linear in ξ, the block average equals the twist at the
    M cell midpoints, repeated N/M times — piecewise constant on the level-M
    partition, hence exactly representable at every dyadic N ≥ M. This is the
    common initial datum x_M(0) of lem:AN-monotone / thm:main (main5.tex).
    """
    N = xi.numel()
    if N % M != 0:
        raise ValueError(f"N={N} must be a multiple of M={M}")
    mid = (torch.arange(M, device=xi.device, dtype=xi.dtype) + 0.5) / M
    vals = twisted_state(mid, k)                     # (M,)
    return vals.repeat_interleave(N // M)            # (N,)


def smooth_ring_kernel(xi: torch.Tensor, r: float) -> torch.Tensor:
    """
    Mollified ring kernel: smooth bump of range r in torus distance,
    row-normalized to a row-stochastic W. Smooth in ξ, hence of finite
    receiver variation (unlike the sharp indicator (1/2r) 1_{[-r,r]}).
    """
    d = (xi.unsqueeze(0) - xi.unsqueeze(1)).abs()
    d = torch.minimum(d, 1.0 - d)  # torus distance
    u = (d / r).clamp(max=1.0 - 1e-6)
    bump = torch.where(d < r, torch.exp(-1.0 / (1.0 - u ** 2)), torch.zeros_like(d))
    return bump / bump.sum(dim=1, keepdim=True)


# ============================================================
# Fast rollout: W depends only on t, so precompute all W(t_n)
# in a few batched forward passes, then run the Euler loop.
# ============================================================


def e1_cost(xs, Ws, cfg: E1Config):
    """Paper objective, equation (60); shared across both experiments."""
    return paper_cost(xs, Ws, cfg)


def order_parameter(xs: torch.Tensor) -> torch.Tensor:
    """
    Kuramoto order parameter r(t) = | (1/N) Σ_j e^{i x_j(t)} |.
    xs: (..., N) -> (...)
    """
    return torch.sqrt(torch.cos(xs).mean(-1) ** 2 + torch.sin(xs).mean(-1) ** 2)


def winding_number(xs: torch.Tensor) -> torch.Tensor:
    """
    Winding number of the state along the label circle:
      W(t) = (1/2π) [ Σ_i wrap(x_{i+1} - x_i) + wrap(x_1 - x_N) ],
    with wrap(d) = ((d + π) mod 2π) - π. The k-twisted state has W = k;
    consensus has W = 0. Untwisting proceeds in integer quanta: each unit of
    winding requires transporting the outer agents a full 2π of real phase
    (the sine coupling is blind to exact 2π gaps), reeled in through the
    intermediate agents. xs: (..., N) -> (...).
    """
    gaps = xs[..., 1:] - xs[..., :-1]
    close = (xs[..., :1] - xs[..., -1:])
    all_gaps = torch.cat([gaps, close], dim=-1)
    wrapped = torch.remainder(all_gaps + math.pi, 2.0 * math.pi) - math.pi
    return wrapped.sum(dim=-1) / (2.0 * math.pi)




# ============================================================
# Main experiment
# ============================================================

def run_e1(cfg: E1Config, tag: str = "", out_root: str = "out_1", fig_root: str = "figures",
           mirror_figs: bool = False, make_plots: bool = True):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    dtype = torch.float32

    cfg.__post_init__()
    run_dir = str(create_run_directory(out_root, "e1", tag or None))
    fig_dir = os.path.join(fig_root, os.path.basename(run_dir))
    if mirror_figs and make_plots and Path(fig_dir).exists():
        raise FileExistsError(f"Figure export already exists: {fig_dir}")
    write_run_provenance(run_dir, cfg, cfg.seed, cfg.evaluation_seed, device=cfg.device)

    xi = (torch.arange(cfg.N, device=device, dtype=dtype) + 0.5) / cfg.N
    if cfg.x0_level > 0:
        base = coarse_twisted_state(xi, cfg.k, cfg.x0_level)
    else:
        base = twisted_state(xi, cfg.k)
    NT = cfg.NT

    # Sanity assert: the twisted datum (smooth or coarse) must be exactly
    # stationary for the mean-field baseline (roots-of-unity cancellation);
    # if this fails, the "pinned baseline" claim of §5.2 is wrong for this grid.
    with torch.no_grad():
        drift0 = (mean_field_kernel(cfg.N, device) * phi_sin(
            base.unsqueeze(0) - base.unsqueeze(1))).sum(-1)
        pin_err = float(drift0.abs().max())
    if pin_err > 1e-5:
        raise RuntimeError(f"baseline pinning violated: |drift|={pin_err:.2e} "
                           f"(N={cfg.N}, k={cfg.k}, x0_level={cfg.x0_level})")

    print("=" * 64)
    print(f"E1 Kuramoto untwisting | k={cfg.k} N={cfg.N} T={cfg.T} dt={cfg.dt}")
    print(f"steps={cfg.num_steps} batch={cfg.batch_size} lr={cfg.lr} "
          f"beta={cfg.beta} gamma={cfg.gamma_target} lambda={cfg.lambda_terminal}")
    print(f"device={cfg.device} tag={tag}")
    print("=" * 64)

    # ---------- model ----------
    model = TimeLabelGraphonControl(
        xi=xi,
        num_fourier_freqs=cfg.num_fourier_freqs,
        embed_dim=cfg.embed_dim,
        hidden_embed=cfg.hidden_embed,
        hidden_score=cfg.hidden_score,
        time_scale=max(cfg.T, 1e-8),
        enforce_symmetry=False,
        score_bound=cfg.score_bound,
    ).to(device)
    def objective():
        Ws = precompute_Ws(model, NT, cfg.dt, device, cfg.time_chunk)
        x0 = base.unsqueeze(0) + cfg.jitter * torch.randn(
            cfg.batch_size, cfg.N, device=device, dtype=dtype)
        xs = rollout_with_Ws(Ws, x0, cfg.dt, phi_sin, sigma=cfg.sigma)
        return e1_cost(xs, Ws, cfg)

    history, _ = train_controller(model, objective, cfg.num_steps, cfg.lr,
                                  cfg.grad_clip, description="E1 training")
    torch.manual_seed(cfg.evaluation_seed)

    # ---------- evaluation: deterministic, exact twist, all three arms ----------
    model.eval()
    arms: Dict[str, object] = {
        "meanfield": ConstantKernel(mean_field_kernel(cfg.N, device)),
        "ring": ConstantKernel(smooth_ring_kernel(xi, cfg.ring_range)),
        "trained": model,
    }
    x0_eval = base.unsqueeze(0)  # (1, N), no jitter: the exact stationary point

    xs_arms_np: Dict[str, np.ndarray] = {}
    Ws_arms: Dict[str, torch.Tensor] = {}
    r_series: Dict[str, np.ndarray] = {}
    var_series: Dict[str, np.ndarray] = {}
    wind_series: Dict[str, np.ndarray] = {}
    cost_summary: Dict[str, Dict[str, float]] = {}

    with torch.no_grad():
        for arm, kernel in arms.items():
            Ws_eval = precompute_Ws(kernel, NT, cfg.dt, device, cfg.time_chunk)
            xs_eval = rollout_with_Ws(Ws_eval, x0_eval, cfg.dt, phi_sin, sigma=0.0)
            parts = e1_cost(xs_eval, Ws_eval, cfg)
            X = xs_eval[:, 0, :].cpu().numpy()          # (NT+1, N)
            xs_arms_np[arm] = X
            Ws_arms[arm] = Ws_eval
            r_series[arm] = order_parameter(xs_eval[:, 0, :]).cpu().numpy()
            var_series[arm] = xs_eval[:, 0, :].var(dim=1, unbiased=False).cpu().numpy()
            wind_series[arm] = winding_number(xs_eval[:, 0, :]).cpu().numpy()
            cost_summary[arm] = {kk: float(vv) for kk, vv in parts.items()}

    vxi_trained = vxi_discrete(Ws_arms["trained"]).cpu().numpy()   # (NT,)
    vxi_ring = float(vxi_discrete(Ws_arms["ring"][0]))
    vxi_mf = float(vxi_discrete(Ws_arms["meanfield"][0]))

    t_grid = np.arange(NT + 1) * cfg.dt
    t_grid_W = np.arange(NT) * cfg.dt

    # ---------- noisy ensemble evaluation (robustness runs, sigma > 0) ----------
    # The deterministic evaluation above supplies the paper's table values (the
    # noisy training loss carries a noise floor ~ sigma^2 T^2 / 2); here we
    # additionally roll the trained control on the noisy dynamics it was trained
    # for, over an ensemble of Brownian paths from the exact twist.
    noisy_eval: Dict[str, np.ndarray] = {}
    if cfg.sigma > 0.0:
        n_rep = 32
        with torch.no_grad():
            xs_noisy = rollout_with_Ws(Ws_arms["trained"], x0_eval.expand(n_rep, -1),
                                       cfg.dt, phi_sin, sigma=cfg.sigma)  # (NT+1, n_rep, N)
            r_rep = order_parameter(xs_noisy).cpu().numpy()               # (NT+1, n_rep)
            w_rep = winding_number(xs_noisy).cpu().numpy()
        noisy_eval = {
            "r_noisy_mean": r_rep.mean(axis=1),
            "r_noisy_q10": np.quantile(r_rep, 0.10, axis=1),
            "r_noisy_q90": np.quantile(r_rep, 0.90, axis=1),
            "winding_noisy_mean": w_rep.mean(axis=1),
            "winding_noisy_q10": np.quantile(w_rep, 0.10, axis=1),
            "winding_noisy_q90": np.quantile(w_rep, 0.90, axis=1),
        }

    # ---------- persistence ----------
    save_checkpoint(os.path.join(run_dir, "model.pt"), model)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    np.savez_compressed(
        os.path.join(run_dir, "trajectories.npz"),
        t=t_grid, t_W=t_grid_W, xi=xi.cpu().numpy(),
        x_meanfield=xs_arms_np["meanfield"], x_ring=xs_arms_np["ring"],
        x_trained=xs_arms_np["trained"],
        W_trained=Ws_arms["trained"].cpu().numpy().astype(np.float32),
        W_ring=Ws_arms["ring"][0].cpu().numpy().astype(np.float32),
        r_meanfield=r_series["meanfield"], r_ring=r_series["ring"],
        r_trained=r_series["trained"],
        var_meanfield=var_series["meanfield"], var_ring=var_series["ring"],
        var_trained=var_series["trained"],
        winding_meanfield=wind_series["meanfield"], winding_ring=wind_series["ring"],
        winding_trained=wind_series["trained"],
        vxi_trained=vxi_trained,
        **noisy_eval,
    )

    metrics = {
        "k": cfg.k,
        "costs": cost_summary,
        "r_initial": float(r_series["trained"][0]),
        "r_final": {arm: float(r_series[arm][-1]) for arm in arms},
        "var_final": {arm: float(var_series[arm][-1]) for arm in arms},
        "winding_final": {arm: float(wind_series[arm][-1]) for arm in arms},
        "terminal_mean_trained": cost_summary["trained"]["terminal_mean"],
        "vxi": {"trained_max": float(vxi_trained.max()),
                "trained_final": float(vxi_trained[-1]),
                "ring": vxi_ring, "meanfield": vxi_mf},
        "value_gap_meanfield_minus_trained":
            cost_summary["meanfield"]["total"] - cost_summary["trained"]["total"],
    }
    if noisy_eval:
        metrics["noisy_eval"] = {
            "sigma": cfg.sigma,
            "n_replicates": 32,
            "r_final_mean": float(noisy_eval["r_noisy_mean"][-1]),
            "r_final_q10": float(noisy_eval["r_noisy_q10"][-1]),
            "r_final_q90": float(noisy_eval["r_noisy_q90"][-1]),
            "winding_final_mean": float(noisy_eval["winding_noisy_mean"][-1]),
        }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if make_plots:
        plot_run(run_dir, fig_dir=fig_dir if mirror_figs else None)

    # ---------- summary ----------
    lines = [
        "=== E1 Kuramoto untwisting: run summary ===",
        f"timestamp : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"run_dir   : {run_dir}",
        f"k = {cfg.k}, N = {cfg.N}, T = {cfg.T}, dt = {cfg.dt}, steps = {cfg.num_steps}",
        "",
        "arm         J_total      r(T)      Var(T)    winding(T)",
    ]
    for arm in ["meanfield", "ring", "trained"]:
        lines.append(f"{arm:<11s} {cost_summary[arm]['total']:>9.4f}  "
                     f"{metrics['r_final'][arm]:>8.4f}  {metrics['var_final'][arm]:>10.6f}  "
                     f"{metrics['winding_final'][arm]:>+9.3f}")
    lines += [
        "",
        f"V_xi trained: max {metrics['vxi']['trained_max']:.3f}, "
        f"final {metrics['vxi']['trained_final']:.3f} "
        f"(ring {vxi_ring:.3f}, mean-field {vxi_mf:.3f})",
        f"value gap (mean-field - trained): "
        f"{metrics['value_gap_meanfield_minus_trained']:.4f}",
    ]
    summary = "\n".join(lines)
    with open(os.path.join(run_dir, "run_summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(summary)
    print(f"\nAll outputs in: {run_dir}")
    if mirror_figs and make_plots:
        print(f"Paper figures mirrored to: {fig_dir}")
    with open(os.path.join(run_dir, "complete.json"), "w") as f:
        json.dump({"status": "complete", "plots_generated": make_plots}, f)
    return metrics


# ============================================================
# CLI
# ============================================================


def plot_run(run_dir, fig_dir=None):
    """Regenerate figures from archived arrays; does not train or load executable code."""
    from .kuramoto_plots import (fig_r_panels, fig_r_combined, fig_winding, fig_variance, fig_vxi, fig_kernel_strip, fig_state_evolution, fig_training_history, save_untwist_gif, ARM_LABELS, ARM_COLORS, PAPER_ARMS)
    from ..plotting import save_graphon_gif
    cfg = E1Config(**json.loads((Path(run_dir) / "config.json").read_text()))
    history = json.loads((Path(run_dir) / "history.json").read_text())
    metrics = json.loads((Path(run_dir) / "metrics.json").read_text())
    with np.load(Path(run_dir) / "trajectories.npz", allow_pickle=False) as archive:
        data = dict(archive)
    NT = cfg.NT
    t_grid, t_grid_W = data["t"], data["t_W"]
    xi = torch.as_tensor(data["xi"])
    xs_arms_np = {arm: data[f"x_{arm}"] for arm in ("meanfield", "ring", "trained")}
    r_series = {arm: data[f"r_{arm}"] for arm in xs_arms_np}
    var_series = {arm: data[f"var_{arm}"] for arm in xs_arms_np}
    wind_series = {arm: data[f"winding_{arm}"] for arm in xs_arms_np}
    Ws_arms = {"trained": torch.as_tensor(data["W_trained"])}
    vxi_trained = data["vxi_trained"]
    vxi_ring = metrics["vxi"]["ring"]
    if fig_dir is not None:
        Path(fig_dir).mkdir(parents=True, exist_ok=False)
    # ---------- figures ----------
    def both(name: str) -> List[str]:
        if fig_dir is None:
            return [os.path.join(run_dir, name)]
        return [os.path.join(run_dir, name), os.path.join(fig_dir, name)]

    for p in both("r_panels.pdf"):
        fig_r_panels(t_grid, r_series, cfg.k, p, arms=PAPER_ARMS)
    for p in both("r_combined.pdf"):
        fig_r_combined(t_grid, r_series, cfg.k, p, arms=PAPER_ARMS)
    for p in both("variance.pdf"):
        fig_variance(t_grid, var_series, p, arms=PAPER_ARMS)
    for p in both("winding.pdf"):
        fig_winding(t_grid, wind_series, cfg.k, p, arms=PAPER_ARMS)
    for p in both("vxi.pdf"):
        fig_vxi(t_grid_W, vxi_trained, vxi_ring, p)
    for p in both("kernel_strip.pdf"):
        fig_kernel_strip(Ws_arms["trained"], cfg.dt, p)
    for p in both("state_evolution.pdf"):
        fig_state_evolution(t_grid, xi.cpu().numpy(), xs_arms_np, cfg.k, p,
                            arms=PAPER_ARMS)
    fig_training_history(history, os.path.join(run_dir, "training_history.pdf"))

    # ---------- GIFs ----------
    save_untwist_gif(t_grid, xi.cpu().numpy(), xs_arms_np, r_series, cfg.k,
                     os.path.join(run_dir, "untwist.gif"), arms=PAPER_ARMS)
    n_gif = min(60, NT)
    gif_idx = np.linspace(0, NT - 1, n_gif).round().astype(int)
    gif_frames = [Ws_arms["trained"][i].cpu() * cfg.N for i in gif_idx]
    save_graphon_gif(gif_frames, os.path.join(run_dir, "kernel_time.gif"),
                     fps=10, t_vals=[float(i * cfg.dt) for i in gif_idx])


    complete = Path(run_dir) / "complete.json"
    if complete.exists():
        payload = json.loads(complete.read_text())
        payload["plots_generated"] = True
        complete.write_text(json.dumps(payload) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Kuramoto twisted-state control")
    add_common_arguments(parser)
    parser.add_argument("--k", type=int)
    parser.add_argument("--x0-level", type=int, dest="x0_level")
    args = parser.parse_args(argv)
    try:
        if args.replot:
            plot_run(args.replot)
            return
        cfg = configuration_from_args(E1Config, args)
        run_e1(cfg, tag=args.tag, out_root=args.out_root, fig_root=args.fig_root,
               mirror_figs=args.mirror_figures, make_plots=not args.no_plots)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

# e1_kuramoto.py
#
# Experiment E1: untwisting a twisted state (Kuramoto).
# Paper reference: main5.tex, subsec:exp-kuramoto (protocol in subsec:exp-protocol).
#
#   dynamics    : dx_i/dt = (1/N) Σ_j wbar_ij(t) sin(x_j - x_i),  wbar = N * W row-softmax
#   initial data: k-twisted state  x(0, ξ) = 2πk(ξ - 1/2)  (+ jitter during training)
#   baselines   : (i) mean-field  w ≡ 1  (W_ij = 1/N),
#                 (ii) smooth ring bump of range r (mollified — the sharp ring
#                      indicator has infinite receiver variation V_ξ and is NOT
#                      in U_BV; see §5.2)
#   claim       : twisted states are stationary for every symmetric convolution
#                 kernel (odd integrand vs. even kernel), so both baselines stay
#                 pinned at order parameter r ≈ 0; the trained control must break
#                 the translational symmetry to untwist.
#
# Outputs (all under out_1/e1_<tag>/, paper-ready PDFs mirrored to
# figures/e1_k{k}_T{T}[_sigma*]/). All three arms are always evaluated and
# archived in trajectories.npz/metrics.json; the FIGURES show the arms in
# PAPER_ARMS (mean-field vs trained control, per the paper's two-arm design).
#   config.json, metrics.json, run_summary.txt, model.pt, history.json
#   trajectories.npz     — x(t) per arm, W(t) for trained, r/Var/V_ξ series
#                          (+ r_noisy_*/winding_noisy_* ensemble stats if σ > 0)
#   r_panels.pdf         — r(t), one panel per paper arm
#   r_combined.pdf       — same data, single axis
#   variance.pdf         — Var(x(t)) per arm (log scale)
#   winding.pdf          — winding number W(t) per arm: untwisting happens in
#                          integer quanta (each quantum = 2π of real transport
#                          for the outer agents; sin is blind to exact 2π gaps,
#                          and r(t)=1 only certifies consensus mod 2π)
#   vxi.pdf              — receiver variation V_ξ(ŵ(t)) of the trained kernel
#   kernel_strip.pdf     — trained kernel heatmaps at 6 times, shared color scale
#   state_evolution.pdf  — x(t, ξ) heatmaps per arm
#   training_history.pdf — loss components over training
#   untwist.gif          — agents as phases on the circle, all three arms animated
#   kernel_time.gif      — trained kernel w(t, ξ, ζ) animated
#
# Usage:
#   python e1_kuramoto.py                     # full run (k=1, N=64, 2000 steps)
#   python e1_kuramoto.py --k 2 --T 12        # deeper twist, longer horizon
#   python e1_kuramoto.py --k 3 --T 20 --steps 3000   # the paper's headline run
#   python e1_kuramoto.py --k 2 --T 12 --sigma 0.3    # noisy robustness run
#                                             # (adds 32-path noisy-ensemble eval)
#   python e1_kuramoto.py --smoke             # 20-step smoke test
#   python e1_kuramoto.py --device mps        # Apple GPU

import argparse
import dataclasses
import datetime
import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from model_1 import TimeLabelGraphonControl, _random_run_tag
from model_1_viz import imshow_graphon, save_graphon_gif


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
    seed: int = 0
    device: str = "cpu"
    time_chunk: int = 50       # chunk size for the batched W(t) precomputation

    @property
    def NT(self) -> int:
        return int(round(self.T / self.dt))


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


def mean_field_kernel(N: int, device, dtype=torch.float32) -> torch.Tensor:
    """Row-stochastic representation of w ≡ 1: W_ij = 1/N."""
    return torch.full((N, N), 1.0 / N, device=device, dtype=dtype)


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


class ConstantKernel:
    """Baseline wrapper exposing the same call signature as the trained model."""

    def __init__(self, W0: torch.Tensor):
        self.W0 = W0

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            return self.W0
        return self.W0.unsqueeze(0).expand(t.shape[0], *self.W0.shape)


# ============================================================
# Fast rollout: W depends only on t, so precompute all W(t_n)
# in a few batched forward passes, then run the Euler loop.
# ============================================================

def precompute_Ws(model, NT: int, dt: float, device, chunk: int) -> torch.Tensor:
    """Returns (NT, N, N); differentiable w.r.t. model parameters."""
    t_vals = torch.arange(NT, dtype=torch.float32, device=device) * dt
    outs = [model(t_vals[a:a + chunk]) for a in range(0, NT, chunk)]
    outs = [o.unsqueeze(0) if o.ndim == 2 else o for o in outs]  # chunk of 1 returns (N, N)
    return torch.cat(outs, dim=0)


def rollout_with_Ws(
    Ws: torch.Tensor,
    x0: torch.Tensor,
    dt: float,
    phi: Callable[[torch.Tensor], torch.Tensor],
    sigma: float = 0.0,
) -> torch.Tensor:
    """
    Euler(-Maruyama) rollout with precomputed kernels.
    Ws: (NT, N, N) row-stochastic; x0: (B, N).
    Sign convention (paper, eqn FN definition): diff[b, i, j] = x_j - x_i.
    Returns xs: (NT + 1, B, N).
    """
    B, N = x0.shape
    NT = Ws.shape[0]
    sqrt_dt = math.sqrt(dt)
    x = x0
    xs = [x]
    for n in range(NT):
        diff = x.unsqueeze(1) - x.unsqueeze(2)          # (B, N, N): x_j - x_i
        drift = (Ws[n].unsqueeze(0) * phi(diff)).sum(-1)  # (B, N)
        x = x + dt * drift
        if sigma > 0.0:
            x = x + sigma * sqrt_dt * torch.randn_like(x)
        xs.append(x)
    return torch.stack(xs, dim=0)


def e1_cost(xs: torch.Tensor, Ws: torch.Tensor, cfg: E1Config) -> Dict[str, torch.Tensor]:
    """
    Discretized cost J^{N,Δt}, PAPER convention (eqn: JN definition / eqn:J-cost-1):
      Σ_n Δt [ Var(x^n) + β ||w(t_n)||²_{L²(I²)} ] + γ (mean(x^T) - x_target)²
    where ||w||²_{L²(I²)} = Σ_ij W_ij² for the row-stochastic W (wbar = N·W).
    lambda_terminal · Var(x^T) is an optional extra knob, 0 in the paper cost.
    xs: (NT+1, B, N); Ws: (NT, N, N).
    """
    var_t = xs.var(dim=2, unbiased=False).mean(dim=1)        # (NT+1,) batch-averaged Var
    running_state = cfg.dt * var_t[:-1].sum()
    running_reg = cfg.dt * cfg.beta * (Ws ** 2).sum()
    terminal_var = cfg.lambda_terminal * var_t[-1]
    mean_T = xs[-1].mean(dim=1)                              # (B,)
    terminal_target = cfg.gamma_target * ((mean_T - cfg.x_target) ** 2).mean()
    total = running_state + running_reg + terminal_var + terminal_target
    return {
        "total": total,
        "state_cost": running_state,
        "l2_reg_cost": running_reg,
        "terminal_var": var_t[-1],
        "terminal_target": terminal_target,
        "terminal_mean": mean_T.mean(),
    }


# ============================================================
# Diagnostics
# ============================================================

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


def vxi_discrete(W: torch.Tensor) -> torch.Tensor:
    """
    Discrete receiver variation (paper eqn:receiver-tv-discrete) of the graphon
    values wbar = N * W:
      V_ξ = Σ_{i=1}^{N-1} sqrt( (1/N) Σ_j (wbar_{i+1,j} - wbar_{i,j})^2 ).
    Accepts (N, N) or batched (..., N, N).
    """
    wbar = W.shape[-1] * W
    d = wbar[..., 1:, :] - wbar[..., :-1, :]
    return torch.sqrt((d ** 2).mean(dim=-1)).sum(dim=-1)


# ============================================================
# Figures
# ============================================================

ARM_LABELS = {"meanfield": r"mean-field  $w \equiv 1$",
              "ring": "smooth ring",
              "trained": "trained control"}
ARM_COLORS = {"meanfield": "tab:gray", "ring": "tab:orange", "trained": "tab:blue"}


PAPER_ARMS = ["meanfield", "trained"]   # arms shown in paper figures (ring is
                                        # still evaluated and archived in npz)


def fig_r_panels(t_grid, r_series: Dict[str, np.ndarray], k: int, path: str,
                 arms: List[str] = None):
    arms = arms or ["meanfield", "ring", "trained"]
    fig, axes = plt.subplots(1, len(arms), figsize=(4.4 * len(arms), 3.6),
                             sharey=True, squeeze=False)
    for ax, arm in zip(axes[0], arms):
        for other in arms:
            if other != arm:
                ax.plot(t_grid, r_series[other], color="0.85", lw=1.0)
        ax.plot(t_grid, r_series[arm], color=ARM_COLORS[arm], lw=2.2)
        ax.set_title(ARM_LABELS[arm])
        ax.set_xlabel(r"$t$")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel(r"order parameter $r(t)$")
    fig.suptitle(rf"Untwisting the $k={k}$ twisted state", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_r_combined(t_grid, r_series, k: int, path: str, arms: List[str] = None):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for arm in (arms or ["meanfield", "ring", "trained"]):
        ax.plot(t_grid, r_series[arm], color=ARM_COLORS[arm], lw=2.0, label=ARM_LABELS[arm])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$r(t)$")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(loc="center right")
    ax.set_title(rf"Order parameter, $k={k}$ twisted state")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_winding(t_grid, wind_series: Dict[str, np.ndarray], k: int, path: str,
                arms: List[str] = None):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for arm in (arms or ["meanfield", "ring", "trained"]):
        ax.plot(t_grid, wind_series[arm], color=ARM_COLORS[arm], lw=2.0,
                label=ARM_LABELS[arm])
    for q in range(k + 1):
        ax.axhline(q, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"winding number $\mathcal{W}(t)$")
    ax.set_ylim(-0.25, k + 0.25)
    ax.grid(alpha=0.15)
    ax.legend()
    ax.set_title("Untwisting proceeds in integer quanta")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_variance(t_grid, var_series: Dict[str, np.ndarray], path: str,
                 arms: List[str] = None):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for arm in (arms or ["meanfield", "ring", "trained"]):
        ax.semilogy(t_grid, np.maximum(var_series[arm], 1e-12),
                    color=ARM_COLORS[arm], lw=2.0, label=ARM_LABELS[arm])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathrm{Var}(x(t))$")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    ax.set_title("Variance decay")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_vxi(t_grid_W, vxi_trained: np.ndarray, vxi_ring: float, path: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(t_grid_W, vxi_trained, color=ARM_COLORS["trained"], lw=2.0,
            label=r"trained $V_\xi(\hat{w}(t))$")
    ax.axhline(0.0, color=ARM_COLORS["meanfield"], lw=1.5, ls="--",
               label=r"mean-field ($V_\xi = 0$)")
    ax.axhline(vxi_ring, color=ARM_COLORS["ring"], lw=1.5, ls="--",
               label=rf"smooth ring ($V_\xi = {vxi_ring:.1f}$)")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$V_\xi$")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Receiver variation of the kernel (measured, not enforced)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_kernel_strip(Ws: torch.Tensor, dt: float, path: str, n_panels: int = 6):
    NT = Ws.shape[0]
    idxs = np.linspace(0, NT - 1, n_panels).round().astype(int)
    wbars = [Ws.shape[-1] * Ws[i].cpu().numpy() for i in idxs]
    vmin = min(w.min() for w in wbars)
    vmax = max(w.max() for w in wbars)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.6 * n_panels, 2.9))
    for ax, i, w in zip(axes, idxs, wbars):
        im = ax.imshow(w, origin="lower", extent=(0, 1, 0, 1), aspect="equal",
                       cmap="viridis", interpolation="bicubic", vmin=vmin, vmax=vmax)
        ax.set_title(rf"$t = {i * dt:.1f}$", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        if ax is axes[0]:
            ax.set_ylabel(r"$\xi$")
        ax.set_xlabel(r"$\zeta$")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label=r"$\hat{w}(t,\xi,\zeta)$")
    fig.suptitle("Trained kernel over time", y=1.04)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_state_evolution(t_grid, xi, xs_arms: Dict[str, np.ndarray], k: int, path: str,
                        arms: List[str] = None):
    arms = arms or ["meanfield", "ring", "trained"]
    lim = math.pi * max(k, 1) * 1.1
    fig, axes = plt.subplots(1, len(arms), figsize=(4.4 * len(arms), 3.6),
                             sharey=True, squeeze=False)
    for ax, arm in zip(axes[0], arms):
        X = xs_arms[arm]  # (NT+1, N)
        im = ax.imshow(X.T, origin="lower", aspect="auto",
                       extent=(t_grid[0], t_grid[-1], 0.0, 1.0),
                       cmap="twilight_shifted", vmin=-lim, vmax=lim)
        ax.set_title(ARM_LABELS[arm])
        ax.set_xlabel(r"$t$")
    axes = axes[0]
    axes[0].set_ylabel(r"label $\xi$")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label=r"$x(t,\xi)$")
    fig.suptitle("State evolution", y=1.04)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_training_history(history: Dict[str, List[float]], path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(history["loss"], lw=1.0, color="tab:blue")
    axes[0].set_title("total loss")
    axes[0].set_xlabel("step")
    axes[0].grid(alpha=0.25)
    for key, color in [("state_cost", "tab:green"), ("l2_reg_cost", "tab:red"),
                       ("terminal_var", "tab:purple"), ("terminal_target", "tab:brown")]:
        axes[1].semilogy(np.maximum(np.array(history[key]), 1e-10), lw=1.0, label=key, color=color)
    axes[1].set_title("components (log scale)")
    axes[1].set_xlabel("step")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_untwist_gif(t_grid, xi, xs_arms: Dict[str, np.ndarray],
                     r_series: Dict[str, np.ndarray], k: int, path: str,
                     stride: int = 2, fps: int = 12, dpi: int = 80,
                     arms: List[str] = None):
    """
    Animated circle view: each agent is a dot at angle x_i(t) on the unit circle,
    colored by its label ξ. One panel per arm.
    """
    from PIL import Image
    from io import BytesIO

    arms = arms or ["meanfield", "ring", "trained"]
    colors = plt.cm.hsv(xi)
    frames = list(range(0, len(t_grid), stride))
    if frames[-1] != len(t_grid) - 1:
        frames.append(len(t_grid) - 1)

    pil_imgs = []
    theta = np.linspace(0, 2 * np.pi, 200)
    for n in frames:
        fig, axes = plt.subplots(1, len(arms), figsize=(3.5 * len(arms), 3.8),
                                 squeeze=False)
        for ax, arm in zip(axes[0], arms):
            x_n = xs_arms[arm][n]  # (N,)
            ax.plot(np.cos(theta), np.sin(theta), color="0.85", lw=1.0, zorder=1)
            ax.scatter(np.cos(x_n), np.sin(x_n), c=colors, s=28,
                       edgecolors="none", zorder=2)
            ax.set_xlim(-1.25, 1.25)
            ax.set_ylim(-1.25, 1.25)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{ARM_LABELS[arm]}\n$r = {r_series[arm][n]:.3f}$", fontsize=10)
        fig.suptitle(rf"$k={k}$ twist:  $t = {t_grid[n]:.2f}$", y=0.99)
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        pil_imgs.append(Image.open(buf).copy())

    pil_imgs[0].save(path, save_all=True, append_images=pil_imgs[1:], loop=0,
                     duration=int(1000 / fps), optimize=False)


# ============================================================
# Main experiment
# ============================================================

def run_e1(cfg: E1Config, tag: str, out_root: str = "out_1", fig_root: str = "figures",
           mirror_figs: bool = True):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    dtype = torch.float32

    run_dir = os.path.join(out_root, f"e1_{tag}")
    # sigma > 0 runs mirror to their own figure dir so they never overwrite the
    # noiseless run at the same (k, T); dot-free suffix keeps LaTeX paths safe
    sigma_suffix = f"_sigma{cfg.sigma:g}".replace(".", "p") if cfg.sigma > 0 else ""
    fig_dir = os.path.join(fig_root, f"e1_k{cfg.k}_T{cfg.T:g}{sigma_suffix}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

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
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # ---------- training ----------
    history: Dict[str, List[float]] = {kk: [] for kk in
                                       ["loss", "state_cost", "l2_reg_cost",
                                        "terminal_var", "terminal_target", "terminal_mean"]}
    pbar = tqdm(range(cfg.num_steps), desc="E1 training", unit="step")
    for _ in pbar:
        model.train()
        optimizer.zero_grad()
        Ws = precompute_Ws(model, NT, cfg.dt, device, cfg.time_chunk)
        x0 = base.unsqueeze(0) + cfg.jitter * torch.randn(cfg.batch_size, cfg.N,
                                                          device=device, dtype=dtype)
        xs = rollout_with_Ws(Ws, x0, cfg.dt, phi_sin, sigma=cfg.sigma)
        parts = e1_cost(xs, Ws, cfg)
        parts["total"].backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        for key in ["loss", "state_cost", "l2_reg_cost",
                    "terminal_var", "terminal_target", "terminal_mean"]:
            src = "total" if key == "loss" else key
            history[key].append(float(parts[src].detach()))
        pbar.set_postfix(loss=f"{history['loss'][-1]:.3f}",
                         term_var=f"{history['terminal_var'][-1]:.4f}")

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
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
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

    # ---------- figures ----------
    def both(name: str) -> List[str]:
        if not mirror_figs:
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
    print(f"\nAll outputs in: {run_dir}\nPaper figures mirrored to: {fig_dir}")
    return metrics


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E1: Kuramoto twisted-state untwisting")
    parser.add_argument("--k", type=int, default=1, help="winding number of the twist")
    parser.add_argument("--T", type=float, default=5.0, help="time horizon")
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--x0-level", type=int, default=0, dest="x0_level",
                        help="if > 0, use the coarse datum P_M(twist) with M = this level")
    parser.add_argument("--no-mirror", action="store_true",
                        help="do not mirror figures into figures/ (sweep runs)")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.0,
                        help="Euler-Maruyama noise level (training dynamics and "
                             "the noisy ensemble evaluation)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--smoke", action="store_true", help="20-step smoke test")
    args = parser.parse_args()

    cfg = E1Config(k=args.k, T=args.T, N=args.N, num_steps=args.steps,
                   batch_size=args.batch, sigma=args.sigma, seed=args.seed,
                   device=args.device, x0_level=args.x0_level)
    if args.smoke:
        cfg.num_steps = 20

    tag = args.tag or f"k{args.k}_{_random_run_tag()}"
    if args.smoke:
        tag = f"smoke_{tag}"
    run_e1(cfg, tag=tag, mirror_figs=not args.no_mirror)

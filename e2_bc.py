# e2_bc.py
#
# Experiment E2: preventing opinion fragmentation (bounded confidence).
# Paper reference: main5.tex, subsec:exp-bc, protocol in subsec:exp-protocol.
#
#   dynamics    : dx_i/dt = (1/N) Σ_j wbar_ij(t) φ_ε(x_j - x_i),  wbar = N * W
#   interaction : φ_ε(z) = a · z · (1 - z²/ε²)₊²  — a C^{1,1} compactly
#                 supported mollification of the sharp Hegselmann-Krause
#                 influence z · 1{|z| ≤ ε}. Odd, φ_ε(0) = 0, Lipschitz
#                 constant a, ||φ_ε||_∞ = (16/25√5) aε, support exactly [-ε, ε].
#   initial data: x(0, ξ) = s(ξ - 1/2) + A sin(2πmξ)  (+ jitter during training)
#                 — monotone-dominant in the label (readable kernel heatmaps),
#                 span s ≫ ε so the uncontrolled dynamics fragments into
#                 several frozen opinion clusters.
#   baselines   : (i) mean-field  w ≡ 1  (W_ij = 1/N),
#                 (ii) label-band: mollified bump of range r in |ξ - ζ| on the
#                      interval (no torus wrap; opinions are not periodic) —
#                      the admissible stand-in for the local bounded-confidence
#                      variant. Optional --hk-ref adds the classical sharp
#                      state-dependent HK kernel as a NON-admissible reference.
#   claim       : uncontrolled, the population fragments into clusters with all
#                 inter-cluster gaps > ε; such profiles are stationary for
#                 EVERY admissible kernel (the interaction has exact support ε),
#                 so control must act early — prevention, not reversal.
#
# Outputs (all under out_1/e2_<tag>/, paper-ready PDFs mirrored to
# figures/e2_eps{eps}_T{T}/). All arms are evaluated and archived in
# trajectories.npz/metrics.json; the FIGURES show mean-field vs trained
# control (the paper's two-arm design), unless --uncontrolled-only.
#   config.json, metrics.json, run_summary.txt, model.pt, history.json
#   trajectories.npz     — x(t) per arm, W(t) trained, Var/K/authority series
#   state_evolution.pdf  — x(t, ξ) heatmaps per arm (stripes vs consensus)
#   opinion_traj.pdf     — classical HK "cluster tree": x_i(t) per arm
#   kernel_strip.pdf     — trained kernel heatmaps at 6 times
#   cluster_count.pdf    — number of opinion clusters K(t) per arm
#   variance.pdf         — Var(x(t)) per arm (log scale)
#   authority.pdf        — control authority (1/N) Σ_i max_j |φ_ε(x_j - x_i)|
#   vxi.pdf              — receiver variation V_ξ(ŵ(t)) of the trained kernel
#   phi.pdf              — φ_ε against the sharp HK influence
#   training_history.pdf, opinions.gif, kernel_time.gif
#
# Usage:
#   python e2_bc.py                        # full run (2000 steps)
#   python e2_bc.py --uncontrolled-only    # baselines only: verify fragmentation
#   python e2_bc.py --smoke                # 20-step smoke test
#   python e2_bc.py --device mps           # Apple GPU

import argparse
import dataclasses
import datetime
import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from model_1 import TimeLabelGraphonControl, _random_run_tag
from model_1_viz import save_graphon_gif


# ============================================================
# Configuration (paper protocol; deviations stated in subsec:exp-bc)
# ============================================================

@dataclass
class E2Config:
    # interaction
    eps: float = 1.0           # confidence radius
    gain: float = 4.0          # gain a (Lipschitz constant of φ_ε)
    phi_form: str = "poly"     # "poly" (compact support) | "gauss" (fallback)

    # initial profile x(0, ξ) = span (ξ - 1/2) + amp sin(2π modes ξ)
    span: float = 6.0
    amp: float = 0.5
    modes: int = 4
    jitter: float = 0.1        # i.i.d. jitter on the training initial data

    # label-band baseline
    band_range: float = 0.10

    # discretization
    N: int = 64
    T: float = 10.0
    dt: float = 0.05
    sigma: float = 0.0

    # cost — PAPER convention (eqn:J-cost-1), with β = 1 (stated deviation from
    # the e1 protocol's β = 10: merging clusters needs concentrated attention,
    # and the pilot decides the final value; see --beta)
    beta: float = 1.0
    lambda_terminal: float = 0.0
    gamma_target: float = 100.0
    x_target: float = 0.0

    # optimization
    batch_size: int = 16
    num_steps: int = 2000
    lr: float = 1e-3
    grad_clip: float = 1.0

    # architecture (model_1.py defaults, as in e1)
    num_fourier_freqs: int = 8
    embed_dim: int = 64
    hidden_embed: int = 128
    hidden_score: int = 128

    # bookkeeping
    seed: int = 0
    device: str = "cpu"
    time_chunk: int = 50
    hk_ref: bool = False       # add the sharp state-dependent HK reference arm

    @property
    def NT(self) -> int:
        return int(round(self.T / self.dt))


# ============================================================
# E2 ingredients
# ============================================================

def phi_bc_poly(y: torch.Tensor, eps: float, gain: float) -> torch.Tensor:
    """
    Mollified bounded-confidence interaction
      φ_ε(z) = a z (1 - z²/ε²)₊²,
    C^{1,1}, odd, Lipschitz constant a, support exactly [-ε, ε].
    """
    u = y / eps
    return gain * y * torch.clamp(1.0 - u * u, min=0.0) ** 2


def phi_bc_gauss(y: torch.Tensor, eps: float, gain: float) -> torch.Tensor:
    """Gaussian-tail fallback φ_ε(z) = a z exp(-z²/ε²) (no dead gradients)."""
    return gain * y * torch.exp(-(y / eps) ** 2)


def make_phi(cfg: E2Config) -> Callable[[torch.Tensor], torch.Tensor]:
    base = {"poly": phi_bc_poly, "gauss": phi_bc_gauss}[cfg.phi_form]
    return lambda y: base(y, cfg.eps, cfg.gain)


def bc_profile(xi: torch.Tensor, span: float, amp: float, modes: int) -> torch.Tensor:
    """Initial opinion profile x(0, ξ) = span (ξ - 1/2) + amp sin(2π modes ξ)."""
    return span * (xi - 0.5) + amp * torch.sin(2.0 * math.pi * modes * xi)


def mean_field_kernel(N: int, device, dtype=torch.float32) -> torch.Tensor:
    """Row-stochastic representation of w ≡ 1: W_ij = 1/N."""
    return torch.full((N, N), 1.0 / N, device=device, dtype=dtype)


def band_kernel(xi: torch.Tensor, r: float) -> torch.Tensor:
    """
    Label-band kernel: smooth bump of range r in |ξ - ζ| on the INTERVAL
    (no torus wrap — opinions are not periodic; row normalization absorbs the
    boundary), row-normalized to a row-stochastic W. Smooth in ξ, hence of
    finite receiver variation, unlike the sharp band indicator.
    """
    d = (xi.unsqueeze(0) - xi.unsqueeze(1)).abs()
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
# Rollouts and cost (e1 conventions)
# ============================================================

def precompute_Ws(model, NT: int, dt: float, device, chunk: int) -> torch.Tensor:
    """Returns (NT, N, N); differentiable w.r.t. model parameters."""
    t_vals = torch.arange(NT, dtype=torch.float32, device=device) * dt
    outs = [model(t_vals[a:a + chunk]) for a in range(0, NT, chunk)]
    outs = [o.unsqueeze(0) if o.ndim == 2 else o for o in outs]
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
        diff = x.unsqueeze(1) - x.unsqueeze(2)            # (B, N, N): x_j - x_i
        drift = (Ws[n].unsqueeze(0) * phi(diff)).sum(-1)  # (B, N)
        x = x + dt * drift
        if sigma > 0.0:
            x = x + sigma * sqrt_dt * torch.randn_like(x)
        xs.append(x)
    return torch.stack(xs, dim=0)


def rollout_hk_reference(x0: torch.Tensor, NT: int, dt: float, eps: float) -> torch.Tensor:
    """
    Classical sharp Hegselmann-Krause reference: state-dependent kernel
    W_ij(t) ∝ 1{|x_i - x_j| ≤ ε}, row-normalized, with φ(z) = z within range.
    NOT admissible in our control class (state-dependent, infinite V_ξ);
    reported only as a reference portrait. x0: (B, N) -> (NT+1, B, N).
    """
    x = x0
    xs = [x]
    for _ in range(NT):
        diff = x.unsqueeze(1) - x.unsqueeze(2)            # (B, N, N): x_j - x_i
        adj = (diff.abs() <= eps).to(x.dtype)
        W = adj / adj.sum(dim=-1, keepdim=True)           # diagonal is always in range
        x = x + dt * (W * diff).sum(-1)
        xs.append(x)
    return torch.stack(xs, dim=0)


def e2_cost(xs: torch.Tensor, Ws: torch.Tensor, cfg: E2Config) -> Dict[str, torch.Tensor]:
    """
    Discretized cost J^{N,Δt}, PAPER convention (eqn: JN definition / eqn:J-cost-1):
      Σ_n Δt [ Var(x^n) + β ||w(t_n)||²_{L²(I²)} ] + γ (mean(x^T) - x_target)²
    where ||w||²_{L²(I²)} = Σ_ij W_ij² for the row-stochastic W (wbar = N·W).
    xs: (NT+1, B, N); Ws: (NT, N, N).
    """
    var_t = xs.var(dim=2, unbiased=False).mean(dim=1)
    running_state = cfg.dt * var_t[:-1].sum()
    running_reg = cfg.dt * cfg.beta * (Ws ** 2).sum()
    terminal_var = cfg.lambda_terminal * var_t[-1]
    mean_T = xs[-1].mean(dim=1)
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

def cluster_count(x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Number of opinion clusters: 1 + #{gaps > ε in the sorted state}. A gap
    larger than ε is definitively frozen — the interaction has support ε, so
    no admissible kernel can act across it. x: (..., N) -> (...) int.
    """
    xs, _ = torch.sort(x, dim=-1)
    gaps = xs[..., 1:] - xs[..., :-1]
    return 1 + (gaps > eps).sum(dim=-1)


def cluster_inventory(x: torch.Tensor, eps: float) -> List[Dict[str, float]]:
    """Cluster centers and mass fractions of a single state x: (N,)."""
    xs, _ = torch.sort(x)
    xs = xs.cpu().numpy()
    breaks = np.where(np.diff(xs) > eps)[0]
    out = []
    start = 0
    for b in list(breaks) + [len(xs) - 1]:
        block = xs[start:b + 1]
        out.append({"center": float(block.mean()), "mass": len(block) / len(xs)})
        start = b + 1
    return out


def control_authority(x: torch.Tensor,
                      phi: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """
    Control authority (1/N) Σ_i max_j |φ_ε(x_j - x_i)|: the strongest force any
    admissible kernel can still exert, averaged over receivers. Decays to 0 both
    at fragmentation (window of controllability closed) and at consensus
    (nothing left to do) — the two are distinguished by Var. x: (..., N) -> (...).
    """
    diff = x.unsqueeze(-2) - x.unsqueeze(-1)   # (..., N, N): [i, j] = x_j - x_i
    return phi(diff).abs().amax(dim=-1).mean(dim=-1)


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
              "band": "label-band",
              "trained": "trained control",
              "hk": "sharp HK (reference)"}
ARM_COLORS = {"meanfield": "tab:gray", "band": "tab:orange",
              "trained": "tab:blue", "hk": "tab:green"}


def fig_state_evolution(t_grid, xs_arms: Dict[str, np.ndarray], arms: List[str],
                        cfg: E2Config, path: str):
    lim = cfg.span / 2 + cfg.amp + 0.5
    fig, axes = plt.subplots(1, len(arms), figsize=(4.3 * len(arms), 3.6),
                             sharey=True, squeeze=False)
    for ax, arm in zip(axes[0], arms):
        X = xs_arms[arm]  # (NT+1, N)
        im = ax.imshow(X.T, origin="lower", aspect="auto",
                       extent=(t_grid[0], t_grid[-1], 0.0, 1.0),
                       cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(ARM_LABELS[arm])
        ax.set_xlabel(r"$t$")
    axes[0][0].set_ylabel(r"label $\xi$")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label=r"$x(t,\xi)$")
    fig.suptitle("State evolution", y=1.04)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_opinion_traj(t_grid, xi, xs_arms: Dict[str, np.ndarray], arms: List[str],
                     cfg: E2Config, path: str):
    """The classical HK cluster-tree: opinion trajectories colored by label."""
    colors = plt.cm.viridis(xi)
    fig, axes = plt.subplots(1, len(arms), figsize=(4.3 * len(arms), 3.6),
                             sharey=True, squeeze=False)
    for ax, arm in zip(axes[0], arms):
        X = xs_arms[arm]  # (NT+1, N)
        for i in range(X.shape[1]):
            ax.plot(t_grid, X[:, i], color=colors[i], lw=0.8, alpha=0.8)
        ax.axhline(cfg.x_target, color="0.3", lw=1.0, ls="--")
        ax.set_title(ARM_LABELS[arm])
        ax.set_xlabel(r"$t$")
        ax.grid(alpha=0.15)
    # ε scale bar on the first panel: the interaction's dead-zone width
    ax0 = axes[0][0]
    t_bar = t_grid[0] + 0.05 * (t_grid[-1] - t_grid[0])
    y0 = cfg.span / 2 + cfg.amp - cfg.eps
    ax0.annotate("", xy=(t_bar, y0), xytext=(t_bar, y0 + cfg.eps),
                 arrowprops=dict(arrowstyle="<->", color="0.2", lw=1.2))
    ax0.text(t_bar + 0.02 * (t_grid[-1] - t_grid[0]), y0 + cfg.eps / 2,
             r"$\varepsilon$", va="center", fontsize=11, color="0.2")
    axes[0][0].set_ylabel(r"opinion $x_i(t)$")
    fig.suptitle("Opinion trajectories", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_kernel_strip(Ws: torch.Tensor, dt: float, path: str, n_panels: int = 6,
                     times: List[float] = None):
    """
    Kernel heatmaps on a LOG color scale: the merge-phase attention peaks are
    an order of magnitude above the uniform level wbar = 1, and a linear scale
    would flatten the relax-to-uniform endpoint. Panel times default to a
    uniform grid; pass `times` to concentrate panels in the merge window.
    """
    from matplotlib.colors import LogNorm
    NT = Ws.shape[0]
    if times is None:
        idxs = np.linspace(0, NT - 1, n_panels).round().astype(int)
    else:
        idxs = np.clip(np.round(np.array(times) / dt).astype(int), 0, NT - 1)
        n_panels = len(idxs)
    wbars = [Ws.shape[-1] * Ws[i].cpu().numpy() for i in idxs]
    vmax = max(w.max() for w in wbars)
    vmin = max(min(w.min() for w in wbars), vmax * 1e-3)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.6 * n_panels, 2.9))
    for ax, i, w in zip(axes, idxs, wbars):
        im = ax.imshow(np.maximum(w, vmin), origin="lower", extent=(0, 1, 0, 1),
                       aspect="equal", cmap="viridis", interpolation="bicubic",
                       norm=LogNorm(vmin=vmin, vmax=vmax))
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


def fig_cluster_count(t_grid, K_series: Dict[str, np.ndarray], arms: List[str],
                      path: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for arm in arms:
        ax.step(t_grid, K_series[arm], where="post", color=ARM_COLORS[arm],
                lw=2.0, label=ARM_LABELS[arm])
    ax.axhline(1, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"clusters $K(t)$")
    Kmax = max(int(K_series[a].max()) for a in arms)
    ax.set_yticks(range(1, Kmax + 1))
    ax.set_ylim(0.75, Kmax + 0.25)
    ax.grid(alpha=0.15)
    ax.legend()
    ax.set_title("Opinion cluster count")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_variance(t_grid, var_series: Dict[str, np.ndarray], arms: List[str],
                 path: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for arm in arms:
        ax.semilogy(t_grid, np.maximum(var_series[arm], 1e-12),
                    color=ARM_COLORS[arm], lw=2.0, label=ARM_LABELS[arm])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathrm{Var}(x(t))$")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    ax.set_title("Variance: frozen plateau vs consensus")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_authority(t_grid, auth_series: Dict[str, np.ndarray], arms: List[str],
                  path: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for arm in arms:
        ax.plot(t_grid, auth_series[arm], color=ARM_COLORS[arm], lw=2.0,
                label=ARM_LABELS[arm])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\frac{1}{N}\sum_i \max_j |\varphi_\varepsilon(x_j - x_i)|$")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Control authority (window of controllability)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_vxi(t_grid_W, vxi_trained: np.ndarray, vxi_band: float, path: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(t_grid_W, vxi_trained, color=ARM_COLORS["trained"], lw=2.0,
            label=r"trained $V_\xi(\hat{w}(t))$")
    ax.axhline(0.0, color=ARM_COLORS["meanfield"], lw=1.5, ls="--",
               label=r"mean-field ($V_\xi = 0$)")
    ax.axhline(vxi_band, color=ARM_COLORS["band"], lw=1.5, ls="--",
               label=rf"label-band ($V_\xi = {vxi_band:.1f}$)")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$V_\xi$")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Receiver variation of the kernel (measured, not enforced)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_phi(cfg: E2Config, path: str):
    z = torch.linspace(-2.0 * cfg.eps, 2.0 * cfg.eps, 801)
    phi = make_phi(cfg)
    sharp = torch.where(z.abs() <= cfg.eps, cfg.gain * z, torch.zeros_like(z))
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(z.numpy(), sharp.numpy(), color="0.6", lw=1.5, ls="--",
            label=r"sharp HK  $a\,z\,\mathbf{1}\{|z| \leq \varepsilon\}$")
    ax.plot(z.numpy(), phi(z).numpy(), color="tab:blue", lw=2.2,
            label=r"$\varphi_\varepsilon(z) = a\,z\,(1 - z^2/\varepsilon^2)_+^2$")
    for s in (-1, 1):
        ax.axvline(s * cfg.eps, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\varphi(z)$")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    ax.set_title(rf"Bounded-confidence interaction ($\varepsilon = {cfg.eps:g}$, "
                 rf"$a = {cfg.gain:g}$)")
    fig.tight_layout()
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
        axes[1].semilogy(np.maximum(np.array(history[key]), 1e-10), lw=1.0,
                         label=key, color=color)
    axes[1].set_title("components (log scale)")
    axes[1].set_xlabel("step")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_opinion_gif(t_grid, xi, xs_arms: Dict[str, np.ndarray],
                     K_series: Dict[str, np.ndarray], arms: List[str],
                     cfg: E2Config, path: str,
                     stride: int = 2, fps: int = 12, dpi: int = 80):
    """Animated opinion axis: dot at (x_i(t), ξ_i), colored by label."""
    from PIL import Image
    from io import BytesIO

    colors = plt.cm.viridis(xi)
    lim = cfg.span / 2 + cfg.amp + 0.7
    frames = list(range(0, len(t_grid), stride))
    if frames[-1] != len(t_grid) - 1:
        frames.append(len(t_grid) - 1)

    pil_imgs = []
    for n in frames:
        fig, axes = plt.subplots(1, len(arms), figsize=(3.5 * len(arms), 3.6),
                                 squeeze=False)
        for ax, arm in zip(axes[0], arms):
            x_n = xs_arms[arm][n]  # (N,)
            ax.scatter(x_n, xi, c=colors, s=22, edgecolors="none")
            ax.axvline(cfg.x_target, color="0.8", lw=1.0, ls="--", zorder=0)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel(r"opinion $x$")
            ax.set_title(f"{ARM_LABELS[arm]}\n$K = {int(K_series[arm][n])}$",
                         fontsize=10)
        axes[0][0].set_ylabel(r"label $\xi$")
        fig.suptitle(rf"Bounded confidence:  $t = {t_grid[n]:.2f}$", y=0.99)
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        pil_imgs.append(Image.open(buf).copy())

    pil_imgs[0].save(path, save_all=True, append_images=pil_imgs[1:], loop=0,
                     duration=int(1000 / fps), optimize=False)


# ============================================================
# Evaluation helper (shared by full and uncontrolled-only runs)
# ============================================================

def evaluate_arms(arm_specs: Dict[str, object], x0_eval: torch.Tensor,
                  cfg: E2Config, phi, device):
    """
    arm_specs values: a kernel (ConstantKernel / trained model), or the string
    "hk" for the sharp state-dependent reference. Deterministic, exact datum.
    """
    out = {}
    with torch.no_grad():
        for arm, spec in arm_specs.items():
            if spec == "hk":
                xs_eval = rollout_hk_reference(x0_eval, cfg.NT, cfg.dt, cfg.eps)
                Ws_eval, parts = None, None
            else:
                Ws_eval = precompute_Ws(spec, cfg.NT, cfg.dt, device, cfg.time_chunk)
                xs_eval = rollout_with_Ws(Ws_eval, x0_eval, cfg.dt, phi, sigma=0.0)
                parts = e2_cost(xs_eval, Ws_eval, cfg)
            x_t = xs_eval[:, 0, :]                               # (NT+1, N)
            out[arm] = {
                "X": x_t.cpu().numpy(),
                "Ws": Ws_eval,
                "var": x_t.var(dim=1, unbiased=False).cpu().numpy(),
                "K": cluster_count(x_t, cfg.eps).cpu().numpy(),
                "authority": control_authority(x_t, phi).cpu().numpy(),
                "cost": None if parts is None
                        else {kk: float(vv) for kk, vv in parts.items()},
            }
    return out


def freeze_time(K: np.ndarray, dt: float) -> Optional[float]:
    """First time after which K(t) never changes again (None if K constant)."""
    changes = np.where(K[1:] != K[:-1])[0]
    if len(changes) == 0:
        return None
    return float((changes[-1] + 1) * dt)


# ============================================================
# Main experiment
# ============================================================

def run_e2(cfg: E2Config, tag: str, out_root: str = "out_1", fig_root: str = "figures",
           mirror_figs: bool = True, uncontrolled_only: bool = False):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    dtype = torch.float32

    run_dir = os.path.join(out_root, f"e2_{tag}")
    fig_dir = os.path.join(fig_root, f"e2_eps{cfg.eps:g}_T{cfg.T:g}".replace(".", "p"))
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    xi = (torch.arange(cfg.N, device=device, dtype=dtype) + 0.5) / cfg.N
    base = bc_profile(xi, cfg.span, cfg.amp, cfg.modes)
    NT = cfg.NT
    phi = make_phi(cfg)

    print("=" * 64)
    print(f"E2 bounded confidence | eps={cfg.eps} gain={cfg.gain} "
          f"span={cfg.span} N={cfg.N} T={cfg.T} dt={cfg.dt}")
    print(f"steps={cfg.num_steps} batch={cfg.batch_size} lr={cfg.lr} "
          f"beta={cfg.beta} gamma={cfg.gamma_target} phi={cfg.phi_form}")
    print(f"device={cfg.device} tag={tag}")
    print("=" * 64)

    # ---------- sanity gate: the uncontrolled arm must fragment ----------
    # (the narrative claim of subsec:exp-bc is false for this config otherwise)
    with torch.no_grad():
        W_mf = mean_field_kernel(cfg.N, device)
        Ws_gate = W_mf.unsqueeze(0).expand(NT, -1, -1)
        xs_gate = rollout_with_Ws(Ws_gate, base.unsqueeze(0), cfg.dt, phi)
        K_gate = int(cluster_count(xs_gate[-1, 0], cfg.eps))
    print(f"sanity gate: uncontrolled mean-field arm ends with K = {K_gate} clusters")
    if K_gate < 2 and not uncontrolled_only:
        raise RuntimeError(
            f"uncontrolled arm does not fragment (K={K_gate}); "
            f"raise --span, lower --eps, or raise --gain")

    arm_specs: Dict[str, object] = {
        "meanfield": ConstantKernel(W_mf),
        "band": ConstantKernel(band_kernel(xi, cfg.band_range)),
    }

    # ---------- training ----------
    model = None
    history: Dict[str, List[float]] = {kk: [] for kk in
                                       ["loss", "state_cost", "l2_reg_cost",
                                        "terminal_var", "terminal_target",
                                        "terminal_mean"]}
    if not uncontrolled_only:
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

        pbar = tqdm(range(cfg.num_steps), desc="E2 training", unit="step")
        for _ in pbar:
            model.train()
            optimizer.zero_grad()
            Ws = precompute_Ws(model, NT, cfg.dt, device, cfg.time_chunk)
            x0 = base.unsqueeze(0) + cfg.jitter * torch.randn(
                cfg.batch_size, cfg.N, device=device, dtype=dtype)
            xs = rollout_with_Ws(Ws, x0, cfg.dt, phi, sigma=cfg.sigma)
            parts = e2_cost(xs, Ws, cfg)
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

        model.eval()
        arm_specs["trained"] = model

    if cfg.hk_ref:
        arm_specs["hk"] = "hk"

    # ---------- evaluation: deterministic, exact datum ----------
    x0_eval = base.unsqueeze(0)
    results = evaluate_arms(arm_specs, x0_eval, cfg, phi, device)
    arms = list(arm_specs.keys())
    cost_arms = [a for a in arms if results[a]["cost"] is not None]

    vxi_band = float(vxi_discrete(results["band"]["Ws"][0]))
    vxi_mf = float(vxi_discrete(results["meanfield"]["Ws"][0]))
    vxi_trained = (vxi_discrete(results["trained"]["Ws"]).cpu().numpy()
                   if "trained" in results else None)

    t_grid = np.arange(NT + 1) * cfg.dt
    t_grid_W = np.arange(NT) * cfg.dt

    # ---------- persistence ----------
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    if model is not None:
        torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    npz = {"t": t_grid, "t_W": t_grid_W, "xi": xi.cpu().numpy(),
           "W_band": results["band"]["Ws"][0].cpu().numpy().astype(np.float32)}
    for arm in arms:
        npz[f"x_{arm}"] = results[arm]["X"]
        npz[f"var_{arm}"] = results[arm]["var"]
        npz[f"K_{arm}"] = results[arm]["K"]
        npz[f"authority_{arm}"] = results[arm]["authority"]
    if "trained" in results:
        npz["W_trained"] = results["trained"]["Ws"].cpu().numpy().astype(np.float32)
        npz["vxi_trained"] = vxi_trained
    np.savez_compressed(os.path.join(run_dir, "trajectories.npz"), **npz)

    var_final = {arm: float(results[arm]["var"][-1]) for arm in arms}
    K_final = {arm: int(results[arm]["K"][-1]) for arm in arms}
    metrics = {
        "eps": cfg.eps, "gain": cfg.gain, "span": cfg.span, "beta": cfg.beta,
        "costs": {arm: results[arm]["cost"] for arm in cost_arms},
        "var_final": var_final,
        "K_final": K_final,
        "clusters_meanfield_T": cluster_inventory(
            torch.from_numpy(results["meanfield"]["X"][-1]), cfg.eps),
        "freeze_time_meanfield": freeze_time(results["meanfield"]["K"], cfg.dt),
        "vxi": {"band": vxi_band, "meanfield": vxi_mf},
    }
    if "trained" in results:
        cons = np.where(results["trained"]["var"] < 1e-2)[0]
        metrics["consensus_time_trained"] = (float(cons[0] * cfg.dt)
                                             if len(cons) else None)
        metrics["terminal_mean_trained"] = results["trained"]["cost"]["terminal_mean"]
        metrics["vxi"]["trained_max"] = float(vxi_trained.max())
        metrics["vxi"]["trained_final"] = float(vxi_trained[-1])
        metrics["value_gap_meanfield_minus_trained"] = (
            results["meanfield"]["cost"]["total"] - results["trained"]["cost"]["total"])
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---------- figures ----------
    def both(name: str) -> List[str]:
        if not mirror_figs:
            return [os.path.join(run_dir, name)]
        return [os.path.join(run_dir, name), os.path.join(fig_dir, name)]

    xs_arms_np = {arm: results[arm]["X"] for arm in arms}
    K_series = {arm: results[arm]["K"] for arm in arms}
    var_series = {arm: results[arm]["var"] for arm in arms}
    auth_series = {arm: results[arm]["authority"] for arm in arms}

    # paper figures show mean-field vs trained only; the band arm stays in the
    # npz/metrics archive (uncontrolled-only runs keep whatever arms they have)
    fig_arms = ([a for a in arms if a in ("meanfield", "trained")]
                if "trained" in results else arms)

    for p in both("state_evolution.pdf"):
        fig_state_evolution(t_grid, xs_arms_np, fig_arms, cfg, p)
    for p in both("opinion_traj.pdf"):
        fig_opinion_traj(t_grid, xi.cpu().numpy(), xs_arms_np, fig_arms, cfg, p)
    for p in both("cluster_count.pdf"):
        fig_cluster_count(t_grid, K_series, fig_arms, p)
    for p in both("variance.pdf"):
        fig_variance(t_grid, var_series, fig_arms, p)
    for p in both("authority.pdf"):
        fig_authority(t_grid, auth_series, fig_arms, p)
    for p in both("phi.pdf"):
        fig_phi(cfg, p)
    if "trained" in results:
        strip_times = [0.0, 0.1 * cfg.T, 0.2 * cfg.T, 0.3 * cfg.T, 0.4 * cfg.T, cfg.T]
        for p in both("kernel_strip.pdf"):
            fig_kernel_strip(results["trained"]["Ws"], cfg.dt, p, times=strip_times)
        for p in both("vxi.pdf"):
            fig_vxi(t_grid_W, vxi_trained, vxi_band, p)
        fig_training_history(history, os.path.join(run_dir, "training_history.pdf"))

    # ---------- GIFs ----------
    save_opinion_gif(t_grid, xi.cpu().numpy(), xs_arms_np, K_series, fig_arms, cfg,
                     os.path.join(run_dir, "opinions.gif"))
    if "trained" in results:
        n_gif = min(60, NT)
        gif_idx = np.linspace(0, NT - 1, n_gif).round().astype(int)
        gif_frames = [results["trained"]["Ws"][i].cpu() * cfg.N for i in gif_idx]
        save_graphon_gif(gif_frames, os.path.join(run_dir, "kernel_time.gif"),
                         fps=10, t_vals=[float(i * cfg.dt) for i in gif_idx])

    # ---------- summary ----------
    lines = [
        "=== E2 bounded confidence: run summary ===",
        f"timestamp : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"run_dir   : {run_dir}",
        f"eps = {cfg.eps}, gain = {cfg.gain}, span = {cfg.span}, "
        f"N = {cfg.N}, T = {cfg.T}, dt = {cfg.dt}, steps = {cfg.num_steps}, "
        f"beta = {cfg.beta}",
        "",
        "arm         J_total      Var(T)     K(T)   mean(T)",
    ]
    for arm in arms:
        j = (f"{results[arm]['cost']['total']:>9.4f}"
             if results[arm]["cost"] is not None else "        —")
        mean_T = float(results[arm]["X"][-1].mean())
        lines.append(f"{arm:<11s} {j}  {var_final[arm]:>10.6f}   "
                     f"{K_final[arm]:>3d}   {mean_T:>+8.4f}")
    lines.append("")
    lines.append(f"mean-field clusters at T: "
                 f"{[(round(c['center'], 2), round(c['mass'], 3)) for c in metrics['clusters_meanfield_T']]}")
    if metrics["freeze_time_meanfield"] is not None:
        lines.append(f"mean-field freeze time: t = {metrics['freeze_time_meanfield']:.2f}")
    if "trained" in results:
        lines.append(
            f"V_xi trained: max {metrics['vxi']['trained_max']:.3f}, "
            f"final {metrics['vxi']['trained_final']:.3f} "
            f"(band {vxi_band:.3f}, mean-field {vxi_mf:.3f})")
        lines.append(f"value gap (mean-field - trained): "
                     f"{metrics['value_gap_meanfield_minus_trained']:.4f}")
        if metrics["consensus_time_trained"] is not None:
            lines.append(f"trained consensus time (Var < 1e-2): "
                         f"t = {metrics['consensus_time_trained']:.2f}")
    summary = "\n".join(lines)
    with open(os.path.join(run_dir, "run_summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(summary)
    print(f"\nAll outputs in: {run_dir}")
    if mirror_figs:
        print(f"Paper figures mirrored to: {fig_dir}")
    return metrics


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="E2: bounded-confidence cluster formation and prevention")
    parser.add_argument("--eps", type=float, default=1.0, help="confidence radius")
    parser.add_argument("--gain", type=float, default=4.0, help="interaction gain a")
    parser.add_argument("--span", type=float, default=6.0, help="initial opinion span")
    parser.add_argument("--amp", type=float, default=0.5, help="seed sine amplitude")
    parser.add_argument("--modes", type=int, default=4, help="seed sine mode count")
    parser.add_argument("--phi-form", type=str, default="poly", dest="phi_form",
                        choices=["poly", "gauss"])
    parser.add_argument("--T", type=float, default=10.0, help="time horizon")
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "mps", "cuda"])
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--smoke", action="store_true", help="20-step smoke test")
    parser.add_argument("--no-mirror", action="store_true",
                        help="do not mirror figures into figures/ (pilot runs)")
    parser.add_argument("--uncontrolled-only", action="store_true",
                        dest="uncontrolled_only",
                        help="baselines only: verify fragmentation, no training")
    parser.add_argument("--hk-ref", action="store_true", dest="hk_ref",
                        help="add the sharp state-dependent HK reference arm")
    args = parser.parse_args()

    cfg = E2Config(eps=args.eps, gain=args.gain, span=args.span, amp=args.amp,
                   modes=args.modes, phi_form=args.phi_form, T=args.T, N=args.N,
                   num_steps=args.steps, batch_size=args.batch, beta=args.beta,
                   seed=args.seed, device=args.device, hk_ref=args.hk_ref)
    if args.smoke:
        cfg.num_steps = 20

    tag = args.tag or f"eps{args.eps:g}_{_random_run_tag()}"
    if args.smoke:
        tag = f"smoke_{tag}"
    run_e2(cfg, tag=tag, mirror_figs=not args.no_mirror,
           uncontrolled_only=args.uncontrolled_only)

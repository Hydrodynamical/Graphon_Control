"""Model-specific figures, loaded only when rendering archived runs."""
import math
from typing import Dict, List
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..plotting import imshow_graphon, save_graphon_gif
from .bounded_confidence import E2Config, phi_bc_poly, phi_bc_gauss, make_phi


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

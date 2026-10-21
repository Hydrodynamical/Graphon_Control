"""Model-specific figures, loaded only when rendering archived runs."""
import math
from typing import Dict, List
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..plotting import imshow_graphon, save_graphon_gif


ARM_LABELS = {"meanfield": r"mean-field  $w \equiv 1$",
              "ring": "smooth ring",
              "trained": "trained control"}


ARM_COLORS = {"meanfield": "tab:gray", "ring": "tab:orange", "trained": "tab:blue"}


PAPER_ARMS = ["meanfield", "trained"]


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

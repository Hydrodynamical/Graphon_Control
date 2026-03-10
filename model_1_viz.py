import numpy as np
import torch
import matplotlib.pyplot as plt


def xi_grid(N: int, device=None, dtype=torch.float32):
    """
    Cell centers in [0,1]:
        xi_i = (i + 0.5)/N
    Returns shape (N,).
    """
    dev = device if device is not None else "cpu"
    return (torch.arange(N, device=dev, dtype=dtype) + 0.5) / N


def _as_numpy(A):
    if torch.is_tensor(A):
        return A.detach().cpu().numpy()
    return np.asarray(A)


def _default_vmin_vmax(Wnp, symmetric=False, q=None):
    """
    Choose plotting range.
    - symmetric=True: use [-m,m]
    - q: if not None, use percentile clipping, e.g. q=0.01
    """
    if q is not None:
        lo = np.quantile(Wnp, q)
        hi = np.quantile(Wnp, 1 - q)
    else:
        lo = np.min(Wnp)
        hi = np.max(Wnp)

    if symmetric:
        m = max(abs(lo), abs(hi))
        return -m, m
    return lo, hi


def imshow_graphon(
    W,
    title="",
    vmin=None,
    vmax=None,
    ax=None,
    show=True,
    cmap="viridis",
    interpolation="bicubic",
    colorbar=True,
    symmetric=False,
    percentile_clip=None,
):
    """
    Nicely display W_{ij} as a graphon approximation w(xi_i, zeta_j).

    x-axis = zeta in [0,1]
    y-axis = xi   in [0,1]

    Parameters
    ----------
    interpolation:
        'nearest', 'bilinear', 'bicubic', etc.
        For a smoother graphon-like appearance, 'bicubic' is nice.
    symmetric:
        Use symmetric color scale around 0 (useful if entries can be signed).
    percentile_clip:
        Example: 0.01 clips 1% low/high tails for better contrast.
    """
    Wnp = _as_numpy(W)

    if Wnp.ndim != 2 or Wnp.shape[0] != Wnp.shape[1]:
        raise ValueError(f"W must be square 2D array, got shape {Wnp.shape}")

    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = _default_vmin_vmax(
            Wnp, symmetric=symmetric, q=percentile_clip
        )
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))

    im = ax.imshow(
        Wnp,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        aspect="equal",
        cmap=cmap,
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title if title else r"Graphon heatmap")
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel(r"$\xi$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def contour_graphon(
    W,
    levels=12,
    title="",
    ax=None,
    show=True,
    cmap="viridis",
    colorbar=True,
):
    """
    Contour plot of graphon W.
    """
    Wnp = _as_numpy(W)
    N = Wnp.shape[0]

    zeta = (np.arange(N) + 0.5) / N
    xi = (np.arange(N) + 0.5) / N
    ZZ, XX = np.meshgrid(zeta, xi)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))

    cs = ax.contourf(ZZ, XX, Wnp, levels=levels, cmap=cmap)
    ax.set_title(title if title else r"Graphon contour plot")
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel(r"$\xi$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if colorbar:
        plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_slices_over_zeta(
    W,
    xis=(0.1, 0.3, 0.5, 0.7, 0.9),
    title="",
    ax=None,
    show=True,
    linewidth=2.0,
):
    """
    Fix xi and plot zeta -> w(xi, zeta) for a few xi values.
    """
    Wnp = _as_numpy(W)
    N = Wnp.shape[0]
    zeta = (np.arange(N) + 0.5) / N

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    for xi_val in xis:
        i = int(np.clip(np.floor(xi_val * N), 0, N - 1))
        xi_actual = (i + 0.5) / N
        ax.plot(zeta, Wnp[i, :], linewidth=linewidth, label=rf"$\xi\approx {xi_actual:.2f}$")

    ax.set_title(title if title else r"Slices $\zeta \mapsto w(\xi,\zeta)$")
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel(r"$w(\xi,\zeta)$")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_slices_over_xi(
    W,
    zetas=(0.1, 0.3, 0.5, 0.7, 0.9),
    title="",
    ax=None,
    show=True,
    linewidth=2.0,
):
    """
    Fix zeta and plot xi -> w(xi, zeta) for a few zeta values.
    """
    Wnp = _as_numpy(W)
    N = Wnp.shape[0]
    xi = (np.arange(N) + 0.5) / N

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    for zeta_val in zetas:
        j = int(np.clip(np.floor(zeta_val * N), 0, N - 1))
        zeta_actual = (j + 0.5) / N
        ax.plot(xi, Wnp[:, j], linewidth=linewidth, label=rf"$\zeta\approx {zeta_actual:.2f}$")

    ax.set_title(title if title else r"Slices $\xi \mapsto w(\xi,\zeta)$")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$w(\xi,\zeta)$")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_graphon_diagnostics(
    W,
    title_prefix="",
    cmap="viridis",
    interpolation="bicubic",
    percentile_clip=None,
    symmetric=False,
    show=True,
):
    """
    Three-panel diagnostic:
      1) heatmap
      2) slices in zeta for several xi
      3) row/column masses

    Very useful for seeing whether W is:
      - uniform,
      - banded,
      - blocky,
      - strongly heterogeneous.
    """
    Wnp = _as_numpy(W)
    N = Wnp.shape[0]
    grid = (np.arange(N) + 0.5) / N

    row_mass = Wnp.sum(axis=1)
    col_mass = Wnp.sum(axis=0)

    if percentile_clip is not None:
        vmin, vmax = _default_vmin_vmax(Wnp, symmetric=symmetric, q=percentile_clip)
    else:
        vmin, vmax = _default_vmin_vmax(Wnp, symmetric=symmetric, q=None)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # Panel 1: heatmap
    im = axes[0].imshow(
        Wnp,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        aspect="equal",
        cmap=cmap,
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(f"{title_prefix}heatmap")
    axes[0].set_xlabel(r"$\zeta$")
    axes[0].set_ylabel(r"$\xi$")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: slices
    xis = (0.1, 0.3, 0.5, 0.7, 0.9)
    for xi_val in xis:
        i = int(np.clip(np.floor(xi_val * N), 0, N - 1))
        axes[1].plot(grid, Wnp[i, :], linewidth=2, label=rf"$\xi\approx {(i+0.5)/N:.2f}$")
    axes[1].set_title(f"{title_prefix}slices over $\\zeta$")
    axes[1].set_xlabel(r"$\zeta$")
    axes[1].set_ylabel(r"$w(\xi,\zeta)$")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9)

    # Panel 3: row/column masses
    axes[2].plot(grid, row_mass, linewidth=2, label="row mass")
    axes[2].plot(grid, col_mass, linewidth=2, linestyle="--", label="column mass")
    axes[2].set_title(f"{title_prefix}row/column mass")
    axes[2].set_xlabel(r"$\xi$ or $\zeta$")
    axes[2].set_ylabel("mass")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    plt.tight_layout()
    if show:
        plt.show()

    return axes


def heterogeneity_stats(W):
    """
    Scalar summaries for how structured W is.
    """
    Wt = W if torch.is_tensor(W) else torch.tensor(W, dtype=torch.float32)
    eps = 1e-12

    P = (Wt / (Wt.sum(dim=1, keepdim=True).clamp_min(eps))).clamp_min(eps)
    row_entropy = float((-P * P.log()).sum(dim=1).mean().item())

    N = Wt.shape[0]
    k = max(1, int(0.05 * N))
    topk_mass = float(torch.topk(Wt, k=k, dim=1).values.sum(dim=1).mean().item())

    uniform = torch.full_like(Wt, 1.0 / max(N - 1, 1))
    uniform.fill_diagonal_(0.0)
    row_l2_dev = float(((Wt - uniform).pow(2).sum(dim=1).mean()).sqrt().item())

    return {
        "mean_row_entropy": row_entropy,
        "mean_topk_mass_(k=5%N)": topk_mass,
        "mean_row_L2_dev_from_uniform": row_l2_dev,
        "min": float(Wt.min().item()),
        "max": float(Wt.max().item()),
        "mean": float(Wt.mean().item()),
    }


def frob_norm(W):
    Wnp = _as_numpy(W)
    return float(np.linalg.norm(Wnp.reshape(-1), ord=2))


def dist_frob(Wa, Wb):
    A = _as_numpy(Wa)
    B = _as_numpy(Wb)
    return float(np.linalg.norm((A - B).reshape(-1), ord=2))


@torch.no_grad()
def sample_graphon_on_fine_grid(model_or_callable, t=None, x=None, M=200, device="cpu"):
    """
    Sample either:
      - a callable w(t,x,xi,zeta), or
      - a callable w(xi,zeta),
    on a fine M x M grid, for a smoother display.

    This is useful if your model can evaluate directly at arbitrary (xi,zeta),
    rather than only returning an N x N matrix.
    """
    grid = (torch.arange(M, device=device, dtype=torch.float32) + 0.5) / M
    XI, ZETA = torch.meshgrid(grid, grid, indexing="ij")

    try:
        W = model_or_callable(t, x, XI, ZETA)
    except TypeError:
        W = model_or_callable(XI, ZETA)

    return W.detach().cpu()


@torch.no_grad()
def plot_state_histogram_over_time_imshow(
    xs,
    dt: float,
    bins=80,
    value_range=None,
    density=True,
    cmap="viridis",
    title="Empirical state histogram over time",
    show=True,
):
    X = torch.stack(xs, dim=0).detach().cpu().numpy()   # (Tsteps, B, N)
    Tsteps = X.shape[0]

    flat_all = X.reshape(-1)
    if value_range is None:
        xmin = float(flat_all.min())
        xmax = float(flat_all.max())
        pad = 0.05 * max(1e-8, xmax - xmin)
        xmin -= pad
        xmax += pad
        value_range = (xmin, xmax)

    x_edges = np.linspace(value_range[0], value_range[1], bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    H = np.zeros((Tsteps, bins), dtype=float)
    for n in range(Tsteps):
        vals = X[n].reshape(-1)
        hist, _ = np.histogram(vals, bins=x_edges, density=density)
        H[n, :] = hist

    _, ax = plt.subplots(1, 1, figsize=(8, 5))
    im = ax.imshow(
        H,
        origin="lower",
        aspect="auto",
        extent=(x_centers[0], x_centers[-1], 0.0, dt * (Tsteps - 1)),
        cmap=cmap,
    )
    ax.set_xlabel("state x")
    ax.set_ylabel("time t")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="density" if density else "count")

    if show:
        plt.show()

    return H, x_edges, ax
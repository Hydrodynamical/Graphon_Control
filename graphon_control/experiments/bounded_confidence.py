"""Bounded-confidence opinions experiment using the shared paper conventions.

See docs/NUMERICS.md for equations and docs/DEVELOPMENT.md for reproducible runs.
Model-specific rendering is in bounded_confidence_plots.py.
"""
import argparse
import datetime
import json
import math
import os
from dataclasses import dataclass, replace
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
    evaluation_seed: int = 1729
    score_bound: Optional[float] = None
    seed: int = 0
    device: str = "cpu"
    time_chunk: int = 50
    uncontrolled_only: bool = False
    hk_ref: bool = False       # add the sharp state-dependent HK reference arm

    def __post_init__(self):
        validate_common_config(self)
        if not isinstance(self.uncontrolled_only, bool) or not isinstance(self.hk_ref, bool):
            raise ValueError("uncontrolled_only and hk_ref must be booleans")
        for name in ("eps", "gain", "span", "band_range"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.phi_form not in ("poly", "gauss"):
            raise ValueError("phi_form must be 'poly' or 'gauss'")
        if not math.isfinite(self.amp) or isinstance(self.modes, bool) or not isinstance(self.modes, int) or self.modes < 1:
            raise ValueError("amp must be finite and modes a positive integer")

    @property
    def NT(self) -> int:
        return n_time_steps(self.T, self.dt)


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


def e2_cost(xs, Ws, cfg: E2Config):
    """Paper objective, equation (60); shared across both experiments."""
    return paper_cost(xs, Ws, cfg)


def cluster_count(x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Number of opinion clusters: 1 + #{gaps > ε in the sorted state}. A gap
    larger than ε separates components for the compact-support interaction.
    For Gaussian tails this is only a descriptive gap threshold.
    x: (..., N) -> (...) int.
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

def run_e2(cfg: E2Config, tag: str = "", out_root: str = "out_1", fig_root: str = "figures",
           mirror_figs: bool = False, uncontrolled_only: Optional[bool] = None, make_plots: bool = True):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    dtype = torch.float32

    uncontrolled_only = cfg.uncontrolled_only if uncontrolled_only is None else uncontrolled_only
    cfg = replace(cfg, uncontrolled_only=uncontrolled_only)
    cfg.__post_init__()
    run_dir = str(create_run_directory(out_root, "e2", tag or None))
    fig_dir = os.path.join(fig_root, os.path.basename(run_dir))
    if mirror_figs and make_plots and Path(fig_dir).exists():
        raise FileExistsError(f"Figure export already exists: {fig_dir}")
    write_run_provenance(run_dir, cfg, cfg.seed, cfg.evaluation_seed, device=cfg.device)

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

    # ---------- validate the paper-specific fragmentation scenario ----------
    # (the narrative claim of subsec:exp-bc is false for this config otherwise)
    with torch.no_grad():
        W_mf = mean_field_kernel(cfg.N, device)
        Ws_gate = W_mf.unsqueeze(0).expand(NT, -1, -1)
        xs_gate = rollout_with_Ws(Ws_gate, base.unsqueeze(0), cfg.dt, phi)
        K_gate = int(cluster_count(xs_gate[-1, 0], cfg.eps))
    print(f"sanity gate: uncontrolled mean-field arm ends with K = {K_gate} clusters")
    if K_gate < 2 and not uncontrolled_only and cfg.phi_form == "poly":
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
            score_bound=cfg.score_bound,
        ).to(device)
        def objective():
            Ws = precompute_Ws(model, NT, cfg.dt, device, cfg.time_chunk)
            x0 = base.unsqueeze(0) + cfg.jitter * torch.randn(
                cfg.batch_size, cfg.N, device=device, dtype=dtype)
            xs = rollout_with_Ws(Ws, x0, cfg.dt, phi, sigma=cfg.sigma)
            return e2_cost(xs, Ws, cfg)

        history, _ = train_controller(model, objective, cfg.num_steps, cfg.lr,
                                      cfg.grad_clip, description="E2 training")

        model.eval()
        arm_specs["trained"] = model

    if cfg.hk_ref:
        arm_specs["hk"] = "hk"

    torch.manual_seed(cfg.evaluation_seed)

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
    if model is not None:
        save_checkpoint(os.path.join(run_dir, "model.pt"), model)
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

    if make_plots:
        plot_run(run_dir, fig_dir=fig_dir if mirror_figs else None)

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
    from .bounded_confidence_plots import (fig_state_evolution, fig_opinion_traj, fig_kernel_strip, fig_cluster_count, fig_variance, fig_authority, fig_vxi, fig_phi, fig_training_history, save_opinion_gif, ARM_LABELS, ARM_COLORS)
    from ..plotting import save_graphon_gif
    cfg = E2Config(**json.loads((Path(run_dir) / "config.json").read_text()))
    history_path = Path(run_dir) / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else {}
    metrics = json.loads((Path(run_dir) / "metrics.json").read_text())
    with np.load(Path(run_dir) / "trajectories.npz", allow_pickle=False) as archive:
        data = dict(archive)
    NT = cfg.NT
    t_grid, t_grid_W = data["t"], data["t_W"]
    xi = torch.as_tensor(data["xi"])
    arms = list(metrics["var_final"])
    results = {arm: {"X": data[f"x_{arm}"], "K": data[f"K_{arm}"],
                     "var": data[f"var_{arm}"], "authority": data[f"authority_{arm}"]}
               for arm in arms}
    if "trained" in results:
        results["trained"]["Ws"] = torch.as_tensor(data["W_trained"])
    vxi_trained = data.get("vxi_trained")
    vxi_band = metrics["vxi"]["band"]
    if fig_dir is not None:
        Path(fig_dir).mkdir(parents=True, exist_ok=False)
    # ---------- figures ----------
    def both(name: str) -> List[str]:
        if fig_dir is None:
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


    complete = Path(run_dir) / "complete.json"
    if complete.exists():
        payload = json.loads(complete.read_text())
        payload["plots_generated"] = True
        complete.write_text(json.dumps(payload) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bounded-confidence opinion control")
    add_common_arguments(parser)
    for name in ("eps", "gain", "span", "amp"):
        parser.add_argument(f"--{name}", type=float)
    parser.add_argument("--modes", type=int)
    parser.add_argument("--phi-form", choices=["poly", "gauss"], dest="phi_form")
    parser.add_argument("--uncontrolled-only", action="store_true", default=None)
    parser.add_argument("--hk-ref", action="store_true", default=None, dest="hk_ref")
    args = parser.parse_args(argv)
    try:
        if args.replot:
            plot_run(args.replot)
            return
        cfg = configuration_from_args(E2Config, args)
        run_e2(cfg, tag=args.tag, out_root=args.out_root, fig_root=args.fig_root,
               mirror_figs=args.mirror_figures, make_plots=not args.no_plots, uncontrolled_only=args.uncontrolled_only)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

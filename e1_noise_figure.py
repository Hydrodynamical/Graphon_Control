# e1_noise_figure.py
#
# Robustness panel for the noisy Kuramoto runs (main5.tex, subsec:exp-kuramoto):
# trained control evaluated on deterministic dynamics (dashed) against the
# noisy-ensemble statistics saved by e1_kuramoto.py for sigma > 0 runs
# (mean over 32 Brownian replicates, shaded 10-90% band), for both the order
# parameter r(t) and the winding number W(t).
#
# Usage:
#   python e1_noise_figure.py --noisy k2_T12_sig01 [--noiseless k2_T12] [--out ...]

import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="E1 noise robustness panel")
    parser.add_argument("--noisy", type=str, default="k2_T12_sig01",
                        help="tag of the sigma > 0 run (out_1/e1_<tag>)")
    parser.add_argument("--noiseless", type=str, default="k2_T12",
                        help="tag of the sigma = 0 reference run")
    parser.add_argument("--out", type=str, default="",
                        help="output path (default: run dir + figure mirror)")
    args = parser.parse_args()

    noisy_dir = os.path.join("out_1", f"e1_{args.noisy}")
    clean_dir = os.path.join("out_1", f"e1_{args.noiseless}")
    with open(os.path.join(noisy_dir, "config.json")) as f:
        cfg = json.load(f)
    sigma, k, T = cfg["sigma"], cfg["k"], cfg["T"]
    if sigma <= 0:
        raise SystemExit(f"run {args.noisy} has sigma = {sigma}; need a noisy run")

    noisy = np.load(os.path.join(noisy_dir, "trajectories.npz"))
    clean = np.load(os.path.join(clean_dir, "trajectories.npz"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    panels = [("r", r"order parameter $r(t)$", (-0.03, 1.03)),
              ("winding", r"winding number $\mathcal{W}(t)$", (-0.25, k + 0.25))]
    for ax, (key, ylabel, ylim) in zip(axes, panels):
        ax.fill_between(noisy["t"], noisy[f"{key}_noisy_q10"],
                        noisy[f"{key}_noisy_q90"], color="tab:blue", alpha=0.25,
                        label="noisy ensemble, 10–90%")
        ax.plot(noisy["t"], noisy[f"{key}_noisy_mean"], color="tab:blue", lw=2.0,
                label=rf"noisy ensemble mean ($\sigma = {sigma:g}$)")
        ax.plot(clean["t"], clean[f"{key}_trained"], color="0.25", lw=1.8, ls="--",
                label=r"deterministic ($\sigma = 0$)")
        if key == "winding":
            for q in range(k + 1):
                ax.axhline(q, color="0.85", lw=0.8, zorder=0)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle(rf"Trained control under noise ($k = {k}$, $T = {T:g}$)", y=1.02)
    fig.tight_layout()

    if args.out:
        paths = [args.out]
    else:
        suffix = f"_sigma{sigma:g}".replace(".", "p")
        fig_dir = os.path.join("figures", f"e1_k{k}_T{T:g}{suffix}")
        os.makedirs(fig_dir, exist_ok=True)
        paths = [os.path.join(noisy_dir, "r_noise_panel.pdf"),
                 os.path.join(fig_dir, "r_noise_panel.pdf")]
    for p in paths:
        fig.savefig(p, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()

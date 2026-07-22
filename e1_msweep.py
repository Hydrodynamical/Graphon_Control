# e1_msweep.py
#
# M-sweep of experiment E1: the numerical portrait of thm:main's OUTER limit
# and of cor:value-stability (main5.tex, §5.3).
#
# Fix N = 64 and sweep the DATUM resolution: initial datum P_M(k=1 twist) for
# M in {2, 4, 8, 16, 32, 64}, three seeds each. cor:value-stability predicts
#   |A_inf(x_M(0)) - A_inf(x(0))| <= C ||P_M x0 - x0||_{L2} = C * 2pi/(sqrt(12) M),
# a genuine 1/M convergence curve. Closed-form anchor at M = 2: the two blocks
# sit at +-pi/2, every pairwise difference lies in {0, pi} where sin vanishes,
# so the drift is identically zero FOR EVERY admissible kernel: the datum is
# uncontrollable and
#   A(M=2) = Var * T + beta * ||w=1||^2 * T = (pi^2/4) * 5 + 10 * 5 = 62.337.
# (The optimizer should recover this by driving the kernel to uniform.)
#
# Reused cells (identical problems already trained):
#   M=8  s0  <- out_1/e1_sweep_N64_s0   (N-sweep run, x0_level=8)
#   M=8  s1  <- out_1/e1_sweep_N64_s1
#   M=64 s0  <- out_1/e1_k1_T5          (P_64 twist == smooth twist on the N=64 grid)
#
# Usage:
#   .venv312/bin/python e1_msweep.py                 # run + aggregate
#   .venv312/bin/python e1_msweep.py --aggregate-only

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

MS = [64, 32, 16, 8, 4, 2]
SEEDS = [0, 1, 2]
N = 64
T = 5.0
BETA = 10.0
LANES = 3
PY = sys.executable

REUSE = {
    (8, 0): "out_1/e1_sweep_N64_s0",
    (8, 1): "out_1/e1_sweep_N64_s1",
    (64, 0): "out_1/e1_k1_T5",
}

J_M2_EXACT = (math.pi ** 2 / 4.0) * T + BETA * 1.0 * T  # uncontrollable anchor


def cell_dir(M, s):
    return os.path.join("out_1", f"e1_msweep_M{M}_s{s}")


def done(M, s):
    return os.path.exists(os.path.join(cell_dir(M, s), "metrics.json"))


def stage_reuse():
    for (M, s), src in REUSE.items():
        if done(M, s):
            continue
        if os.path.exists(os.path.join(src, "metrics.json")):
            shutil.copytree(src, cell_dir(M, s))
            print(f"reused M={M} s={s} <- {src}")


def launch_all(steps: int):
    jobs = [(M, s) for s in SEEDS for M in MS if not done(M, s)]
    print(f"{len(jobs)} runs to do")
    active = []
    while jobs or active:
        while jobs and len(active) < LANES:
            M, s = jobs.pop(0)
            log = open(os.path.join("out_1", f"e1_msweep_M{M}_s{s}.log"), "w")
            p = subprocess.Popen(
                [PY, "e1_kuramoto.py", "--k", "1", "--T", str(T), "--N", str(N),
                 "--seed", str(s), "--x0-level", str(M), "--steps", str(steps),
                 "--tag", f"msweep_M{M}_s{s}", "--no-mirror"],
                stdout=log, stderr=subprocess.STDOUT)
            active.append((p, M, s, log))
            print(f"launched M={M} seed={s} (pid {p.pid})", flush=True)
        for t in list(active):
            if t[0].poll() is not None:
                active.remove(t)
                t[3].close()
                status = "ok" if t[0].returncode == 0 else f"FAILED rc={t[0].returncode}"
                print(f"finished M={t[1]} seed={t[2]}: {status}", flush=True)
        time.sleep(10)


def aggregate():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for M in sorted(MS):
        for s in SEEDS:
            d = cell_dir(M, s)
            if not os.path.exists(os.path.join(d, "metrics.json")):
                print(f"missing: M={M} seed={s}")
                continue
            with open(os.path.join(d, "metrics.json")) as f:
                m = json.load(f)
            rows.append({"M": M, "seed": s,
                         "J_trained": m["costs"]["trained"]["total"],
                         "J_meanfield": m["costs"]["meanfield"]["total"],
                         "r_final": m["r_final"]["trained"],
                         "var_final": m["var_final"]["trained"],
                         "vxi_max": m["vxi"]["trained_max"]})
    os.makedirs("out_1/e1_msweep", exist_ok=True)
    with open("out_1/e1_msweep/msweep_summary.json", "w") as f:
        json.dump({"rows": rows, "J_M2_exact": J_M2_EXACT}, f, indent=2)

    Ms = sorted({r["M"] for r in rows})
    jmin = {M: min(r["J_trained"] for r in rows if r["M"] == M) for M in Ms}
    jlo = {M: min(r["J_trained"] for r in rows if r["M"] == M) for M in Ms}
    jhi = {M: max(r["J_trained"] for r in rows if r["M"] == M) for M in Ms}
    jmf = {M: np.median([r["J_meanfield"] for r in rows if r["M"] == M]) for M in Ms}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    ax.fill_between(Ms, [jlo[M] for M in Ms], [jhi[M] for M in Ms],
                    color="tab:blue", alpha=0.18, label="seed range")
    ax.plot(Ms, [jmin[M] for M in Ms], "o-", color="tab:blue", lw=2,
            label=r"trained $\widehat{J}(M)$ (min over seeds)")
    ax.plot(Ms, [jmf[M] for M in Ms], "s--", color="tab:gray", lw=1.5,
            label=r"mean-field baseline")
    ax.plot([2], [J_M2_EXACT], marker="*", ms=16, color="tab:red", ls="none",
            label=r"$M=2$: exact (uncontrollable)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ms); ax.set_xticklabels([str(m) for m in Ms])
    ax.set_xlabel(r"datum resolution $M$"); ax.set_ylabel(r"$\widehat{J}$")
    ax.set_title(r"Values vs.\ datum resolution ($N = 64$)")
    ax.grid(alpha=0.25); ax.legend(fontsize=9)

    ax = axes[1]
    Mref = max(Ms)
    Mplot = [M for M in Ms if M != Mref]
    gaps = [abs(jmin[M] - jmin[Mref]) for M in Mplot]
    ax.loglog(Mplot, gaps, "o-", color="tab:blue", lw=2,
              label=r"$|\widehat{J}(M) - \widehat{J}(%d)|$" % Mref)
    guide1 = [gaps[1] * Mplot[1] / M for M in Mplot]
    guide2 = [gaps[1] * (Mplot[1] / M) ** 2 for M in Mplot]
    ax.loglog(Mplot, guide1, "--", color="0.6",
              label=r"$1/M$ (bound of \Cref{cor:value-stability})"
              if False else r"$1/M$ (corollary bound)")
    ax.loglog(Mplot, guide2, ":", color="0.4", label=r"$1/M^2$ (observed)")
    ax.set_xticks(Mplot); ax.set_xticklabels([str(m) for m in Mplot])
    ax.set_xlabel(r"$M$"); ax.set_ylabel("value gap")
    ax.set_title(r"Convergence in the datum resolution")
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=9)
    fig.tight_layout()
    for outp in ["out_1/e1_msweep/convergence_M.pdf",
                 "figures/e1_msweep/convergence_M.pdf"]:
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        fig.savefig(outp, bbox_inches="tight")
    plt.close(fig)

    print(f"aggregated {len(rows)} cells -> convergence_M.pdf")
    for M in Ms:
        rs = [r for r in rows if r["M"] == M]
        print(f"  M={M:3d}  J min {jmin[M]:8.4f}  range [{jlo[M]:.4f}, {jhi[M]:.4f}]  "
              f"baseline {jmf[M]:8.4f}  r(T) max {max(r['r_final'] for r in rs):.4f}")
    print(f"  M=2 exact anchor: {J_M2_EXACT:.4f}")
    print("MSWEEP_COMPLETE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    if not args.aggregate_only:
        stage_reuse()
        launch_all(steps=args.steps)
    aggregate()

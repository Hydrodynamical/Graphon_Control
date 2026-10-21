"""Run and summarize the Kuramoto initial-datum resolution study.

Each cell fixes N=64, k=1, T=5 and changes the block-averaged initial datum P_M x0.
The plotted values are optimizer outputs, not certified value functions or a
proof of the manuscript's continuum limit. At M=2 the sine drift vanishes for
all kernels, giving the exact uniform-kernel anchor recorded in the summary.

Existing cells are reused only after validating config, provenance, and metrics.
Incomplete cells are never overwritten: select a new --sweep-tag to retry.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from graphon_control.artifacts import (
    SCHEMA_VERSION, atomic_write_json, create_run_directory, generated_run_tag,
)


MS = [64, 32, 16, 8, 4, 2]
SEEDS = [0, 1, 2]
N = 64
T = 5.0
BETA = 10.0
LANES = 3
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = REPO_ROOT / "out_1"
J_M2_EXACT = (math.pi ** 2 / 4.0) * T + BETA * T


def cell_tag(M, seed, sweep_tag="msweep"):
    letters_and_digits = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    if (not sweep_tag or sweep_tag[0] not in letters_and_digits
            or any(c not in letters_and_digits + "_.-" for c in sweep_tag)):
        raise ValueError("sweep_tag must be a nonempty safe path component")
    return f"{sweep_tag}_M{M}_s{seed}"


def cell_dir(M, seed, out_root=DEFAULT_OUT_ROOT, sweep_tag="msweep"):
    return Path(out_root) / f"e1_{cell_tag(M, seed, sweep_tag)}"


def _expected_config(M, seed, steps):
    return {"N": N, "k": 1, "T": T, "dt": 0.05, "sigma": 0.0,
            "beta": BETA, "x0_level": M, "seed": seed, "num_steps": steps}


def read_cell(M, seed, *, steps=2000, out_root=DEFAULT_OUT_ROOT, sweep_tag="msweep"):
    """Return verified metrics, None if absent, or fail on an incompatible cell."""
    directory = cell_dir(M, seed, out_root, sweep_tag)
    if not directory.exists():
        return None
    if not directory.is_dir():
        raise ValueError(f"Cell path is not a directory: {directory}")
    try:
        with (directory / "config.json").open(encoding="utf-8") as stream:
            config = json.load(stream)
        with (directory / "provenance.json").open(encoding="utf-8") as stream:
            provenance = json.load(stream)
        with (directory / "metrics.json").open(encoding="utf-8") as stream:
            metrics = json.load(stream)
        with (directory / "complete.json").open(encoding="utf-8") as stream:
            completion = json.load(stream)
        if not isinstance(completion, dict) or completion.get("status") != "complete":
            raise ValueError("run has no successful completion marker")
        if not (directory / "model.pt").is_file():
            raise ValueError("missing model.pt")
        for key, value in _expected_config(M, seed, steps).items():
            if config.get(key) != value:
                raise ValueError(f"config {key}: expected {value!r}, got {config.get(key)!r}")
        encoded = json.dumps(config, sort_keys=True, allow_nan=False).encode("utf-8")
        if provenance.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("missing or unsupported provenance schema")
        if provenance.get("training_seed") != seed:
            raise ValueError("provenance training seed mismatch")
        if provenance.get("config_sha256") != hashlib.sha256(encoded).hexdigest():
            raise ValueError("configuration does not match saved provenance")
        values = [metrics["costs"]["trained"]["total"], metrics["costs"]["meanfield"]["total"],
                  metrics["r_final"]["trained"], metrics["var_final"]["trained"],
                  metrics["vxi"]["trained_max"]]
        if not all(np.isfinite(v) for v in values):
            raise ValueError("nonfinite metrics")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Incomplete or incompatible sweep cell {directory}: {exc}. "
            "Use a new --sweep-tag instead of overwriting it."
        ) from exc
    return metrics


def done(M, seed, *, steps=2000, out_root=DEFAULT_OUT_ROOT, sweep_tag="msweep"):
    return read_cell(M, seed, steps=steps, out_root=out_root, sweep_tag=sweep_tag) is not None


def launch_all(steps, *, out_root=DEFAULT_OUT_ROOT, lanes=LANES, sweep_tag="msweep", device="cpu"):
    if steps < 1 or lanes < 1:
        raise ValueError("steps and lanes must be positive")
    root = Path(out_root).expanduser().resolve()
    # Preflight existing cells before starting any subprocess.
    jobs = [(M, seed) for seed in SEEDS for M in MS
            if not done(M, seed, steps=steps, out_root=root, sweep_tag=sweep_tag)]
    root.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} runs to do", flush=True)
    active = []
    try:
        while jobs or active:
            while jobs and len(active) < lanes:
                M, seed = jobs.pop(0)
                log_path = root / f"e1_{cell_tag(M, seed, sweep_tag)}.log"
                log = log_path.open("x", encoding="utf-8")
                command = [sys.executable, str(REPO_ROOT / "e1_kuramoto.py"),
                           "--k", "1", "--T", str(T), "--N", str(N), "--seed", str(seed),
                           "--x0-level", str(M), "--steps", str(steps), "--device", device,
                           "--out-root", str(root), "--tag", cell_tag(M, seed, sweep_tag),
                           "--no-mirror"]
                try:
                    process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                active.append((process, M, seed, log, log_path))
                print(f"launched M={M} seed={seed} (pid {process.pid})", flush=True)
            for job in list(active):
                process, M, seed, log, log_path = job
                if process.poll() is None:
                    continue
                active.remove(job)
                log.close()
                if process.returncode:
                    raise RuntimeError(f"M={M} seed={seed} failed ({process.returncode}); see {log_path}")
                if not done(M, seed, steps=steps, out_root=root, sweep_tag=sweep_tag):
                    raise RuntimeError(f"M={M} seed={seed} exited without a complete run; see {log_path}")
                print(f"finished M={M} seed={seed}", flush=True)
            if active:
                time.sleep(1)
    finally:
        for process, _, _, log, _ in active:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()


def aggregate(*, steps=2000, out_root=DEFAULT_OUT_ROOT, sweep_tag="msweep", allow_partial=False):
    rows, missing = [], []
    for M in sorted(MS):
        for seed in SEEDS:
            metrics = read_cell(M, seed, steps=steps, out_root=out_root, sweep_tag=sweep_tag)
            if metrics is None:
                missing.append({"M": M, "seed": seed})
                continue
            rows.append({"M": M, "seed": seed,
                         "J_trained": metrics["costs"]["trained"]["total"],
                         "J_meanfield": metrics["costs"]["meanfield"]["total"],
                         "r_final": metrics["r_final"]["trained"],
                         "var_final": metrics["var_final"]["trained"],
                         "vxi_max": metrics["vxi"]["trained_max"]})
    if not rows:
        raise ValueError("No complete sweep cells are available; no summary was written")
    if missing and not allow_partial:
        raise ValueError(f"Sweep is incomplete ({len(missing)} missing cells); use --allow-partial to summarize explicitly")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = sorted({r["M"] for r in rows})
    jmin = {M: min(r["J_trained"] for r in rows if r["M"] == M) for M in levels}
    jhi = {M: max(r["J_trained"] for r in rows if r["M"] == M) for M in levels}
    jmf = {M: np.median([r["J_meanfield"] for r in rows if r["M"] == M]) for M in levels}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    ax.fill_between(levels, [jmin[M] for M in levels], [jhi[M] for M in levels],
                    color="tab:blue", alpha=0.18, label="available seed range")
    ax.plot(levels, [jmin[M] for M in levels], "o-", color="tab:blue", lw=2,
            label=r"trained $\widehat{J}(M)$ (min over available seeds)")
    ax.plot(levels, [jmf[M] for M in levels], "s--", color="tab:gray", label="mean-field baseline")
    ax.plot([2], [J_M2_EXACT], "*", ms=16, color="tab:red", label=r"$M=2$: exact")
    ax.set_xscale("log", base=2)
    ax.set_xticks(levels, [str(M) for M in levels])
    ax.set(xlabel=r"datum resolution $M$", ylabel=r"$\widehat{J}$",
           title=f"Optimizer values vs. datum resolution (N={N})")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax = axes[1]
    reference = max(levels)
    plotted = [M for M in levels if M != reference]
    gaps = [abs(jmin[M] - jmin[reference]) for M in plotted]
    if plotted:
        ax.loglog(plotted, gaps, "o-", color="tab:blue", lw=2,
                  label=rf"$|\widehat{{J}}(M)-\widehat{{J}}({reference})|$")
        positive = [(M, gap) for M, gap in zip(plotted, gaps) if gap > 0]
        if positive:
            anchor_M, anchor_gap = positive[0]
            ax.loglog(plotted, [anchor_gap * anchor_M / M for M in plotted], "--",
                      color="0.6", label=r"$1/M$ reference slope")
        else:
            ax.text(0.5, 0.5, "All observed gaps are zero", transform=ax.transAxes, ha="center")
        ax.set_xticks(plotted, [str(M) for M in plotted])
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Need at least two resolutions for gaps", transform=ax.transAxes, ha="center")
    ax.set(xlabel=r"$M$", ylabel="optimizer value gap", title="Exploratory resolution comparison")
    ax.grid(alpha=0.25, which="both")
    if missing:
        fig.suptitle(f"PARTIAL SWEEP: {len(rows)} of {len(MS) * len(SEEDS)} cells")
    fig.tight_layout()
    output = create_run_directory(out_root, "e1", f"{sweep_tag}_summary_{generated_run_tag()}")
    try:
        fig.savefig(output / "convergence_M.pdf", bbox_inches="tight")
    finally:
        plt.close(fig)
    atomic_write_json(output / "msweep_summary.json", {
        "rows": rows, "J_M2_exact": J_M2_EXACT, "complete": not missing,
        "missing": missing, "steps": steps, "sweep_tag": sweep_tag,
        "interpretation": "Optimizer upper-bound estimates; no certified convergence rate.",
    })
    print(f"aggregated {len(rows)} cells -> {output}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--sweep-tag", default="msweep")
    parser.add_argument("--lanes", type=int, default=LANES)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    try:
        if not args.aggregate_only:
            launch_all(args.steps, out_root=args.out_root, lanes=args.lanes,
                       sweep_tag=args.sweep_tag, device=args.device)
        aggregate(steps=args.steps, out_root=args.out_root, sweep_tag=args.sweep_tag,
                  allow_partial=args.allow_partial)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()

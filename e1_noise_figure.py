"""Compare compatible, separately trained clean and noisy Kuramoto runs.

Use --noisy and --noiseless with run directories, or with legacy run tags under
--out-root. No shared manuscript figure directory is inferred or modified.
"""

import argparse
import json
from pathlib import Path

import numpy as np


_REQUIRED_COMPARISON_FIELDS = ("N", "k", "T", "dt")
_OPTIONAL_COMPARISON_FIELDS = (
    "x0_level", "beta", "lambda_terminal", "gamma_target", "x_target", "jitter",
    "ring_range", "batch_size", "num_steps", "lr", "grad_clip",
    "num_fourier_freqs", "embed_dim", "hidden_embed", "hidden_score", "score_bound",
)


def validate_run_configs(noisy_config, clean_config):
    """Refuse comparisons with different problems, grids, or training settings."""
    for field in (*_REQUIRED_COMPARISON_FIELDS, "sigma"):
        if field not in noisy_config or field not in clean_config:
            raise ValueError(f"Both run configurations must record {field!r}")
    noisy_sigma = float(noisy_config["sigma"])
    clean_sigma = float(clean_config["sigma"])
    if not np.isfinite(noisy_sigma) or noisy_sigma <= 0:
        raise ValueError("The noisy run must have finite sigma > 0")
    if clean_sigma != 0:
        raise ValueError("The noiseless reference must have sigma = 0")
    for field in (*_REQUIRED_COMPARISON_FIELDS, *_OPTIONAL_COMPARISON_FIELDS):
        if field in _REQUIRED_COMPARISON_FIELDS or field in noisy_config or field in clean_config:
            if noisy_config.get(field) != clean_config.get(field):
                raise ValueError(
                    f"Incompatible {field}: noisy={noisy_config.get(field)!r}, "
                    f"noiseless={clean_config.get(field)!r}"
                )


def resolve_run_directory(value, out_root):
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.name != str(path) or path.name in {".", ".."}:
        raise FileNotFoundError(f"Run directory does not exist: {path}")
    tagged = Path(out_root).expanduser() / f"e1_{value}"
    if not tagged.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {tagged}")
    return tagged.resolve()


def load_comparison(noisy_dir, clean_dir):
    """Load and check figure inputs before creating any output."""
    configs = []
    for directory in (noisy_dir, clean_dir):
        with (Path(directory) / "config.json").open(encoding="utf-8") as stream:
            configs.append(json.load(stream))
    noisy_config, clean_config = configs
    validate_run_configs(noisy_config, clean_config)
    with np.load(Path(noisy_dir) / "trajectories.npz", allow_pickle=False) as archive:
        noisy = {key: archive[key] for key in archive.files}
    with np.load(Path(clean_dir) / "trajectories.npz", allow_pickle=False) as archive:
        clean = {key: archive[key] for key in archive.files}
    for name, data in (("noisy", noisy), ("noiseless", clean)):
        if "t" not in data or data["t"].ndim != 1 or len(data["t"]) < 2:
            raise ValueError(f"Invalid time grid in {name} trajectories")
        if not np.isfinite(data["t"]).all() or not np.all(np.diff(data["t"]) > 0):
            raise ValueError(f"Invalid time grid in {name} trajectories")
        for key in ("r", "winding"):
            suffixes = ("noisy_mean", "noisy_q10", "noisy_q90") if name == "noisy" else ("trained",)
            for suffix in suffixes:
                series = data.get(f"{key}_{suffix}")
                if series is None or series.shape != data["t"].shape or not np.isfinite(series).all():
                    raise ValueError(f"Missing or invalid {key}_{suffix} in {name} trajectories")
            if name == "noisy" and np.any(data[f"{key}_noisy_q10"] > data[f"{key}_noisy_q90"]):
                raise ValueError(f"Invalid {key} quantile band in noisy trajectories")
    if noisy["t"].shape != clean["t"].shape or not np.allclose(noisy["t"], clean["t"]):
        raise ValueError("Saved trajectory time grids differ")
    expected_t = np.arange(len(noisy["t"])) * noisy_config["dt"]
    if not np.allclose(noisy["t"], expected_t) or not np.isclose(expected_t[-1], noisy_config["T"]):
        raise ValueError("Trajectory time grid does not match config T and dt")
    return noisy_config, noisy, clean


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noisy", required=True, help="sigma > 0 run directory or tag")
    parser.add_argument("--noiseless", required=True, help="sigma = 0 run directory or tag")
    parser.add_argument("--out-root", type=Path, default=Path(__file__).resolve().parent / "out_1")
    parser.add_argument("--out", type=Path, help="output PDF (default: noisy run/r_noise_panel.pdf)")
    parser.add_argument("--overwrite", action="store_true", help="explicitly replace an existing panel")
    args = parser.parse_args(argv)
    try:
        noisy_dir = resolve_run_directory(args.noisy, args.out_root)
        clean_dir = resolve_run_directory(args.noiseless, args.out_root)
        config, noisy, clean = load_comparison(noisy_dir, clean_dir)
        output = args.out.expanduser() if args.out else noisy_dir / "r_noise_panel.pdf"
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Pass --overwrite to replace it.")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigma, k, T = config["sigma"], config["k"], config["T"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    panels = [("r", r"order parameter $r(t)$", (-0.03, 1.03)),
              ("winding", r"winding number $\mathcal{W}(t)$", (-0.25, k + 0.25))]
    for ax, (key, ylabel, ylim) in zip(axes, panels):
        ax.fill_between(noisy["t"], noisy[f"{key}_noisy_q10"],
                        noisy[f"{key}_noisy_q90"], color="tab:blue", alpha=0.25,
                        label="noisy ensemble, 10–90%")
        ax.plot(noisy["t"], noisy[f"{key}_noisy_mean"], color="tab:blue", lw=2,
                label=rf"noise-trained control ($\sigma = {sigma:g}$)")
        ax.plot(clean["t"], clean[f"{key}_trained"], color="0.25", lw=1.8, ls="--",
                label=r"separately trained reference ($\sigma = 0$)")
        if key == "winding":
            for q in range(k + 1):
                ax.axhline(q, color="0.85", lw=0.8, zorder=0)
        ax.set(xlabel=r"$t$", ylabel=ylabel, ylim=ylim)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(rf"Separate training with and without noise ($k = {k}$, $T = {T:g}$)", y=1.02)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

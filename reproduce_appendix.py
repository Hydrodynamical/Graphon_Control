"""Export the existing appendix from a hash-pinned archive, without simulation.

The manifest records the original five runs. Only JSON and NPZ contents are
interpreted; optional checkpoint, history, and summary files are hashed as bytes.
This command never deserializes a controller or executes training or dynamics.
"""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

RUN_NAMES = ("bc", "k1", "k2", "k3", "noise")
INPUT_FILES = ("config.json", "metrics.json", "trajectories.npz")
ARMS = ("meanfield", "trained")
RTOL, ATOL = 2e-5, 1e-6  # archived computations used float32


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_manifest(manifest_path):
    """Capture and hash every input before parsing arrays or creating outputs."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    if manifest.get("schema_version") != 1 or set(manifest.get("runs", {})) != set(RUN_NAMES):
        raise ValueError("Manifest must have schema_version 1 and runs bc, k1, k2, k3, noise")
    captured, inputs = {}, {}
    for name in RUN_NAMES:
        spec = manifest["runs"][name]
        directory = Path(spec["path"]).expanduser()
        if not directory.is_absolute():
            directory = manifest_path.parent / directory
        directory = directory.resolve()
        if not set(INPUT_FILES).issubset(spec.get("sha256", {})):
            raise ValueError(f"{name}: manifest must pin all three input files")
        captured[name], inputs[name] = {}, {}
        for filename in spec["sha256"]:
            if Path(filename).name != filename:
                raise ValueError(f"{name}: hash keys must be plain filenames")
            path = directory / filename
            raw = path.read_bytes()
            digest = sha256(raw)
            if digest != spec["sha256"][filename]:
                raise ValueError(f"SHA-256 mismatch: {name}/{filename}")
            if filename in INPUT_FILES:
                captured[name][filename] = raw
            inputs[name][filename] = {"path": str(path), "sha256": digest, "bytes": len(raw)}
    runs = {}
    for name in RUN_NAMES:
        payload = captured[name]
        with np.load(io.BytesIO(payload["trajectories.npz"]), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        runs[name] = {"config": json.loads(payload["config.json"]),
                      "metrics": json.loads(payload["metrics.json"]), "arrays": arrays}
    return runs, {"manifest": {"path": str(manifest_path), "sha256": sha256(raw_manifest)},
                  "inputs": inputs}


def _require_config(name, config):
    expected = {"N": 64, "dt": .05, "lambda_terminal": 0., "gamma_target": 100.,
                "x_target": 0., "jitter": .1, "batch_size": 16, "lr": .001,
                "num_fourier_freqs": 8, "embed_dim": 64, "hidden_embed": 128,
                "hidden_score": 128, "num_steps": 3000 if name == "k3" else 2000}
    if name == "bc":
        expected.update(T=10., sigma=0., beta=1., eps=1., gain=4., phi_form="poly",
                        span=6., amp=.5, modes=4)
    else:
        k = 2 if name == "noise" else int(name[1:])
        expected.update(k=k, T={1: 5., 2: 12., 3: 20.}[k], beta=10.,
                        sigma=.3 if name == "noise" else 0.)
    for field, value in expected.items():
        if field not in config or isinstance(config[field], bool) or config[field] != value:
            raise ValueError(f"{name}: original scientific setting {field} must be recorded as {value!r}")
    if config.get("score_bound") is not None or config.get("uncontrolled_only", False):
        raise ValueError(f"{name}: configuration differs from the archived original protocol")


def _array(data, key, shape, name):
    if key not in data:
        raise ValueError(f"{name}: missing saved array {key}")
    value = np.asarray(data[key])
    if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{name}: invalid saved array {key}; expected finite numeric {shape}")
    return value.astype(np.float64)


def _check(name, field, actual, recorded, checks):
    actual = np.asarray(actual)
    recorded = np.asarray(recorded)
    if actual.shape != recorded.shape or not np.isfinite(recorded).all():
        raise ValueError(f"{name}: invalid recorded metric {field}")
    error = float(np.max(np.abs(actual - recorded)))
    passed = bool(np.allclose(actual, recorded, rtol=RTOL, atol=ATOL))
    checks.append({"field": field, "passed": passed, "max_absolute_error": error})
    if not passed:
        raise ValueError(f"{name}: saved arrays disagree with {field} (max error {error:g})")


def winding(x):
    gaps = np.roll(x, -1, axis=-1) - x
    return (((gaps + np.pi) % (2 * np.pi)) - np.pi).sum(axis=-1) / (2 * np.pi)


def diagnostics(name, run):
    """Recompute objective and observables directly from the archived arrays."""
    c, m, d = run["config"], run["metrics"], run["arrays"]
    _require_config(name, c)
    nt, n, dt = round(c["T"] / c["dt"]), c["N"], c["dt"]
    checks, report = [], {}
    t = _array(d, "t", (nt + 1,), name)
    if not np.allclose(t, np.arange(nt + 1) * dt, rtol=0, atol=1e-10):
        raise ValueError(f"{name}: time grid does not match recorded T and dt")
    tw = _array(d, "t_W", (nt,), name)
    if not np.allclose(tw, t[:-1], rtol=0, atol=1e-10):
        raise ValueError(f"{name}: kernel time grid does not match saved states")
    xi = _array(d, "xi", (n,), name)
    if not np.allclose(xi, (np.arange(n) + .5) / n, rtol=0, atol=1e-7):
        raise ValueError(f"{name}: expected original midpoint label grid")
    ws = _array(d, "W_trained", (nt, n, n), name)
    if np.min(ws) < -1e-7 or not np.allclose(ws.sum(-1), 1, rtol=0, atol=1e-5):
        raise ValueError(f"{name}: archived trained kernels are not row stochastic")
    vxi = np.sqrt(np.mean((n * np.diff(ws, axis=1)) ** 2, axis=-1)).sum(-1)
    _check(name, "vxi_trained", vxi, _array(d, "vxi_trained", (nt,), name), checks)
    _check(name, "vxi.trained_max", vxi.max(), m["vxi"]["trained_max"], checks)
    _check(name, "vxi.trained_final", vxi[-1], m["vxi"]["trained_final"], checks)
    for arm in ARMS:
        x = _array(d, "x_" + arm, (nt + 1, n), name)
        expected_initial = (c["span"] * (xi - .5) + c["amp"] * np.sin(2 * np.pi * c["modes"] * xi)
                            if name == "bc" else 2 * np.pi * c["k"] * (xi - .5))
        if not np.allclose(x[0], expected_initial, rtol=1e-6, atol=1e-6):
            raise ValueError(f"{name}: archived initial state does not match the appendix profile")
        means = x.mean(-1)
        var = ((x - means[:, None]) ** 2).mean(-1)
        _check(name, "var_" + arm, var, _array(d, "var_" + arm, (nt + 1,), name), checks)
        _check(name, "var_final." + arm, var[-1], m["var_final"][arm], checks)
        state = float(dt * var[:-1].sum())
        l2 = float(dt * c["beta"] * np.square(ws).sum()) if arm == "trained" else float(c["beta"] * c["T"])
        target = float(c["gamma_target"] * (means[-1] - c["x_target"]) ** 2)
        values = {"total": state + l2 + c["lambda_terminal"] * var[-1] + target,
                  "state_cost": state, "l2_reg_cost": l2, "terminal_var": float(var[-1]),
                  "terminal_target": target, "terminal_mean": float(means[-1])}
        for field, value in values.items():
            _check(name, f"costs.{arm}.{field}", value, m["costs"][arm][field], checks)
        result = {**values, "max_vxi": float(vxi.max()) if arm == "trained" else 0.}
        if name != "bc":
            r, w = np.abs(np.exp(1j * x).mean(-1)), winding(x)
            _check(name, "r_" + arm, r, _array(d, "r_" + arm, (nt + 1,), name), checks)
            _check(name, "winding_" + arm, w, _array(d, "winding_" + arm, (nt + 1,), name), checks)
            _check(name, "r_final." + arm, r[-1], m["r_final"][arm], checks)
            if "winding_final" in m:
                _check(name, "winding_final." + arm, w[-1], m["winding_final"][arm], checks)
            result.update(r_final=float(r[-1]), winding_final=float(w[-1]))
        else:
            clusters = 1 + int(np.count_nonzero(np.diff(np.sort(x[-1])) > c["eps"]))
            saved_clusters = _array(d, "K_" + arm, (nt + 1,), name)
            _check(name, "K_" + arm + "[-1]", clusters, saved_clusters[-1], checks)
            if "K_final" in m:
                _check(name, "K_final." + arm, clusters, m["K_final"][arm], checks)
            result["clusters_final"] = clusters
        report[arm] = result
    if name == "noise":
        for key in ("r", "winding"):
            for suffix in ("mean", "q10", "q90"):
                a = _array(d, f"{key}_noisy_{suffix}", (nt + 1,), name)
                if key == "r" and (a.min() < -1e-6 or a.max() > 1 + 1e-6):
                    raise ValueError("noise: order parameter must lie in [0, 1]")
            if np.any(d[f"{key}_noisy_q10"] > d[f"{key}_noisy_q90"]):
                raise ValueError(f"noise: inverted {key} quantile band")
        for metric, array in (("r_final_mean", "r_noisy_mean"), ("r_final_q10", "r_noisy_q10"),
                              ("r_final_q90", "r_noisy_q90"), ("winding_final_mean", "winding_noisy_mean")):
            _check(name, "noisy_eval." + metric, float(d[array][-1]), m["noisy_eval"][metric], checks)
        if m["noisy_eval"]["sigma"] != .3 or m["noisy_eval"]["n_replicates"] != 32:
            raise ValueError("noise: expected recorded sigma=.3 and 32 replicates")
    expected_j = {"bc": (38.08, 17.93), "k1": (66.45, 57.52), "k2": (277.88, 176.09),
                  "k3": (792.03, 368.62)}
    rounding = {}
    if name in expected_j:
        for arm, expected in zip(ARMS, expected_j[name]):
            rounding["J_" + arm] = {"expected": expected, "rounded_recomputed": round(report[arm]["total"], 2),
                                     "matches": bool(round(report[arm]["total"], 2) == expected)}
    rounding_matches = all(check["matches"] for check in rounding.values())
    return {"checks": checks, "recomputed": report, "saved_array_checks_passed": True,
            "passed": rounding_matches, "appendix_rounding": rounding,
            "status": "verified" if rounding_matches else "appendix_rounding_mismatch"}


def _noise_figure(noisy, clean, path):
    # Same saved aggregate series and layout as e1_noise_figure; no file reloads.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d, ref = noisy["arrays"], clean["arrays"]
    c = noisy["config"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    try:
        for ax, (key, ylabel, ylim) in zip(axes, [
            ("r", r"order parameter $r(t)$", (-.03, 1.03)),
            ("winding", r"winding number $\mathcal{W}(t)$", (-.25, c["k"] + .25))]):
            ax.fill_between(d["t"], d[f"{key}_noisy_q10"], d[f"{key}_noisy_q90"],
                            color="tab:blue", alpha=.25, label="noisy ensemble, 10–90%")
            ax.plot(d["t"], d[f"{key}_noisy_mean"], color="tab:blue", lw=2,
                    label=rf"noisy ensemble mean ($\sigma = {c['sigma']:g}$)")
            ax.plot(ref["t"], ref[f"{key}_trained"], color=".25", lw=1.8, ls="--",
                    label=r"deterministic ($\sigma = 0$)")
            if key == "winding":
                for q in range(c["k"] + 1):
                    ax.axhline(q, color=".85", lw=.8, zorder=0)
            ax.set(xlabel=r"$t$", ylabel=ylabel, ylim=ylim)
            ax.grid(alpha=.2)
        axes[0].legend(loc="lower right", fontsize=9)
        fig.suptitle(rf"Trained control under noise ($k = {c['k']}$, $T = {c['T']:g}$)", y=1.02)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
    finally:
        plt.close(fig)


def export_appendix(manifest_path, output):
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output directory already exists: {output}")
    runs, provenance = load_manifest(manifest_path)
    verification = {name: diagnostics(name, run) for name, run in runs.items()}
    # Compare recorded data directly: historical configs did not all record x0_level.
    for arm in ARMS:
        if not np.array_equal(runs["noise"]["arrays"]["x_" + arm][0],
                              runs["k2"]["arrays"]["x_" + arm][0]):
            raise ValueError("Noise and clean reference have different archived initial data")
    noise_summary = {
        **runs["noise"]["metrics"]["noisy_eval"],
        "source": "archived ensemble mean and 10th/90th percentiles",
        "individual_noisy_paths_saved": False,
        "all_paths_zero_terminal_winding_verified": False,
        "limitation": "Individual path arrays are absent; aggregate winding statistics cannot verify every path.",
    }
    from graphon_control.experiments.bounded_confidence import E2Config
    from graphon_control.experiments import bounded_confidence_plots as bc_plots
    from graphon_control.experiments import kuramoto_plots as kur_plots
    import torch
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".appendix-stage-", dir=output.parent))
    try:
        bc, k3 = runs["bc"]["arrays"], runs["k3"]["arrays"]
        cfg = E2Config(**runs["bc"]["config"])
        bc_plots.fig_opinion_traj(bc["t"], bc["xi"], {a: bc["x_" + a] for a in ARMS},
                                 list(ARMS), cfg, str(stage / "bc_opinion_traj.pdf"))
        bc_plots.fig_kernel_strip(torch.from_numpy(bc["W_trained"]), cfg.dt,
                                 str(stage / "bc_kernel_strip.pdf"), times=[0, .1*cfg.T, .2*cfg.T, .3*cfg.T, .4*cfg.T, cfg.T])
        kur_plots.fig_rw_panel(k3["t"], {a: k3["r_" + a] for a in ARMS},
                               {a: k3["winding_" + a] for a in ARMS}, 3, str(stage / "kur_rw_panel.pdf"))
        kur_plots.fig_kernel_strip(torch.from_numpy(k3["W_trained"]), runs["k3"]["config"]["dt"],
                                   str(stage / "kur_kernel_strip.pdf"))
        _noise_figure(runs["noise"], runs["k2"], stage / "kur_noise_panel.pdf")
        rows = []
        for name in ("bc", "k1", "k2", "k3"):
            for arm in ARMS:
                d = verification[name]["recomputed"][arm]
                rows.append({"run": name, "arm": arm, "J": d["total"], "terminal_variance": d["terminal_var"],
                             "terminal_mean": d["terminal_mean"], "max_Vxi": d["max_vxi"],
                             "r_final": d.get("r_final", ""), "winding_final": d.get("winding_final", ""),
                             "clusters_final": d.get("clusters_final", "")})
        with (stage / "tables.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _json(stage / "noise_summary.json", noise_summary)
        _json(stage / "verification.json", {"operation": "saved_array_verification", "new_experiments": False,
              "float32_comparison_tolerance": {"rtol": RTOL, "atol": ATOL}, "runs": verification,
              "noise_individual_paths": noise_summary["limitation"]})
        provenance.update(operation="saved_array_appendix_export", new_experiments=False,
                          checkpoint_loads=False, outputs={p.name: {"sha256": sha256(p.read_bytes()), "bytes": p.stat().st_size}
                                                           for p in sorted(stage.iterdir())})
        _json(stage / "provenance.json", provenance)
        # mkdir is exclusive, including when another writer races after validation.
        output.mkdir()
        published = []
        try:
            for path in stage.iterdir():
                target = output / path.name
                os.link(path, target)
                published.append(target)
        except BaseException:
            for path in published:
                path.unlink()
            try:
                output.rmdir()
            except OSError:
                pass  # preserve unrelated concurrent contents
            raise
    finally:
        shutil.rmtree(stage)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new directory; existing outputs are never replaced")
    args = parser.parse_args(argv)
    try:
        output = export_appendix(args.manifest, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Exported saved appendix data to {output}")


if __name__ == "__main__":
    main()

"""Export a combined Kuramoto panel from an explicitly selected saved run.

Only config.json and trajectories.npz are read. This command neither identifies
a run as a manuscript result nor reconstructs missing numerical data.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from graphon_control.experiments.kuramoto_plots import fig_rw_panel


ARMS = ("meanfield", "trained")


def _vector(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite vector")
    return value


def export_panel(run_dir, output):
    """Write a new PDF and export sidecar, preserving the selected source run."""
    run_dir = Path(run_dir).expanduser().resolve()
    inputs = {"config": run_dir / "config.json",
              "trajectories": run_dir / "trajectories.npz"}
    output = Path(output).expanduser().absolute()
    sidecar = output.with_suffix(output.suffix + ".provenance.json")
    for destination in (output, sidecar):
        if destination.resolve() in {source.resolve() for source in inputs.values()}:
            raise ValueError(f"Output collides with a source artifact: {destination}")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Output already exists: {destination}")
    if output.suffix.lower() != ".pdf":
        raise ValueError("--output must name a new .pdf file")

    # Read once: hashes describe the same bytes used to load the figure inputs.
    payloads = {name: source.read_bytes() for name, source in inputs.items()}
    config = json.loads(payloads["config"])
    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object")
    missing = {"k", "N", "T", "dt"} - set(config)
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")
    k, N = config["k"], config["N"]
    if (isinstance(k, bool) or not isinstance(k, int) or k < 1
            or isinstance(N, bool) or not isinstance(N, int) or N <= 2 * k):
        raise ValueError("Configuration must have integer k >= 1 and N > 2*k")
    for name in ("T", "dt"):
        value = config[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"Configuration {name} must be finite and positive")
    T, dt = config["T"], config["dt"]
    steps = T / dt
    if not np.isfinite(steps) or not np.isclose(steps, round(steps), rtol=1e-10, atol=1e-10):
        raise ValueError("Configuration T/dt must be an integer")
    NT = round(steps)
    keys = {"t", "xi", *(f"{name}_{arm}" for name in ("r", "winding") for arm in ARMS)}
    with np.load(io.BytesIO(payloads["trajectories"]), allow_pickle=False) as archive:
        missing = keys - set(archive.files)
        if missing:
            raise ValueError(f"Missing saved arrays: {sorted(missing)}")
        data = {key: _vector(archive[key], key) for key in keys}
    t, xi = data["t"], data["xi"]
    if (t.shape != (NT + 1,) or len(t) < 2 or not np.all(np.diff(t) > 0)
            or not np.allclose(t, np.arange(NT + 1) * dt, rtol=1e-7, atol=1e-8)):
        raise ValueError("Saved t grid does not match configuration T and dt")
    if (xi.shape != (N,) or not np.all(np.diff(xi) > 0)
            or np.any((xi < 0) | (xi > 1))):
        raise ValueError("Saved xi grid does not match configuration N on [0, 1]")
    for name in ("r", "winding"):
        for arm in ARMS:
            values = data[f"{name}_{arm}"]
            if values.shape != t.shape:
                raise ValueError(f"Saved {name}_{arm} does not match the time grid")
            if name == "r" and np.any((values < -1e-6) | (values > 1 + 1e-6)):
                raise ValueError(f"Saved r_{arm} must lie in [0, 1]")
            if name == "winding" and not np.isclose(values[0], k, rtol=0, atol=1e-5):
                raise ValueError(f"Saved winding_{arm} does not start at configuration k")

    provenance = {
        "schema_version": 1,
        "operation": "saved_array_export",
        "inputs": {name: {"path": str(inputs[name]),
                          "sha256": hashlib.sha256(payload).hexdigest()}
                   for name, payload in payloads.items()},
        "selection": {"arms": list(ARMS), "k": k, "N": N, "T": T, "dt": dt},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".appendix-panel-", dir=output.parent) as stage:
        staged_pdf = Path(stage) / output.name
        fig_rw_panel(t, {arm: data[f"r_{arm}"] for arm in ARMS},
                     {arm: data[f"winding_{arm}"] for arm in ARMS}, k, staged_pdf)
        provenance["output"] = {"path": str(output),
                                "sha256": hashlib.sha256(staged_pdf.read_bytes()).hexdigest()}
        staged_sidecar = Path(stage) / sidecar.name
        staged_sidecar.write_text(json.dumps(provenance, indent=2, sort_keys=True,
                                            allow_nan=False) + "\n", encoding="utf-8")
        # Exclusive publication also refuses destinations created after preflight.
        os.link(staged_pdf, output)
        try:
            os.link(staged_sidecar, sidecar)
        except OSError:
            if output.exists() and output.samefile(staged_pdf):
                output.unlink()
            raise
    return output, sidecar


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="existing archived run directory")
    parser.add_argument("--output", type=Path, required=True, help="new PDF path")
    args = parser.parse_args(argv)
    try:
        paths = export_panel(args.run, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

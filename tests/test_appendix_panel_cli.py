"""Saved-array CLI checks; all numerical values below are handwritten fixtures."""

import hashlib
import json

import numpy as np
import pytest
import torch

import export_appendix_panel as exporter
from graphon_control import artifacts, dynamics, training


def _write_run(directory, *, missing_config=None, missing_array=None, update=None):
    directory.mkdir()
    config = {"k": 1, "N": 8, "T": 1.0, "dt": 0.5}
    data = {"t": np.array([0.0, 0.5, 1.0]), "xi": (np.arange(8) + 0.5) / 8,
            "r_meanfield": np.zeros(3), "r_trained": np.array([0.0, 0.6, 1.0]),
            "winding_meanfield": np.ones(3), "winding_trained": np.array([1.0, 1.0, 0.0]),
            "r_ring": np.zeros(3), "winding_ring": np.ones(3)}
    if missing_config:
        del config[missing_config]
    if missing_array:
        del data[missing_array]
    if update:
        update(config, data)
    (directory / "config.json").write_text(json.dumps(config))
    np.savez(directory / "trajectories.npz", **data)
    return directory


def _hashes(directory):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.iterdir() if path.is_file()}


@pytest.fixture(autouse=True)
def forbid_experiment_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Saved-array export must not train, evaluate dynamics, or load a checkpoint")

    monkeypatch.setattr(training, "train_controller", forbidden)
    monkeypatch.setattr(dynamics, "rollout_with_Ws", forbidden)
    monkeypatch.setattr(dynamics, "rollout_dynamics_w1", forbidden)
    monkeypatch.setattr(dynamics, "precompute_Ws", forbidden)
    monkeypatch.setattr(artifacts, "load_controller", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)


def test_cli_exports_selected_saved_curves_and_new_operation_provenance(tmp_path, monkeypatch):
    run = _write_run(tmp_path / "original-run")
    before = _hashes(run)
    output = tmp_path / "exports" / "fixture-panel.pdf"
    selected = []
    original_plot = exporter.fig_rw_panel

    def plot(t, r_series, winding_series, k, path):
        selected.append((set(r_series), set(winding_series), k))
        return original_plot(t, r_series, winding_series, k, path)

    monkeypatch.setattr(exporter, "fig_rw_panel", plot)
    exporter.main(["--run", str(run), "--output", str(output)])
    sidecar = output.with_suffix(".pdf.provenance.json")
    record = json.loads(sidecar.read_text())
    assert selected == [({"meanfield", "trained"}, {"meanfield", "trained"}, 1)]
    assert output.read_bytes().startswith(b"%PDF-")
    assert record["operation"] == "saved_array_export"
    assert record["inputs"]["config"]["sha256"] == before["config.json"]
    assert record["inputs"]["trajectories"]["sha256"] == before["trajectories.npz"]
    assert record["output"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert record["output"]["path"] == str(output)
    assert record["selection"] == {"arms": ["meanfield", "trained"],
                                    "k": 1, "N": 8, "T": 1.0, "dt": 0.5}
    assert _hashes(run) == before
    assert set(_hashes(output.parent)) == {output.name, sidecar.name}


@pytest.mark.parametrize("key", ["k", "N", "T", "dt"])
def test_missing_config_fields_never_default_or_write_outputs(tmp_path, key):
    run = _write_run(tmp_path / "run", missing_config=key)
    before = _hashes(run)
    output = tmp_path / "exports" / "panel.pdf"
    with pytest.raises(ValueError, match="Missing configuration fields"):
        exporter.export_panel(run, output)
    assert not output.parent.exists()
    assert _hashes(run) == before


@pytest.mark.parametrize("key", ["t", "xi", "r_meanfield", "r_trained",
                                 "winding_meanfield", "winding_trained"])
def test_missing_saved_arrays_never_reconstruct_or_write_outputs(tmp_path, key):
    run = _write_run(tmp_path / "run", missing_array=key)
    before = _hashes(run)
    output = tmp_path / "exports" / "panel.pdf"
    with pytest.raises(ValueError, match="Missing saved arrays"):
        exporter.export_panel(run, output)
    assert not output.parent.exists()
    assert _hashes(run) == before


@pytest.mark.parametrize("case", ["invalid_k", "invalid_N", "invalid_dt", "nonintegral_T",
                                  "wrong_N", "wrong_t", "wrong_shape", "nonfinite_curve",
                                  "wrong_initial_winding", "invalid_order_parameter", "object_array"])
def test_config_and_saved_array_mismatches_fail_before_writes(tmp_path, case):
    def update(config, data):
        if case == "invalid_k":
            config["k"] = True
        elif case == "invalid_N":
            config["N"] = 2
        elif case == "invalid_dt":
            config["dt"] = 0.0
        elif case == "nonintegral_T":
            config["T"] = 1.1
        elif case == "wrong_N":
            config["N"] = 9
        elif case == "wrong_t":
            data["t"][1] = 0.4
        elif case == "wrong_shape":
            data["r_trained"] = np.ones(2)
        elif case == "nonfinite_curve":
            data["r_trained"][1] = np.nan
        elif case == "wrong_initial_winding":
            data["winding_trained"][0] = 0.0
        elif case == "invalid_order_parameter":
            data["r_trained"][1] = 1.1
        elif case == "object_array":
            data["r_trained"] = np.array([0.0, 0.6, 1.0], dtype=object)

    run = _write_run(tmp_path / "run", update=update)
    before = _hashes(run)
    output = tmp_path / "exports" / "panel.pdf"
    with pytest.raises(ValueError):
        exporter.export_panel(run, output)
    assert not output.parent.exists()
    assert _hashes(run) == before


@pytest.mark.parametrize("existing", ["pdf", "sidecar", "pdf_symlink", "sidecar_symlink"])
def test_existing_outputs_and_dangling_symlinks_are_preserved(tmp_path, existing):
    run = _write_run(tmp_path / "run")
    output = tmp_path / "panel.pdf"
    sidecar = output.with_suffix(".pdf.provenance.json")
    occupied = sidecar if existing.startswith("sidecar") else output
    if existing.endswith("symlink"):
        occupied.symlink_to(tmp_path / "missing-target")
    else:
        occupied.write_bytes(b"preserve existing output")
    with pytest.raises(FileExistsError):
        exporter.export_panel(run, output)
    if existing.endswith("symlink"):
        assert occupied.is_symlink()
        assert not (tmp_path / "missing-target").exists()
    else:
        assert occupied.read_bytes() == b"preserve existing output"
    other = output if occupied == sidecar else sidecar
    assert not other.exists()


@pytest.mark.parametrize("source", ["config.json", "trajectories.npz"])
def test_source_output_collisions_are_rejected(tmp_path, source):
    run = _write_run(tmp_path / "run")
    before = _hashes(run)
    with pytest.raises(ValueError, match="collides with a source artifact"):
        exporter.export_panel(run, run / source)
    assert _hashes(run) == before


def test_cli_reports_missing_arrays_without_creating_files(tmp_path):
    run = _write_run(tmp_path / "run", missing_array="r_trained")
    output = tmp_path / "exports" / "panel.pdf"
    with pytest.raises(SystemExit) as error:
        exporter.main(["--run", str(run), "--output", str(output)])
    assert error.value.code == 1
    assert not output.parent.exists()

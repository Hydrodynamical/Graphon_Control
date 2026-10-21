"""Artifact safety and postprocessing checks; no optimization is run here."""

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from graphon_control.artifacts import (
    SCHEMA_VERSION, atomic_write_json, collect_paper_figures,
    create_run_directory, load_controller, save_checkpoint, write_run_provenance,
)
import e1_msweep
from e1_noise_figure import load_comparison, main as noise_main, validate_run_configs


def test_run_directories_are_unique_and_refuse_collisions(tmp_path):
    first = create_run_directory(tmp_path, "e1")
    second = create_run_directory(tmp_path, "e1")
    assert first != second
    assert len(first.name.removeprefix("e1_")) == 32
    create_run_directory(tmp_path, "e1", "paper")
    with pytest.raises(FileExistsError, match="Choose a new tag"):
        create_run_directory(tmp_path, "e1", "paper")
    for invalid in ("", "../elsewhere", "/absolute", "two/parts", "a\\b", "a b"):
        with pytest.raises(ValueError):
            create_run_directory(tmp_path, "e1", invalid)


def test_atomic_json_does_not_replace_or_publish_invalid_data(tmp_path):
    target = atomic_write_json(tmp_path / "record.json", {"value": 1})
    with pytest.raises(FileExistsError):
        atomic_write_json(target, {"value": 2})
    assert json.loads(target.read_text()) == {"value": 1}
    with pytest.raises(ValueError):
        atomic_write_json(tmp_path / "invalid.json", {"value": float("nan")})
    assert not (tmp_path / "invalid.json").exists()
    assert not list(tmp_path.glob(".record.json.*"))


def test_provenance_records_configuration_seeds_and_environment(tmp_path):
    directory = create_run_directory(tmp_path, "e1", "run")
    config = {"N": 4, "device": "cpu", "seed": 7}
    target = write_run_provenance(directory, config, 7, 1007)
    saved = json.loads(target.read_text())
    assert json.loads((directory / "config.json").read_text()) == config
    assert saved["schema_version"] == SCHEMA_VERSION
    assert (saved["training_seed"], saved["evaluation_seed"]) == (7, 1007)
    assert saved["libraries"]["torch"]
    assert saved["device"]["requested"] == "cpu"
    assert saved["code"]["commit"]
    assert len(saved["config_sha256"]) == 64
    with pytest.raises(FileExistsError):
        write_run_provenance(directory, config, 7, 1007)


def test_checkpoint_roundtrip_and_raw_state_dict_compatibility(tmp_path):
    source, restored = torch.nn.Linear(2, 3), torch.nn.Linear(2, 3)
    path = save_checkpoint(tmp_path / "model.pt", source, metadata={"N": 2})
    state = torch.load(path, weights_only=True)
    assert set(state) == {"weight", "bias"}
    assert load_controller(path, restored)["N"] == 2
    for left, right in zip(source.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, source)
    legacy = tmp_path / "legacy.pt"
    torch.save(source.state_dict(), legacy)
    assert load_controller(legacy, restored) == {}


def _manifest(tmp_path, figures):
    path = tmp_path / "selected.json"
    path.write_text(json.dumps({"figures": figures, "output_dir": "exports"}))
    return path


def test_collector_preflights_all_sources_without_partial_writes(tmp_path):
    source = tmp_path / "one.pdf"
    source.write_bytes(b"first figure")
    manifest = _manifest(tmp_path, {"first.pdf": "one.pdf", "second.pdf": "missing.pdf"})
    with pytest.raises(ValueError, match="Missing figure"):
        collect_paper_figures(manifest)
    assert not (tmp_path / "exports").exists()
    # Even an existing destination is untouched when another source is missing.
    (tmp_path / "exports").mkdir()
    retained = tmp_path / "exports" / "first.pdf"
    retained.write_bytes(b"existing")
    with pytest.raises(ValueError):
        collect_paper_figures(manifest, overwrite=True)
    assert retained.read_bytes() == b"existing"


def test_collector_uses_manifest_paths_and_requires_explicit_overwrite(tmp_path, monkeypatch):
    (tmp_path / "one.pdf").write_bytes(b"selected run")
    manifest = _manifest(tmp_path, {"first.pdf": "one.pdf"})
    monkeypatch.chdir(tmp_path.parent)
    [destination] = collect_paper_figures(manifest)
    assert destination.read_bytes() == b"selected run"
    with pytest.raises(ValueError, match="Export exists"):
        collect_paper_figures(manifest)
    collect_paper_figures(manifest, overwrite=True)
    manifest.write_text(json.dumps({"figures": {"../bad.pdf": "one.pdf"}}))
    with pytest.raises(ValueError, match="safe path component"):
        collect_paper_figures(manifest)


def test_shell_collector_can_run_outside_repository(tmp_path):
    (tmp_path / "one.pdf").write_bytes(b"figure")
    manifest = _manifest(tmp_path, {"first.pdf": "one.pdf"})
    script = Path(__file__).resolve().parents[1] / "collect_paper_figures.sh"
    process = subprocess.run([str(script), str(manifest)], cwd=tmp_path,
                             capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert (tmp_path / "exports" / "first.pdf").read_bytes() == b"figure"


def _comparison_configs():
    clean = {"N": 4, "k": 1, "T": 1.0, "dt": 0.5, "sigma": 0.0}
    return {**clean, "sigma": 0.3}, clean


@pytest.mark.parametrize("field,value", [("N", 8), ("k", 2), ("T", 2), ("dt", 0.25),
                                       ("sigma", 0.2), ("score_bound", 1.0)])
def test_noise_comparison_rejects_incompatible_reference(field, value):
    noisy, clean = _comparison_configs()
    clean[field] = value
    with pytest.raises(ValueError):
        validate_run_configs(noisy, clean)


def _write_comparison(directory, config, noisy):
    directory.mkdir()
    atomic_write_json(directory / "config.json", config)
    data = {"t": np.array([0.0, 0.5, 1.0])}
    for key in ("r", "winding"):
        suffixes = ("noisy_mean", "noisy_q10", "noisy_q90") if noisy else ("trained",)
        for suffix in suffixes:
            data[f"{key}_{suffix}"] = np.zeros(3)
    np.savez(directory / "trajectories.npz", **data)


def test_noise_comparison_checks_saved_grid_and_output_collision(tmp_path):
    noisy_config, clean_config = _comparison_configs()
    noisy, clean = tmp_path / "noisy", tmp_path / "clean"
    _write_comparison(noisy, noisy_config, True)
    _write_comparison(clean, clean_config, False)
    loaded_config, _, _ = load_comparison(noisy, clean)
    assert loaded_config == noisy_config
    output = tmp_path / "panel.pdf"
    output.write_bytes(b"keep")
    with pytest.raises(SystemExit) as exc:
        noise_main(["--noisy", str(noisy), "--noiseless", str(clean), "--out", str(output)])
    assert exc.value.code == 1
    assert output.read_bytes() == b"keep"
    with np.load(clean / "trajectories.npz") as archive:
        data = dict(archive)
    data["t"] = np.array([0.0, 0.4, 1.0])
    np.savez(clean / "trajectories.npz", **data)
    with pytest.raises(ValueError, match="time grids differ"):
        load_comparison(noisy, clean)


def _write_sweep_cell(root, M, seed, steps=1):
    directory = e1_msweep.cell_dir(M, seed, root)
    directory.mkdir(parents=True)
    config = e1_msweep._expected_config(M, seed, steps)
    write_run_provenance(directory, config, seed, seed + 1000)
    (directory / "model.pt").write_bytes(b"checkpoint fixture")
    atomic_write_json(directory / "metrics.json", {
        "costs": {"trained": {"total": M}, "meanfield": {"total": 10}},
        "r_final": {"trained": 1}, "var_final": {"trained": 0},
        "vxi": {"trained_max": 2},
    })
    atomic_write_json(directory / "complete.json", {"status": "complete"})
    return directory


def test_sweep_refuses_missing_or_partial_aggregation(tmp_path, monkeypatch):
    monkeypatch.setattr(e1_msweep, "MS", [2, 4])
    monkeypatch.setattr(e1_msweep, "SEEDS", [0])
    with pytest.raises(ValueError, match="No complete sweep"):
        e1_msweep.aggregate(steps=1, out_root=tmp_path)
    assert list(tmp_path.iterdir()) == []
    _write_sweep_cell(tmp_path, 2, 0)
    with pytest.raises(ValueError, match="incomplete"):
        e1_msweep.aggregate(steps=1, out_root=tmp_path)
    assert len(list(tmp_path.iterdir())) == 1
    summary_dir = e1_msweep.aggregate(steps=1, out_root=tmp_path, allow_partial=True)
    summary = json.loads((summary_dir / "msweep_summary.json").read_text())
    assert summary["complete"] is False
    assert summary["missing"] == [{"M": 4, "seed": 0}]
    assert (summary_dir / "convergence_M.pdf").is_file()


def test_sweep_reuse_checks_provenance_and_config(tmp_path):
    directory = _write_sweep_cell(tmp_path, 2, 0)
    assert e1_msweep.done(2, 0, steps=1, out_root=tmp_path)
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text())
    config["beta"] = 999
    atomic_write_json(config_path, config, overwrite=True)
    with pytest.raises(ValueError, match="config beta"):
        e1_msweep.done(2, 0, steps=1, out_root=tmp_path)


def test_sweep_does_not_reuse_runs_that_failed_after_saving_metrics(tmp_path):
    directory = _write_sweep_cell(tmp_path, 2, 0)
    (directory / "complete.json").unlink()
    with pytest.raises(ValueError, match="Incomplete or incompatible"):
        e1_msweep.done(2, 0, steps=1, out_root=tmp_path)


def test_sweep_creates_log_parent_and_reports_child_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(e1_msweep, "MS", [2])
    monkeypatch.setattr(e1_msweep, "SEEDS", [0])
    class FailedProcess:
        pid = 123
        returncode = 3
        def poll(self):
            return self.returncode
    monkeypatch.setattr(e1_msweep.subprocess, "Popen", lambda *args, **kwargs: FailedProcess())
    out_root = tmp_path / "new" / "out"
    with pytest.raises(RuntimeError, match=r"failed \(3\)"):
        e1_msweep.launch_all(1, out_root=out_root, lanes=1)
    assert (out_root / "e1_msweep_M2_s0.log").is_file()

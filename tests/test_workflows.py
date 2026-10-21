"""Full paper artifact workflows, including headless archive-only replotting."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from graphon_control.artifacts import load_controller
from graphon_control.controls import TimeLabelGraphonControl
from graphon_control.dynamics import precompute_Ws
from graphon_control.experiments import bounded_confidence, kuramoto


torch.set_num_threads(1)


EXPERIMENTS = [
    pytest.param(kuramoto, kuramoto.E1Config, "run_e1", "e1", "untwist.gif", id="kuramoto"),
    pytest.param(bounded_confidence, bounded_confidence.E2Config, "run_e2", "e2", "opinions.gif", id="bounded-confidence"),
]


def _tiny_config(config_type):
    return config_type(N=8, T=0.1, dt=0.05, num_steps=1, batch_size=2,
                       num_fourier_freqs=1, embed_dim=3, hidden_embed=4,
                       hidden_score=4, seed=3, evaluation_seed=29)


def _numerical_hashes(run):
    names = ("config.json", "provenance.json", "history.json", "metrics.json",
             "model.pt", "trajectories.npz", "run_summary.txt")
    return {name: hashlib.sha256((run / name).read_bytes()).hexdigest() for name in names}


@pytest.mark.parametrize("module,config_type,runner,prefix,trajectory_gif", EXPERIMENTS)
def test_full_workflow_replot_preserves_archived_numerics(
    tmp_path, monkeypatch, module, config_type, runner, prefix, trajectory_gif,
):
    monkeypatch.delenv("DISPLAY", raising=False)
    # Absolute output paths and a foreign cwd exercise the public run APIs without
    # leaving plots in the source checkout or relying on repository-relative paths.
    working = tmp_path / "foreign-cwd"
    working.mkdir()
    monkeypatch.chdir(working)
    out_root, fig_root = tmp_path / "runs", tmp_path / "unrequested-mirrors"
    config = _tiny_config(config_type)
    getattr(module, runner)(config, tag="full", out_root=str(out_root),
                            fig_root=str(fig_root), make_plots=True)
    run = out_root / f"{prefix}_full"
    marker = json.loads((run / "complete.json").read_text())
    assert marker["status"] == "complete"
    assert marker.get("initial_plots_generated", marker.get("plots_generated")) is True
    assert not fig_root.exists()
    assert list(working.iterdir()) == []
    assert (run / "state_evolution.pdf").stat().st_size > 0
    assert (run / "kernel_strip.pdf").stat().st_size > 0
    assert (run / trajectory_gif).stat().st_size > 0
    assert (run / "kernel_time.gif").stat().st_size > 0
    provenance = json.loads((run / "provenance.json").read_text())
    assert provenance["training_seed"] == config.seed
    assert provenance["evaluation_seed"] == config.evaluation_seed
    before = _numerical_hashes(run)
    with np.load(run / "trajectories.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    restored = TimeLabelGraphonControl(
        torch.as_tensor(arrays["xi"]), config.num_fourier_freqs, config.embed_dim,
        config.hidden_embed, config.hidden_score, time_scale=config.T,
        score_bound=config.score_bound,
    )
    load_controller(run / "model.pt", restored)
    with torch.no_grad():
        reloaded_weights = precompute_Ws(restored, config.NT, config.dt, "cpu", config.time_chunk)
    torch.testing.assert_close(reloaded_weights, torch.as_tensor(arrays["W_trained"]))

    def unexpected_training(*args, **kwargs):
        raise AssertionError("Replot must not train")
    monkeypatch.setattr(module, "train_controller", unexpected_training)
    module.plot_run(run)
    assert _numerical_hashes(run) == before
    assert json.loads((run / "complete.json").read_text()) == marker
    with np.load(run / "trajectories.npz", allow_pickle=False) as archive:
        assert set(archive.files) == set(arrays)
        for name, expected in arrays.items():
            np.testing.assert_array_equal(archive[name], expected)
    # Existing tags are refused before calling the patched training entry point.
    with pytest.raises(FileExistsError, match="Run already exists"):
        getattr(module, runner)(config, tag="full", out_root=str(out_root), make_plots=False)
    assert _numerical_hashes(run) == before


@pytest.mark.parametrize("module,config_type,runner,prefix,trajectory_gif", EXPERIMENTS)
def test_existing_mirror_is_refused_without_modifying_it(
    tmp_path, module, config_type, runner, prefix, trajectory_gif,
):
    config = _tiny_config(config_type)
    getattr(module, runner)(config, tag="export", out_root=str(tmp_path), make_plots=False)
    run = tmp_path / f"{prefix}_export"
    mirror = tmp_path / "selected-mirror"
    mirror.mkdir()
    existing = mirror / "state_evolution.pdf"
    existing.write_bytes(b"previous selected figure")
    before = _numerical_hashes(run)
    with pytest.raises(FileExistsError):
        module.plot_run(run, fig_dir=mirror)
    assert existing.read_bytes() == b"previous selected figure"
    assert _numerical_hashes(run) == before
    assert not list(run.glob("*.pdf"))
    assert not list(run.glob("*.gif"))
    # A new explicit mirror can be generated from a numerical-only run. The
    # completion marker tracks the resulting plots; scientific artifacts stay fixed.
    fresh_mirror = tmp_path / "new-mirror"
    module.plot_run(run, fig_dir=fresh_mirror)
    assert (fresh_mirror / "state_evolution.pdf").stat().st_size > 0
    assert (run / trajectory_gif).stat().st_size > 0
    assert json.loads((run / "complete.json").read_text())["plots_generated"] is True
    assert _numerical_hashes(run) == before


@pytest.mark.parametrize("module,config_type,runner,prefix,trajectory_gif", EXPERIMENTS)
def test_training_failure_has_no_completion_marker(
    tmp_path, monkeypatch, module, config_type, runner, prefix, trajectory_gif,
):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate optimizer failure")
    monkeypatch.setattr(module, "train_controller", fail)
    with pytest.raises(RuntimeError, match="deliberate optimizer failure"):
        getattr(module, runner)(_tiny_config(config_type), tag="failed", out_root=str(tmp_path), make_plots=False)
    directory = tmp_path / f"{prefix}_failed"
    assert (directory / "provenance.json").is_file()
    assert not (directory / "complete.json").exists()

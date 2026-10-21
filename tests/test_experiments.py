"""Experiment entry points preserve the common scientific and CLI contracts."""

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from graphon_control.dynamics import mean_field_kernel
from graphon_control.experiments import bounded_confidence, generic, kuramoto


torch.set_num_threads(1)


@pytest.mark.parametrize("config_type", [kuramoto.E1Config, bounded_confidence.E2Config, generic.TrainConfig])
def test_default_configuration_json_roundtrip(config_type):
    config = config_type()
    restored = config_type(**json.loads(json.dumps(dataclasses.asdict(config))))
    assert dataclasses.asdict(restored) == dataclasses.asdict(config)


@pytest.mark.parametrize("invalid", [{"T": 1.0, "dt": 0.3}, {"sigma": -1}, {"N": 0},
    {"evaluation_seed": -2}, {"score_bound": float("nan")}])
@pytest.mark.parametrize("config_type", [kuramoto.E1Config, bounded_confidence.E2Config, generic.TrainConfig])
def test_invalid_experiment_configuration(config_type, invalid):
    with pytest.raises(ValueError):
        config_type(**invalid)


@pytest.mark.parametrize("config_type", [kuramoto.E1Config, bounded_confidence.E2Config])
def test_experiment_jitter_must_be_nonnegative(config_type):
    with pytest.raises(ValueError, match="jitter"):
        config_type(jitter=-1)


@pytest.mark.parametrize("level", [4.0, True, -2, 3])
def test_kuramoto_coarse_level_requires_integer_partition(level):
    with pytest.raises(ValueError, match="x0_level"):
        kuramoto.E1Config(N=8, x0_level=level)


def test_bounded_confidence_modes_rejects_bool():
    with pytest.raises(ValueError, match="modes"):
        bounded_confidence.E2Config(modes=True)


def test_e1_and_e2_use_identical_objective_for_identical_trajectories():
    xs = torch.tensor([[[0.0, 2.0, 4.0]], [[0.5, 2.0, 3.5]], [[1.0, 2.0, 3.0]]])
    Ws = mean_field_kernel(3).expand(2, 3, 3)
    parameters = dict(N=8, T=0.2, dt=0.1, beta=0.7, gamma_target=1.4,
                      lambda_terminal=0.3, x_target=-1.0)
    # The objective depends on provided trajectories rather than cfg.N.
    one = kuramoto.e1_cost(xs, Ws, kuramoto.E1Config(**parameters))
    two = bounded_confidence.e2_cost(xs, Ws, bounded_confidence.E2Config(**parameters))
    assert one.keys() == two.keys()
    for key in one:
        assert torch.equal(one[key], two[key])


@pytest.mark.parametrize("module,config_type,runner", [
    (kuramoto, kuramoto.E1Config, "run_e1"),
    (bounded_confidence, bounded_confidence.E2Config, "run_e2"),
    (generic, generic.TrainConfig, "run_experiment"),
])
def test_cli_loads_json_flags_override_and_smoke_caps_model(tmp_path, monkeypatch,
                                                          module, config_type, runner):
    config = config_type(N=32, T=0.2, dt=0.1, beta=2.0, batch_size=3,
                         num_steps=123, embed_dim=24)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dataclasses.asdict(config)))
    seen = {}
    def capture(cfg, **kwargs):
        seen.update(cfg=cfg, kwargs=kwargs)
    monkeypatch.setattr(module, runner, capture)
    module.main(["--config", str(path), "--beta", "3.5", "--smoke", "--no-plots", "--tag", "check"])
    restored = seen["cfg"]
    assert restored.beta == 3.5
    assert restored.T == 0.2 and restored.dt == 0.1 and restored.batch_size == 3
    assert restored.num_steps == 20
    assert restored.N == 16 and restored.embed_dim == 8
    assert restored.num_fourier_freqs == 2
    assert not seen["kwargs"]["make_plots"]


@pytest.mark.parametrize("module", [kuramoto, bounded_confidence, generic])
def test_cli_reports_invalid_time_grid_before_running(module, capsys):
    with pytest.raises(SystemExit) as error:
        module.main(["--T", "1", "--dt", "0.3", "--no-plots"])
    assert error.value.code == 2
    assert "T/dt must be" in capsys.readouterr().err


@pytest.mark.parametrize("script", ["e1_kuramoto.py", "e2_bc.py", "model_1.py"])
def test_compatibility_cli_help_works_without_display(tmp_path, script):
    environment = os.environ.copy()
    environment.pop("DISPLAY", None)
    environment.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(tmp_path / "mpl"),
                       XDG_CACHE_HOME=str(tmp_path / "cache"))
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / script), "--help"],
                            cwd=tmp_path, env=environment, capture_output=True,
                            text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout and "--no-plots" in result.stdout


@pytest.mark.parametrize("module,config_type,runner,prefix", [
    (kuramoto, kuramoto.E1Config, "run_e1", "e1"),
    (bounded_confidence, bounded_confidence.E2Config, "run_e2", "e2"),
])
def test_short_paper_experiment_archives_reproducible_numerical_run(tmp_path,
                                                               module, config_type, runner, prefix):
    cfg = config_type(N=8, T=0.1, dt=0.05, batch_size=2, num_steps=2,
                      num_fourier_freqs=2, embed_dim=4, hidden_embed=8,
                      hidden_score=8, seed=0)
    getattr(module, runner)(cfg, tag="numerical", out_root=str(tmp_path), make_plots=False)
    run = tmp_path / f"{prefix}_numerical"
    assert config_type(**json.loads((run / "config.json").read_text())) == cfg
    history = json.loads((run / "history.json").read_text())
    assert len(history["loss"]) == 2
    assert all(np.isfinite(history["loss"]))
    assert (run / "model.pt").is_file()
    assert (run / "metrics.json").is_file()
    assert json.loads((run / "complete.json").read_text())["status"] == "complete"
    with np.load(run / "trajectories.npz", allow_pickle=False) as archive:
        assert archive["x_trained"].shape == (3, 8)
        assert archive["W_trained"].shape == (2, 8, 8)
        assert np.isfinite(archive["x_trained"]).all()
    assert not list(run.glob("*.pdf"))
    assert not list(run.glob("*.gif"))


def test_generic_default_time_scale_uses_horizon_and_runs(tmp_path):
    cfg = generic.TrainConfig(N=4, T=0.1, dt=0.05, sigma=0, batch_size=2,
                              num_steps=2, num_fourier_freqs=2, embed_dim=4,
                              hidden_embed=8, hidden_score=8)
    result = generic.run_experiment(cfg, out_root=tmp_path, run_tag="generic",
                                    eval_batch_size=2, make_plots=False)
    assert result["model"].time_scale == cfg.T
    assert len(result["history"]["loss"]) == 2
    assert len(result["xs"]) == 3
    assert Path(result["run_dir"], "complete.json").is_file()

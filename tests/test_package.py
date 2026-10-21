"""Public package imports, compatibility entry points, and legacy checkpoints."""

import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from graphon_control.artifacts import load_controller
from graphon_control.controls import TimeLabelGraphonControl


def test_numerical_imports_need_no_plotting_and_create_no_files(tmp_path):
    # Check in a fresh interpreter: other tests may intentionally import plotting.
    source_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(source_root))
    program = """
import importlib.abc
import pathlib
import sys
class NoPlotting(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'matplotlib', 'PIL'}:
            raise AssertionError('numerical import attempted plotting: ' + fullname)
sys.meta_path.insert(0, NoPlotting())
import graphon_control
import graphon_control.controls
import graphon_control.dynamics
import graphon_control.objectives
import graphon_control.diagnostics
import graphon_control.training
import graphon_control.experiments.generic
import graphon_control.experiments.kuramoto
import graphon_control.experiments.bounded_confidence
import model_1
import e1_kuramoto
import e2_bc
assert not any(m.startswith('matplotlib') for m in sys.modules)
assert model_1.TimeLabelGraphonControl is graphon_control.TimeLabelGraphonControl
assert e1_kuramoto.run_e1 is graphon_control.experiments.kuramoto.run_e1
assert e2_bc.run_e2 is graphon_control.experiments.bounded_confidence.run_e2
assert list(pathlib.Path.cwd().iterdir()) == []
print('numerical imports have no plotting or output side effects')
"""
    result = subprocess.run([sys.executable, "-c", program], cwd=tmp_path,
                            env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "no plotting or output side effects" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_legacy_raw_controller_checkpoint_loads_through_public_api(tmp_path):
    # A handcrafted original-model state dict fixes the old names/shapes without
    # instantiating the new class to generate the compatibility fixture itself.
    xi = torch.tensor([0.25, 0.75])
    legacy = {
        "xi": xi,
        "psi_xi_cached": torch.cat((torch.sin(2 * torch.pi * xi[:, None]),
                                    torch.cos(2 * torch.pi * xi[:, None])), dim=-1),
        "ff.freqs": torch.tensor([1.0]),
    }
    for module, sizes in (("receiver_net", (2, 3, 3, 2)),
                          ("sender_net", (2, 3, 3, 2)),
                          ("score_net", (5, 4, 4, 1))):
        for index, (inputs, outputs) in zip((0, 2, 4), zip(sizes[:-1], sizes[1:])):
            legacy[f"{module}.net.{index}.weight"] = torch.zeros(outputs, inputs)
            legacy[f"{module}.net.{index}.bias"] = torch.zeros(outputs)
    checkpoint = tmp_path / "legacy-model.pt"
    torch.save(legacy, checkpoint)
    restored = TimeLabelGraphonControl(xi, 1, 2, 3, 4)
    assert load_controller(checkpoint, restored) == {}
    torch.testing.assert_close(restored(torch.tensor([0.0, 0.5])), torch.full((2, 2, 2), 0.5))


def test_generic_run_archives_reloadable_numerics_without_plotting(tmp_path):
    from graphon_control.experiments.generic import TrainConfig, run_experiment

    config = TrainConfig(N=4, T=0.1, dt=0.05, sigma=0, num_steps=1, batch_size=2,
                         num_fourier_freqs=1, embed_dim=3, hidden_embed=4, hidden_score=4,
                         seed=2, evaluation_seed=7, device="cpu")
    result = run_experiment(config, out_root=tmp_path, run_tag="tiny", eval_batch_size=3,
                            make_plots=False)
    directory = Path(result["run_dir"])
    for name in ("config.json", "provenance.json", "history.json", "metrics.json",
                 "model.pt", "trajectories.npz", "complete.json"):
        assert (directory / name).is_file()
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["l2_reg_cost"] >= config.beta * config.T - 1e-8
    assert json.loads((directory / "complete.json").read_text())["plots_generated"] is False
    restored = TimeLabelGraphonControl(result["xi"], 1, 3, 4, 4, time_scale=config.T)
    load_controller(directory / "model.pt", restored)
    times = torch.tensor([0.0, config.dt])
    torch.testing.assert_close(restored(times), result["model"](times))

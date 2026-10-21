"""Saved-array fixtures only: no dynamics, optimization, or checkpoint loading."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import reproduce_appendix as exporter
from graphon_control import artifacts, dynamics, training


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(path, data):
    path.write_text(json.dumps(data))


@pytest.fixture(scope="module")
def archived_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("appendix-handwritten-array-fixture")
    manifest = {"schema_version": 1, "runs": {}}
    for name in exporter.RUN_NAMES:
        directory = root / name
        directory.mkdir()
        cfg = {"N": 64, "dt": .05, "lambda_terminal": 0., "gamma_target": 100.,
               "x_target": 0., "jitter": .1, "batch_size": 16, "lr": .001,
               "num_fourier_freqs": 8, "embed_dim": 64, "hidden_embed": 128,
               "hidden_score": 128, "num_steps": 3000 if name == "k3" else 2000}
        if name == "bc":
            cfg.update(T=10., sigma=0., beta=1., eps=1., gain=4., phi_form="poly",
                       span=6., amp=.5, modes=4)
        else:
            k = 2 if name == "noise" else int(name[1:])
            cfg.update(k=k, T={1: 5., 2: 12., 3: 20.}[k], beta=10., sigma=.3 if name == "noise" else 0.)
        n, nt = cfg["N"], round(cfg["T"] / cfg["dt"])
        xi = (np.arange(n) + .5) / n
        initial = (6*(xi-.5) + .5*np.sin(8*np.pi*xi) if name == "bc"
                   else 2*np.pi*cfg["k"]*(xi-.5))
        # Algebraic test arrays, deliberately not the result of a solver.
        data = {"t": np.arange(nt+1)*cfg["dt"], "t_W": np.arange(nt)*cfg["dt"], "xi": xi,
                "W_trained": np.full((nt,n,n), 1/n), "vxi_trained": np.zeros(nt)}
        metrics = {"vxi": {"trained_max": 0., "trained_final": 0.}, "costs": {}, "var_final": {}}
        if name != "bc":
            metrics["r_final"], metrics["winding_final"] = {}, {}
        for arm in exporter.ARMS:
            factor = np.ones(nt+1) if arm == "meanfield" else np.linspace(1., .1, nt+1)
            x = factor[:,None] * initial
            var, mean = x.var(axis=1), x.mean(axis=1)
            data["x_"+arm], data["var_"+arm] = x, var
            cost = {"state_cost": cfg["dt"]*var[:-1].sum(), "l2_reg_cost": cfg["beta"]*cfg["T"],
                    "terminal_var": var[-1], "terminal_target": cfg["gamma_target"]*mean[-1]**2,
                    "terminal_mean": mean[-1]}
            cost["total"] = cost["state_cost"] + cost["l2_reg_cost"] + cost["terminal_target"]
            metrics["costs"][arm], metrics["var_final"][arm] = cost, var[-1]
            if name != "bc":
                r, w = np.abs(np.exp(1j*x).mean(-1)), exporter.winding(x)
                data["r_"+arm], data["winding_"+arm] = r, w
                metrics["r_final"][arm], metrics["winding_final"][arm] = r[-1], w[-1]
            else:
                data["K_"+arm] = np.ones(nt+1, dtype=int)
        if name == "noise":
            for key in ("r", "winding"):
                for suffix in ("mean", "q10", "q90"):
                    data[f"{key}_noisy_{suffix}"] = data[f"{key}_trained"].copy()
            metrics["noisy_eval"] = {"sigma": .3, "n_replicates": 32,
                "r_final_mean": data["r_noisy_mean"][-1], "r_final_q10": data["r_noisy_q10"][-1],
                "r_final_q90": data["r_noisy_q90"][-1], "winding_final_mean": data["winding_noisy_mean"][-1]}
        _dump(directory/"config.json", cfg)
        _dump(directory/"metrics.json", metrics)
        np.savez_compressed(directory/"trajectories.npz", **data)
        # Optional pinned files are hashed only, never deserialized.
        (directory/"model.pt").write_bytes(b"not a torch checkpoint")
        manifest["runs"][name] = {"path": str(directory), "sha256": {p.name: _hash(p) for p in directory.iterdir()}}
    manifest_path = root/"manifest.json"
    _dump(manifest_path, manifest)
    return manifest_path


@pytest.fixture(autouse=True)
def forbid_experiment_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No training, dynamics, or checkpoint loads are authorized")
    monkeypatch.setattr(training, "train_controller", forbidden)
    for name in ("rollout_with_Ws", "rollout_dynamics_w1", "precompute_Ws"):
        monkeypatch.setattr(dynamics, name, forbidden)
    monkeypatch.setattr(artifacts, "load_controller", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)


def test_saved_array_export_preserves_inputs_and_lists_all_outputs(archived_fixture, tmp_path):
    manifest = json.loads(archived_fixture.read_text())
    before = {p: _hash(Path(p)) for spec in manifest["runs"].values()
              for p in (str(Path(spec["path"])/filename) for filename in spec["sha256"])}
    output = tmp_path/"saved-array-output"
    exporter.main(["--manifest", str(archived_fixture), "--output", str(output)])
    expected = {"bc_opinion_traj.pdf", "bc_kernel_strip.pdf", "kur_rw_panel.pdf", "kur_kernel_strip.pdf",
                "kur_noise_panel.pdf", "tables.csv", "noise_summary.json", "verification.json", "provenance.json"}
    assert {p.name for p in output.iterdir()} == expected
    for pdf in output.glob("*.pdf"):
        assert pdf.read_bytes().startswith(b"%PDF-")
    provenance = json.loads((output/"provenance.json").read_text())
    assert provenance["new_experiments"] is False
    assert provenance["checkpoint_loads"] is False
    assert provenance["manifest"]["sha256"] == _hash(archived_fixture)
    for filename, record in provenance["outputs"].items():
        assert record["sha256"] == _hash(output/filename)
    assert {p: _hash(Path(p)) for p in before} == before
    verification = json.loads((output/"verification.json").read_text())
    assert all(run["saved_array_checks_passed"] for run in verification["runs"].values())
    assert verification["runs"]["bc"]["passed"] is False
    assert verification["runs"]["bc"]["status"] == "appendix_rounding_mismatch"
    # Fixture data must not be passed off as the published values.
    assert not verification["runs"]["bc"]["appendix_rounding"]["J_trained"]["matches"]
    summary = json.loads((output/"noise_summary.json").read_text())
    assert summary["all_paths_zero_terminal_winding_verified"] is False
    assert summary["individual_noisy_paths_saved"] is False
    with (output/"tables.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert {(r["run"], r["arm"]) for r in rows} == {(r,a) for r in ("bc","k1","k2","k3") for a in exporter.ARMS}
    assert not list(tmp_path.glob(".appendix-stage-*"))


def test_bundled_original_results_match_appendix():
    """Verify the published statistics against the actual archived inputs."""
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "appendix_manifest.json"
    runs, _ = exporter.load_manifest(manifest_path)
    results = {name: exporter.diagnostics(name, run) for name, run in runs.items()}
    assert all(result["passed"] for result in results.values())
    bc = results["bc"]["recomputed"]
    assert bc["meanfield"]["clusters_final"] == 4
    assert bc["trained"]["clusters_final"] == 1
    assert abs(bc["meanfield"]["terminal_mean"]) < .0005
    assert f'{bc["trained"]["terminal_mean"]:.3f}' == "-0.001"
    assert f'{bc["meanfield"]["terminal_var"]:.3f}' == "2.800"
    assert bc["trained"]["terminal_var"] < 1e-12
    assert f'{bc["trained"]["max_vxi"]:.1f}' == "46.3"
    for name, variance, peak in (("k1", "1.0e-02", "3.7"),
                                  ("k2", "1.5e-05", "12.0"),
                                  ("k3", "3.6e-08", "24.4")):
        result = results[name]["recomputed"]
        assert f'{result["trained"]["terminal_var"]:.1e}' == variance
        assert f'{result["trained"]["max_vxi"]:.1f}' == peak
        assert result["meanfield"]["max_vxi"] == 0
    k3 = results["k3"]["recomputed"]
    assert f'{k3["meanfield"]["terminal_var"]:.3f}' == "29.602"
    assert f'{k3["meanfield"]["r_final"]:.3f}' == "0.000"
    assert f'{k3["trained"]["r_final"]:.3f}' == "1.000"
    noisy = runs["noise"]["metrics"]["noisy_eval"]
    assert noisy["n_replicates"] == 32
    assert f'{noisy["r_final_mean"]:.3f}' == "0.974"
    manifest = json.loads(manifest_path.read_text())
    for name, source in manifest["figures"].items():
        assert _hash(root / source) == manifest["figures_sha256"][name]


def _mutated_manifest(source, tmp_path, name, filename, mutate):
    manifest = json.loads(source.read_text())
    original = Path(manifest["runs"][name]["path"])/filename
    directory = tmp_path/"mutated"
    directory.mkdir()
    spec = manifest["runs"][name]
    # All remaining files stay read-only in the shared fixture directory.
    for key in spec["sha256"]:
        target = directory/key
        if key != filename:
            target.symlink_to(Path(spec["path"])/key)
    raw = original.read_bytes()
    if filename.endswith(".npz"):
        with np.load(original) as z:
            arrays = {key: z[key] for key in z.files}
        mutate(arrays)
        np.savez_compressed(directory/filename, **arrays)
    else:
        value = json.loads(raw)
        mutate(value)
        _dump(directory/filename, value)
    spec["path"] = str(directory)
    spec["sha256"][filename] = _hash(directory/filename)
    path = tmp_path/"manifest.json"
    _dump(path, manifest)
    return path


def test_tampered_hash_fails_before_any_output(archived_fixture, tmp_path):
    manifest = json.loads(archived_fixture.read_text())
    manifest["runs"]["noise"]["sha256"]["metrics.json"] = "0"*64
    path = tmp_path/"manifest.json"
    _dump(path, manifest)
    output = tmp_path/"absent-parent"/"out"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        exporter.export_appendix(path, output)
    assert not output.parent.exists()


@pytest.mark.parametrize("key", ["x_trained", "W_trained", "vxi_trained", "t", "r_trained", "winding_noisy_q90"])
def test_missing_saved_arrays_never_reconstruct(archived_fixture, tmp_path, key):
    path = _mutated_manifest(archived_fixture, tmp_path, "noise", "trajectories.npz", lambda d: d.pop(key))
    output = tmp_path/"exports"/"out"
    with pytest.raises(ValueError, match="missing saved array"):
        exporter.export_appendix(path, output)
    assert not output.parent.exists()


@pytest.mark.parametrize("field,value", [("N", 128), ("dt", .1), ("num_steps", 10), ("jitter", .2)])
def test_other_experiments_refused(archived_fixture, tmp_path, field, value):
    path = _mutated_manifest(archived_fixture, tmp_path, "bc", "config.json", lambda d: d.update({field:value}))
    output = tmp_path/"out"
    with pytest.raises(ValueError, match="original scientific setting"):
        exporter.export_appendix(path, output)
    assert not output.exists()


def test_missing_scientific_config_never_defaults(archived_fixture, tmp_path):
    path = _mutated_manifest(archived_fixture, tmp_path, "bc", "config.json", lambda d: d.pop("beta"))
    with pytest.raises(ValueError, match="beta must be recorded"):
        exporter.export_appendix(path, tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_metrics_mismatch_refused(archived_fixture, tmp_path):
    path = _mutated_manifest(archived_fixture, tmp_path, "k1", "metrics.json", lambda d: d["costs"]["trained"].update(total=0))
    with pytest.raises(ValueError, match="saved arrays disagree"):
        exporter.export_appendix(path, tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_existing_output_is_never_replaced(archived_fixture, tmp_path):
    output = tmp_path/"out"
    output.mkdir()
    (output/"sentinel").write_text("keep")
    with pytest.raises(FileExistsError):
        exporter.export_appendix(archived_fixture, output)
    assert (output/"sentinel").read_text() == "keep"


def test_output_created_during_render_is_never_overwritten(archived_fixture, tmp_path, monkeypatch):
    output = tmp_path/"out"
    original_mkdir = Path.mkdir
    def racing_mkdir(path, *args, **kwargs):
        if path == output:
            original_mkdir(path)
            (path/"other-writer").write_text("preserve")
            raise FileExistsError("raced output")
        return original_mkdir(path, *args, **kwargs)
    monkeypatch.setattr(Path, "mkdir", racing_mkdir)
    with pytest.raises(FileExistsError, match="raced output"):
        exporter.export_appendix(archived_fixture, output)
    assert {p.name for p in output.iterdir()} == {"other-writer"}
    assert not list(tmp_path.glob(".appendix-stage-*"))


def test_publication_failure_preserves_unrelated_concurrent_content(archived_fixture, tmp_path, monkeypatch):
    output = tmp_path/"out"
    original_link = exporter.os.link
    calls = 0
    def racing_link(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            (output/"other-writer").write_text("preserve")
            raise OSError("simulated publication failure")
        return original_link(source, target)
    monkeypatch.setattr(exporter.os, "link", racing_link)
    with pytest.raises(OSError, match="simulated publication failure"):
        exporter.export_appendix(archived_fixture, output)
    assert {p.name for p in output.iterdir()} == {"other-writer"}
    assert (output/"other-writer").read_text() == "preserve"
    assert not list(tmp_path.glob(".appendix-stage-*"))

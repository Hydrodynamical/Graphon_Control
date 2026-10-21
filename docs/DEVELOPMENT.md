# Developing and reproducing experiments

Install the package and development tools from the repository root:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

`requirements-tested.txt` records the direct dependency versions of the
verified Python 3.12 environment. It is a tested snapshot, not a full transitive
or cross-platform lockfile. CPU tests run in GitHub Actions; accelerator tests
run when compatible hardware is available.

## Structure

- `graphon_control/controls.py`: neural time/label kernel.
- `dynamics.py`, `objectives.py`, `diagnostics.py`: shared scientific operations.
- `training.py`: Adam loop with finite-loss and finite-gradient guards.
- `artifacts.py`: new run directories, provenance, checkpoints and figure export.
- `interactions.py`, `initial_conditions.py`: reusable example ingredients.
- `experiments/`: model-specific configurations, baselines and orchestration.
- `plotting.py`: visualization, separate from the importable numerical core.
- Root Python scripts preserve the previous command and import entry points.

To add a model, supply a differentiable interaction function, an initial-state
sampler and an objective to `train_controller`; reuse the shared rollout and
cost. Keep model-specific diagnostics beside the experiment. Numerical
conventions and the relationship to the paper are in [NUMERICS.md](NUMERICS.md).

## Run and inspect

```bash
python e1_kuramoto.py --smoke --no-mirror
python e2_bc.py --smoke --no-mirror
python model_1.py --smoke --no-plots
```

Smoke runs perform 20 updates on a smaller grid/network and are workflow checks,
not reproductions of manuscript results. `--no-plots` skips rendering only;
configuration, weights, trajectories and diagnostics are still saved.

For the full paper configurations:

```bash
python e2_bc.py --tag paper_bc
python e1_kuramoto.py --k 3 --T 20 --steps 3000 --tag paper_kuramoto
```

Run tags reserve new directories and existing tags are refused. By default,
figures stay in their run directory. `--mirror-figures` also writes into a
new figure directory named for the run, never into a directory selected only
by model parameters.

Every run records a complete `config.json`, `provenance.json` (code revision,
working-tree state, dependency versions and training/evaluation seeds), raw
controller `model.pt`, and numerical artifacts. `complete.json` is written only
after the requested outputs succeed. Training and evaluation have independent
seeds; reproducibility across different hardware/library versions is not assumed.

## Load settings and regenerate plots

```bash
python e1_kuramoto.py --config out_1/e1_paper_kuramoto/config.json --tag rerun
python e1_kuramoto.py --replot out_1/e1_paper_kuramoto
python e2_bc.py --replot out_1/e2_paper_bc
```

Explicit scientific CLI flags override saved settings. Replotting uses saved
arrays and never retrains. It replaces plots within the selected run; original
weights, arrays and metrics are preserved. Generic runs also archive sampled
interaction values so replotting does not need to reconstruct arbitrary Python
callables. Existing raw state_dict checkpoints remain readable through
`artifacts.load_controller`.

## Collect manuscript figures

Choose exact artifacts in a JSON manifest, with paths relative to the manifest:

```json
{
  "output_dir": "figures/paper",
  "figures": {
    "kur_winding.pdf": "out_1/e1_paper_kuramoto/winding.pdf",
    "bc_opinion_traj.pdf": "out_1/e2_paper_bc/opinion_traj.pdf"
  }
}
```

```bash
./collect_paper_figures.sh paper-figures.json
```

Collection checks every input and destination before copying. Existing exports
require `--overwrite`. Add schematic PDFs explicitly once available; the public
checkout does not contain or generate those manuscript assets. The noise panel
helper checks compatible noisy/noiseless run settings before comparing them.

# Graphon Control

**Learning network interactions to steer collective behavior.**

A PyTorch research implementation that learns how agents influence one another over time. It combines neural interaction kernels, differentiable simulation, and stochastic optimization.

Companion code for *Optimal Graphon Control of Collective Dynamics*, by **Joseph K. Miller and Brandon Su**.

## Experiments

- **Opinion dynamics:** Learn interactions that prevent fragmentation into isolated groups.
- **Kuramoto oscillators:** Learn interactions that untwist phase patterns and encourage alignment.

Both experiments compare learned control against baseline networks and save trajectories, diagnostics, plots, and animations.

## Method

**Encode agent labels → construct interaction weights → simulate dynamics → optimize with Adam.**

The controller uses time and agent labels. Directed interaction weights are nonnegative and row-normalized. Simulations support deterministic dynamics and Brownian noise.

## Quick start

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python e1_kuramoto.py --smoke --device cpu --no-mirror
```

This runs a 20-iteration workflow check on a small grid and network. Results appear under `out_1/`, including `untwist.gif`, `kernel_time.gif`, and `metrics.json`.

## Code

| File | Contents |
| --- | --- |
| `graphon_control/` | Reusable controller, dynamics, objectives, and training |
| `model_1.py` | General experiment entry point |
| `e1_kuramoto.py` | Oscillator experiment |
| `e2_bc.py` | Opinion dynamics experiment |
| `model_1_viz.py` | Visualization utilities |

Use `python <script> --help` for experiment options. See [development](docs/DEVELOPMENT.md) for testing and reproduction, and [numerical conventions](docs/NUMERICS.md) for the connection to the paper.

## Research scope

Controls are open loop and assume a known interaction law. Optimization searches a neural control family; global optimality is not certified. Receiver variation is measured rather than constrained during training.

[MIT License](https://github.com/Hydrodynamical/Graphon_Control/blob/main/LICENSE)

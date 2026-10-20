# Graphon Control

**Learning network interactions to steer collective behavior.**

A PyTorch research implementation that learns how agents influence one another over time. It combines neural interaction kernels, differentiable simulation, and stochastic optimization.

Companion code for *Graphon Controlled Opinion Dynamics*, by **Joseph K. Miller and Brandon Su**.

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
python -m pip install torch numpy matplotlib tqdm pillow
python e1_kuramoto.py --smoke --device cpu --no-mirror
```

This runs a 20-iteration workflow check. Results appear under `out_1/`, including `untwist.gif`, `kernel_time.gif`, and `metrics.json`.

## Code

| File | Contents |
| --- | --- |
| `model_1.py` | Neural controller, simulation, and training |
| `e1_kuramoto.py` | Oscillator experiment |
| `e2_bc.py` | Opinion dynamics experiment |
| `model_1_viz.py` | Visualization utilities |

Use `python <script> --help` for experiment options.

## Research scope

Controls are open loop and assume a known interaction law. Optimization searches a neural control family; global optimality is not certified. Receiver variation is measured rather than constrained during training.

[MIT License](https://github.com/Hydrodynamical/Graphon_Control/blob/main/LICENSE)

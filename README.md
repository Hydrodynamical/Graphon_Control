# Graphon_Control

Opinion dynamics in graphon formulation, with a control driven by a neural network.

---

## Background

Each agent is indexed by a label `xi` in `I = [0, 1]` and carries a scalar opinion state `x(t, xi)`. The uncontrolled continuum dynamics are

```text
d/dt x(t, xi) = integral_I  w(t, xi, zeta) * phi(x(t, zeta) - x(t, xi)) dzeta
```

where `w(t, xi, zeta)` is a graphon encoding interaction strengths and `phi` is the interaction nonlinearity (e.g. `tanh`, linear, bounded-confidence). Control is introduced by letting `w` be chosen by a neural-network controller. The paper studies three control classes:

- **W1** — graphon depends on time and labels only: `w(t, xi, zeta)`
- **W2** — also depends on the receiver's local state: `w(t, x(xi), xi, zeta)`
- **W3** — also depends on sender state: `w(t, x(xi), x(zeta), xi, zeta)`

`model_1.py` implements the **W1** control.

---

## `model_1.py` — W1 Control (Time and Label Dependent)

### `TrainConfig`

Dataclass holding all hyperparameters.

| Field | Description |
| --- | --- |
| `N` | Number of agents / label grid points |
| `T`, `dt` | Horizon and time step |
| `sigma` | Noise level in the SDE |
| `beta` | L² control regularization weight (penalty on `‖W‖²`) |
| `eta_uniform` | Uniform-graphon regularization weight (penalty on `‖W − U‖²`) |
| `regularizer_mode` | Which graphon penalty to apply: `"l2"`, `"uniform"`, or `"l2_plus_uniform"` |
| `lambda_terminal` | Weight on terminal variance penalty |
| `gamma_target` | Weight on terminal mean-tracking penalty |
| `x_target` | Target value for the terminal empirical mean |
| `enforce_symmetry` | If `True`, produce a symmetric (undirected) graphon |
| `positive_map` | Nonnegativity map in symmetric mode: `"softplus"` or `"exp"` |
| `batch_size`, `num_steps`, `lr` | Training hyperparameters |
| `seed` | Random seed |

### `TimeLabelGraphonControl` (neural network)

Parameterizes `w_theta : [0,T] x I^2 -> R_+`. Labels are encoded via Fourier features

```text
psi(xi) = (sin(2*pi * 2^k * xi), cos(2*pi * 2^k * xi))  for k = 0, ..., m-1
```

Two modes controlled by `enforce_symmetry`:

- **Directed (`False`)** — separate receiver MLP `phi_theta(psi(xi))` and sender MLP `phi~_theta(psi(zeta))` feed into a score network `g_theta(t, h_i, h~_j)`. Masked row-softmax produces a row-stochastic matrix (each row sums to 1, diagonal zeroed).
- **Undirected (`True`)** — shared embedding, score symmetrized as `0.5*(A + A^T)`, then `softplus` or `exp` applied, diagonal zeroed, and rows normalized by the average row sum to preserve symmetry.

### `rollout_dynamics_w1`

Euler–Maruyama discretization of the W1 SDE (eq. 11 in the paper):

```text
x_i(s_n) = x_i(s_{n-1})
          + (dt/N) * sum_j W_ij(t_n) * phi(x_j(s_{n-1}) - x_i(s_{n-1}))
          + sigma * sqrt(dt) * dB_i
```

Returns `xs` (list of length `N_T + 1`, each `(B, N)`) and `Ws` (list of length `N_T`, each `(N, N)`).

### `sampled_cost_JNdt`

Monte Carlo estimate of the discretized cost `J^{N, dt}(theta)` (eq. 12):

```text
J = sum_n dt * Var(x^N(s_n))                          # running state cost
  + sum_n dt * (beta/2)        * ||W(s_n)||^2          # L² control cost      (modes: l2, l2_plus_uniform)
  + sum_n dt * (eta_uniform/2) * ||W(s_n) - U||^2     # uniform penalty       (modes: uniform, l2_plus_uniform)
  + lambda_terminal * Var(x^N(T))                      # terminal variance
  + gamma_target * (mean(x^N(T)) - x_target)^2        # terminal mean tracking
```

`Var` and the empirical mean are computed over the `N` agents. The active penalty terms depend on `regularizer_mode` (see below).

Returns a scalar loss and an `aux` dict with `total_cost`, `state_cost`, `l2_reg_cost`, `uniform_cost`, `terminal_var`, `terminal_mean`, `target_tracking`, `terminal_var_cost`, `terminal_target_cost`, `terminal_cost`.

### Graphon regularizers

Three helper functions implement the penalty terms:

| Function | Returns |
| --- | --- |
| `graphon_l2_penalty(W)` | `sum(W²)` — discrete approximation of `‖W‖²_{L²(I²)}` (`sum` not `mean`, because `w_ij ~ N·W_ij`) |
| `uniform_graphon_target(N, device, dtype)` | The discrete uniform target `U`: `U_ij = 1/(N−1)` for `i ≠ j`, `U_ii = 0` |
| `graphon_uniform_penalty(W)` | `sum((W − U)²)` — discrete approximation of `‖W − U‖²_{L²(I²)}` (same scaling rationale) |

**Why `1/(N−1)` and not all-ones?** The architecture enforces `W_ii = 0` and row sums of 1, so each row distributes its full weight over the `N−1` off-diagonal entries. The uniform distribution over those entries is `1/(N−1)`, not `1/N`. Using the all-ones matrix as a target would be inconsistent with the row-stochastic constraint.

`regularizer_mode` controls which penalty is active at training time:

| Mode | Active penalties |
| --- | --- |
| `"l2"` (default) | `(beta/2) * ‖W‖²` only |
| `"uniform"` | `(eta_uniform/2) * ‖W − U‖²` only |
| `"l2_plus_uniform"` | Both penalties simultaneously |

Example configs:

```python
# Standard L² regularization (default behaviour)
TrainConfig(beta=1e-2, regularizer_mode="l2")

# Push the graphon toward the uniform graphon
TrainConfig(eta_uniform=1e-2, regularizer_mode="uniform")

# Both penalties together
TrainConfig(beta=5e-3, eta_uniform=5e-3, regularizer_mode="l2_plus_uniform")
```

### Initial condition samplers

All have signature `(batch_size: int, xi: Tensor) -> Tensor` of shape `(B, N)`.

| Sampler | Base profile |
| --- | --- |
| `sample_x0` | i.i.d. Gaussian with configurable mean/std |
| `sample_x0_two_clusters` | `xi < 0.5` starts near −1, else near +1 |
| `sample_x0_structured` | `1.5*sin(2*pi*xi) + 0.8*cos(4*pi*xi)` + noise |
| `sample_x0_linear_label` | `20*(xi - 0.5)` + noise |

### `train_algorithm1_w1`

Algorithm 1 training loop: at each step, draw a fresh Monte Carlo sample `x0`, roll out dynamics, compute `sampled_cost_JNdt`, backpropagate, and update with Adam. Accepts `phi` and `x0_sampler` as arguments so different interaction functions and initial distributions can be swapped in without modifying the loop. Returns `(model, history, xi, last_aux)`.

### `run_experiment`

Convenience wrapper: trains the model, calls `save_run` to persist all artifacts, evaluates the policy on a held-out batch, and saves diagnostic figures. All output goes to `out_1/run_<timestamp>_<tag>/`.

Saved files per run:

| File | Contents |
| --- | --- |
| `config.json` | Full `TrainConfig` as JSON |
| `history.json` | All training curves |
| `final_aux.json` | Last-step cost breakdown |
| `run_summary.txt` | Human-readable summary: config, phi source, sampler name, x0 stats, final diagnostics |
| `eval_diagnostics.txt` | W statistics (row sums, symmetry error, heterogeneity) at early/mid/late time |
| `graphon_mid.png` | Heatmap of learned graphon at mid-time |
| `graphon_contour_mid.png` | Contour plot of graphon at mid-time |
| `graphon_diag_{early,mid,late}.png` | 3-panel diagnostic figures |
| `state_histogram.png` | Agent state density over time |

Running a sweep over different settings:

```python
from functools import partial

experiments = [
    dict(phi=phi_default, x0_sampler=sample_x0_two_clusters,   run_tag="tanh_clusters"),
    dict(phi=phi_default, x0_sampler=sample_x0_linear_label,   run_tag="tanh_linear"),
    dict(phi=lambda y: y, x0_sampler=sample_x0_two_clusters,   run_tag="linear_clusters"),
]
for exp in experiments:
    run_experiment(config=cfg, out_root="out_1", **exp)
```

---

## `model_1_viz.py` — Visualization Utilities

| Function | Purpose |
| --- | --- |
| `imshow_graphon(W)` | Heatmap of the N×N graphon matrix displayed as `w(xi, zeta)` over `[0,1]^2` |
| `contour_graphon(W)` | Filled contour plot of the graphon |
| `plot_slices_over_zeta(W)` | Line plots of `zeta -> w(xi, zeta)` for several fixed `xi` values |
| `plot_slices_over_xi(W)` | Line plots of `xi -> w(xi, zeta)` for several fixed `zeta` values |
| `plot_graphon_diagnostics(W)` | 3-panel figure: heatmap + zeta-slices + row/column mass profiles |
| `heterogeneity_stats(W)` | Scalar summaries: mean row entropy, top-5%-mass, L2 deviation from uniform, min/max/mean |
| `frob_norm(W)` / `dist_frob(Wa, Wb)` | Frobenius norm and pairwise distance between graphons |
| `sample_graphon_on_fine_grid(model, t, M)` | Evaluate a callable graphon on a fine M×M grid for smooth display |
| `plot_state_histogram_over_time_imshow(xs)` | Density heatmap of the agent state distribution over time |

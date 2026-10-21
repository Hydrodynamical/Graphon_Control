# Numerical conventions

The reference is *Optimal Graphon Control of Collective Dynamics*, Joseph K.
Miller and Brandon Su, October 20, 2026, especially Appendix 5. The manuscript
and its schematic PDFs are maintained separately from this code repository.

## Kernel representation

Rows are receivers, columns are senders. On the uniform grid of N agents,
`W[i,j] = wbar[i,j]/N` (equation 64). The directed controller has nonnegative
weights and unit row sums (equation 67); self-interactions are permitted.
The uniform graphon `w = 1` is `W[i,j] = 1/N`, including the diagonal.

Consequently equation 59 is implemented as

```text
x_next[i] = x[i] + dt * sum_j W[i,j] * phi(x[j] - x[i]) + sigma * delta_B[i]
```

There is no additional 1/N in the drift. Brownian increments have variance dt.
`rollout_with_Ws(..., brownian_increments=...)` accepts physical increments,
not standard normal draws. A supplied generator controls automatic draws;
deterministic rollouts consume no random numbers.

Shapes: initial states `(batch,N)`, sampled kernels `(NT,N,N)`, trajectories
`(NT+1,batch,N)`. Scalar time returns `(N,N)`; a time vector returns `(times,N,N)`,
including a vector with one entry. The horizon must satisfy `T/dt` integral;
invalid grids are rejected instead of silently rounding the endpoint.

## Objective

Equation 60 uses the population variance (`unbiased=False`), the left endpoints
of the time grid, and the kernel density penalty

```text
L2(wbar)^2 = (1/N^2) * sum_ij wbar[i,j]^2 = sum_ij W[i,j]^2
J = sum_n dt * (Var(x[n]) + beta * sum_ij W[n,i,j]^2)
    + gamma_target * (mean(x[-1]) - x_target)^2
```

`trajectory_cost` is the single implementation; `paper_cost` adapts it to the
experiment configuration. The generic trainer uses the same beta coefficient.
The old generic prototype used beta/2: to retain that historical regularization
strength, pass half the former beta. The paper experiment coefficients and
defaults are unchanged.

The generic trainer also exposes named extensions: terminal variance,
uniform-kernel penalty (eta_uniform/2), scheduled mean targets and ramped
running variance. These are disabled or configured explicitly. Checkpoint
targets are snapped to the nearest grid point, and the resolved times should
be used when interpreting them.

## Constraints and diagnostics

Directed row softmax enforces nonnegativity and unit row sums. With
`score_bound=S`, bounded scores give `N*W <= exp(2*S)`; the default `None`
preserves the original unconstrained-score experiment architecture. A fixed
attention cap is therefore not claimed for default runs across refinements.

Receiver variation is measured by equation 14 on the interval:

```text
sum_i sqrt(mean_j((N*W[i+1,j] - N*W[i,j])^2))
```

It is not a training constraint. A finite trained result does not certify a
continuum optimum or the theorem's fixed BV class. The optional symmetric
controller normalizes the average row sum, not each row, and is an exploratory
extension outside the paper's directed row-stochastic class.

The compact-support opinion law is equation 68. The Gaussian fallback has
nonzero tails, so its gap-threshold cluster count is descriptive and does not
certify exact isolation or freezing. The Kuramoto experiments use identical
natural frequencies in a rotating frame; order-parameter alignment alone does
not distinguish phase differences that are integer multiples of 2*pi.

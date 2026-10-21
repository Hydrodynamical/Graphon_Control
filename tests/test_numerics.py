"""Analytic and scaling checks for the manuscript's numerical conventions."""

import math
from types import SimpleNamespace

import pytest
import torch

from graphon_control.diagnostics import kernel_constraint_report, vxi_discrete
from graphon_control.dynamics import (
    ConstantKernel, mean_field_kernel, precompute_Ws, rollout_dynamics_w1, rollout_with_Ws,
)
from graphon_control.objectives import (
    empirical_variance, graphon_l2_penalty, graphon_uniform_penalty, paper_cost,
    resolve_target_schedule, sampled_cost_JNdt, trajectory_cost,
)
from graphon_control.validation import n_time_steps, validate_common_config


torch.set_num_threads(1)


@pytest.mark.parametrize("T,dt,expected", [(1.0, 0.02, 50), (0.3, 0.1, 3), (2, 2, 1)])
def test_integral_time_grids(T, dt, expected):
    assert n_time_steps(T, dt) == expected


@pytest.mark.parametrize("T,dt", [(0, 0.1), (1, 0), (-1, 0.1), (1, -0.1),
                                  (1, 2), (1, 0.3), (math.nan, 0.1),
                                  (1, math.inf), (math.inf, 1)])
def test_invalid_time_grid_is_rejected(T, dt):
    with pytest.raises(ValueError):
        n_time_steps(T, dt)


@pytest.mark.parametrize("field,value", [("N", 0), ("batch_size", 1.5), ("sigma", -1),
                                         ("beta", math.nan), ("lr", 0), ("grad_clip", 0),
                                         ("evaluation_seed", -1), ("evaluation_seed", True),
                                         ("score_bound", -1), ("score_bound", math.inf),
                                         ("jitter", -0.1), ("interaction_alpha", math.nan)])
def test_common_config_rejects_invalid_fields(field, value):
    cfg = SimpleNamespace(T=1.0, dt=0.1, N=4, batch_size=2, sigma=0.0, beta=1.0,
                          lr=0.01, grad_clip=1.0)
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=field):
        validate_common_config(cfg)


def test_optional_time_scale_selects_horizon_and_finite_signed_interaction_is_valid():
    cfg = SimpleNamespace(T=1.0, dt=0.1, time_scale=None, interaction_alpha=-2.0,
                          evaluation_seed=1729, score_bound=None)
    validate_common_config(cfg)


@pytest.mark.parametrize("N", [1, 2, 8, 32])
def test_uniform_graphon_scaling_is_resolution_independent(N):
    W = mean_field_kernel(N, dtype=torch.float64)
    assert torch.allclose(W.sum(-1), torch.ones(N, dtype=W.dtype))
    assert graphon_l2_penalty(W).item() == pytest.approx(1.0)
    assert graphon_uniform_penalty(W).item() == 0.0
    assert vxi_discrete(W).item() == 0.0


def test_linear_consensus_matches_exact_euler_solution():
    # For W=1/N and phi(z)=z, the mean is constant, while each deviation is
    # multiplied by (1-dt). This simultaneously checks sign and N scaling.
    x0 = torch.tensor([[1.0, 3.0, 5.0], [-2.0, 4.0, 10.0]], dtype=torch.float64)
    dt, NT = 0.1, 5
    Ws = mean_field_kernel(3, dtype=x0.dtype).expand(NT, 3, 3)
    xs = rollout_with_Ws(Ws, x0, dt, lambda y: y)
    mean = x0.mean(-1, keepdim=True)
    for n in range(NT + 1):
        expected = mean + (1 - dt) ** n * (x0 - mean)
        assert torch.allclose(xs[n], expected, atol=1e-12, rtol=0)


def test_directed_receiver_sender_orientation():
    # Both receivers listen exclusively to sender 1. Sender 1 stays still.
    W = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], dtype=torch.float64)
    x0 = torch.tensor([[0.0, 2.0]], dtype=W.dtype)
    xs = rollout_with_Ws(W, x0, 0.25, lambda z: z)
    assert torch.equal(xs[1], torch.tensor([[0.5, 2.0]], dtype=W.dtype))


def test_prescribed_brownian_increments_are_not_rescaled():
    x0 = torch.zeros((1, 2), dtype=torch.float64)
    Ws = mean_field_kernel(2, dtype=x0.dtype).expand(2, 2, 2)
    increments = torch.tensor([[[0.2, -0.1]], [[-0.3, 0.4]]], dtype=x0.dtype)
    xs = rollout_with_Ws(Ws, x0, 0.04, torch.zeros_like, sigma=2.0,
                         brownian_increments=increments)
    assert torch.allclose(xs[1:], 2.0 * increments.cumsum(0))


def test_generated_brownian_noise_has_sqrt_dt_scaling_and_is_replayable():
    x0 = torch.zeros((2, 3), dtype=torch.float64)
    Ws = mean_field_kernel(3, dtype=x0.dtype).expand(2, 3, 3)
    dt, sigma = 0.09, 1.7
    generator = torch.Generator().manual_seed(123)
    xs = rollout_with_Ws(Ws, x0, dt, torch.zeros_like, sigma, generator=generator)
    replay = torch.Generator().manual_seed(123)
    normals = torch.stack([torch.randn(x0.shape, dtype=x0.dtype, generator=replay)
                           for _ in range(2)])
    assert torch.allclose(xs[1:], sigma * math.sqrt(dt) * normals.cumsum(0))


def test_deterministic_rollout_does_not_consume_generator_state():
    generator = torch.Generator().manual_seed(9)
    state_before = generator.get_state().clone()
    x0 = torch.zeros((1, 2))
    rollout_with_Ws(mean_field_kernel(2).expand(2, 2, 2), x0, 0.1, torch.sin,
                    sigma=0, generator=generator)
    assert torch.equal(generator.get_state(), state_before)


def test_one_step_precomputation_and_legacy_rollout_agree():
    x0 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    model = ConstantKernel(mean_field_kernel(2, dtype=x0.dtype))
    Ws = precompute_Ws(model, 1, 0.1, "cpu", 3)
    assert Ws.shape == (1, 2, 2)
    assert Ws.dtype == torch.float64
    xs, Ws_list = rollout_dynamics_w1(model, x0, 0.1, 0.1, 0, torch.sin)
    assert len(xs) == 2 and len(Ws_list) == 1
    assert torch.equal(torch.stack(xs), rollout_with_Ws(Ws, x0, 0.1, torch.sin))


def test_paper_and_generic_beta_convention_with_analytic_cost():
    # Identity interactions preserve x=[-1,1]; variance=1, ||w||²=2.
    # Integral cost = T*(1+beta*2); terminal mean error is (0-target)².
    model = ConstantKernel(torch.eye(2, dtype=torch.float64))
    x0 = torch.tensor([[-1.0, 1.0]], dtype=torch.float64)
    T, dt, beta = 0.2, 0.1, 3.0
    Ws = precompute_Ws(model, 2, dt, "cpu", 1)
    xs = rollout_with_Ws(Ws, x0, dt, lambda y: y)
    cfg = SimpleNamespace(dt=dt, beta=beta, lambda_terminal=0.5, gamma_target=2.0, x_target=3.0)
    parts = paper_cost(xs, Ws, cfg)
    loss, aux = sampled_cost_JNdt(model, x0, T, dt, 0, beta, 0.5, lambda y: y,
                                 gamma_target=2.0, x_target=3.0)
    assert parts["l2_reg_cost"].item() == pytest.approx(T * beta * 2)
    assert parts["total"].item() == pytest.approx(T * (1 + beta * 2) + 0.5 + 18)
    assert loss.item() == pytest.approx(parts["total"].item())
    assert aux["total_cost"] == pytest.approx(loss.item())


def test_extra_objective_terms_have_explicit_scaling():
    x = torch.tensor([[[0.0, 2.0]]], dtype=torch.float64)
    xs = x.expand(3, 1, 2)
    Ws = torch.eye(2, dtype=x.dtype).expand(2, 2, 2)
    schedule = resolve_target_schedule(0.2, 0.1, 0, [0, 0.2], [2, 0], [3, 4])
    parts = trajectory_cost(xs, Ws, dt=0.1, beta=7, eta_uniform=6,
                            regularizer_mode="uniform", weighted_running_cost=True,
                            target_schedule=schedule)
    assert parts["state_cost"].item() == pytest.approx(0.15)
    assert parts["l2_reg_cost"].item() == 0
    assert parts["uniform_cost"].item() == pytest.approx(0.6)
    assert parts["scheduled_target_cost"].item() == pytest.approx(7)


@pytest.mark.parametrize("times,values,weights", [([0.1], [], []), ([0.1], [1], [1, 2]),
    ([math.nan], [1], [1]), ([0.3], [1], [1]), ([0.1], [math.inf], [1]),
    ([0.1], [1], [-1]), ([0.1], [1], [])])
def test_invalid_checkpoint_schedules(times, values, weights):
    with pytest.raises(ValueError):
        resolve_target_schedule(0.2, 0.1, 0, times, values, weights)


def test_checkpoint_default_weights_and_snapping():
    schedule = resolve_target_schedule(0.3, 0.1, 2, [0.14, 0.3], [1, 2], [])
    assert [target["index"] for target in schedule] == [1, 3]
    assert all(target["weight"] == 2 for target in schedule)


def test_receiver_variation_uses_graphon_values_and_no_periodic_edge():
    W = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    assert vxi_discrete(W).item() == pytest.approx(2.0)
    report = kernel_constraint_report(torch.stack([W, W]))
    assert torch.equal(report["max_row_sum_error"], torch.zeros(2, dtype=W.dtype))
    assert report["receiver_variation"].shape == (2,)


def test_variance_is_stable_under_large_state_translation():
    x = torch.tensor([[1e12 - 1, 1e12 + 1]], dtype=torch.float64)
    assert empirical_variance(x).item() == pytest.approx(1.0)


def test_differentiable_rollout_and_cost_have_finite_gradients():
    logits = torch.tensor([[0.2, -0.1], [-0.4, 0.3]], dtype=torch.float64, requires_grad=True)
    Ws = logits.softmax(-1).expand(3, 2, 2)
    x0 = torch.tensor([[-1.0, 2.0]], dtype=torch.float64)
    xs = rollout_with_Ws(Ws, x0, 0.1, torch.tanh)
    loss = trajectory_cost(xs, Ws, dt=0.1, beta=0.2)["total"]
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    # Gradcheck compares autograd to central finite differences of the full path.
    def objective(theta):
        kernels = theta.softmax(-1).expand(3, 2, 2)
        states = rollout_with_Ws(kernels, x0, 0.1, torch.tanh)
        return trajectory_cost(states, kernels, dt=0.1, beta=0.2)["total"]
    assert torch.autograd.gradcheck(objective, (logits,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("NT,chunk", [(0, 1), (1, 0), (1, 1.5)])
def test_precompute_rejects_empty_or_invalid_batches(NT, chunk):
    with pytest.raises(ValueError):
        precompute_Ws(ConstantKernel(mean_field_kernel(2)), NT, 0.1, "cpu", chunk)


def test_rollout_rejects_mismatched_noise_shape():
    with pytest.raises(ValueError, match="brownian_increments"):
        rollout_with_Ws(mean_field_kernel(2).unsqueeze(0), torch.zeros((1, 2)),
                        0.1, torch.sin, 0.2, brownian_increments=torch.zeros(2))

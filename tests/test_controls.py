import io
import math

import pytest
import torch

from graphon_control.controls import FourierFeatures, MLP, TimeLabelGraphonControl


def control(xi=None, **kwargs):
    if xi is None:
        xi = torch.linspace(0, 1, 5)
    return TimeLabelGraphonControl(
        xi, num_fourier_freqs=2, embed_dim=4, hidden_embed=7, hidden_score=9, **kwargs
    )


def test_directed_row_stochastic_with_self_interactions():
    model = control()
    W = model(torch.tensor([0.0, 0.5, 1.0]))
    assert W.shape == (3, 5, 5)
    assert torch.all(W > 0)
    assert torch.all(W.diagonal(dim1=-2, dim2=-1) > 0)
    torch.testing.assert_close(W.sum(-1), torch.ones(3, 5))


def test_time_shapes_distinguish_scalar_and_one_element_batch():
    model = control()
    scalar = model(torch.tensor(0.2))
    singleton = model(torch.tensor([0.2]))
    batch = model(torch.tensor([0.2, 0.4]))
    assert scalar.shape == (5, 5)
    assert singleton.shape == (1, 5, 5)
    assert batch.shape == (2, 5, 5)
    torch.testing.assert_close(scalar, singleton[0])
    torch.testing.assert_close(scalar, batch[0])
    torch.testing.assert_close(scalar, model(0.2))


@pytest.mark.parametrize("positive_map", ["softplus", "exp"])
def test_symmetric_alternative_has_mean_row_normalization(positive_map):
    # Fix a nonconstant score landscape so row-stochasticity cannot pass by chance.
    model = control(enforce_symmetry=True, positive_map=positive_map)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.1)
    W = model(torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(W, W.transpose(-1, -2))
    torch.testing.assert_close(W.sum(-1).mean(-1), torch.ones(2))
    assert not torch.allclose(W.sum(-1), torch.ones(2, 5))


@pytest.mark.parametrize("positive_map", ["softplus", "exp"])
def test_symmetric_normalization_handles_underflow(positive_map):
    model = control(enforce_symmetry=True, positive_map=positive_map)
    with torch.no_grad():
        model.score_net.net[-1].weight.zero_()
        model.score_net.net[-1].bias.fill_(-1000)
    W = model(0.0)
    assert torch.isfinite(W).all()
    torch.testing.assert_close(W.sum(-1).mean(), torch.tensor(1.0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_parameters_buffers_and_outputs_inherit_label_dtype(dtype):
    model = control(torch.linspace(0, 1, 4, dtype=dtype))
    for value in list(model.parameters()) + list(model.buffers()):
        assert value.dtype == dtype
        assert value.device.type == "cpu"
    assert model(0.1).dtype == dtype


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_accelerator_construction_and_time_conversion(device):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} unavailable")
    model = control(torch.linspace(0, 1, 4, device=device))
    for value in list(model.parameters()) + list(model.buffers()):
        assert value.device.type == device
    # CPU time inputs may be used with a controller constructed on an accelerator.
    W = model(torch.tensor([0.0, 0.2]))
    assert W.device.type == device
    torch.testing.assert_close(W.sum(-1).cpu(), torch.ones(2, 4))


@pytest.mark.parametrize("bound", [0.0, 0.2, 1.0])
def test_bounded_scores_imply_graphon_density_cap(bound):
    model = control(torch.linspace(0, 1, 9), score_bound=bound)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(100)
    W = model(torch.tensor([0.0, 0.5]))
    assert torch.all(9 * W <= math.exp(2 * bound) + 1e-6)
    torch.testing.assert_close(W.sum(-1), torch.ones(2, 9))
    if bound == 0:
        torch.testing.assert_close(W, torch.full_like(W, 1 / 9))


def test_checkpoint_state_names_and_roundtrip():
    model = control(score_bound=0.7)
    expected = {"xi", "psi_xi_cached", "ff.freqs"}
    for module in ("receiver_net", "sender_net", "score_net"):
        for index in (0, 2, 4):
            for kind in ("weight", "bias"):
                expected.add(f"{module}.net.{index}.{kind}")
    assert set(model.state_dict()) == expected
    expected_parameter_order = [
        f"{module}.net.{index}.{kind}"
        for module in ("receiver_net", "sender_net", "score_net")
        for index in (0, 2, 4)
        for kind in ("weight", "bias")
    ]
    assert [name for name, _ in model.named_parameters()] == expected_parameter_order
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = control(score_bound=0.7)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(model(torch.tensor([0.2, 0.8])), restored(torch.tensor([0.2, 0.8])))


@pytest.mark.parametrize("symmetric", [False, True])
def test_finite_nonzero_gradients_through_control(symmetric):
    torch.manual_seed(3)
    model = control(torch.linspace(0, 1, 5, dtype=torch.float64), enforce_symmetry=symmetric)
    times = torch.tensor([0.1, 0.7], dtype=torch.float64, requires_grad=True)
    W = model(times)
    coefficients = torch.arange(W.numel(), dtype=W.dtype).reshape_as(W)
    (W * coefficients.square()).sum().backward()
    assert times.grad is not None and torch.isfinite(times.grad).all()
    gradients = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(torch.any(g != 0) for g in gradients)


def test_label_grid_is_fixed_against_external_mutation():
    xi = torch.linspace(0, 1, 5)
    model = control(xi)
    before = model(0.0)
    xi.zero_()
    torch.testing.assert_close(model(0.0), before)
    torch.testing.assert_close(model.xi, torch.linspace(0, 1, 5))


@pytest.mark.parametrize("xi", [
    torch.tensor([], dtype=torch.float32), torch.tensor([[0.0, 1.0]]),
    torch.tensor([0, 1]), torch.tensor([0.0, float("nan")]),
    torch.tensor([0.0, float("inf")]), torch.tensor([-0.1, 0.5]),
    torch.tensor([0.0, 1.1]), torch.tensor([0.7, 0.2]), [0.0, 1.0],
])
def test_invalid_label_grids(xi):
    with pytest.raises(ValueError, match="xi"):
        control(xi)


@pytest.mark.parametrize("name,value", [
    ("num_fourier_freqs", 0), ("num_fourier_freqs", True),
    ("embed_dim", -1), ("hidden_embed", 1.5), ("hidden_score", 0),
    ("time_scale", 0), ("time_scale", -1), ("time_scale", float("inf")),
    ("time_scale", float("nan")), ("positive_map", "relu"),
    ("enforce_symmetry", "yes"), ("score_bound", -0.1),
    ("score_bound", float("inf")), ("score_bound", float("nan")),
])
def test_invalid_configuration(name, value):
    kwargs = dict(num_fourier_freqs=2, embed_dim=4, hidden_embed=7, hidden_score=9)
    kwargs[name] = value
    with pytest.raises(ValueError, match=name):
        TimeLabelGraphonControl(torch.linspace(0, 1, 5), **kwargs)


@pytest.mark.parametrize("t", [torch.tensor([]), torch.zeros(2, 1),
                                    torch.tensor(float("nan")), torch.tensor([float("inf")]),
                                    torch.tensor(1 + 2j)])
def test_invalid_times(t):
    with pytest.raises(ValueError, match="t must"):
        control()(t)


def test_scaled_times_must_be_representable():
    with pytest.raises(ValueError, match="scaled times"):
        control(time_scale=1e-300)(1.0)


def test_fourier_feature_convention():
    features = FourierFeatures(2)(torch.tensor([0.0, 0.25], dtype=torch.float64))
    expected = torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, -1.0]], dtype=torch.float64)
    torch.testing.assert_close(features, expected, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("args", [(0, 2, 1), (2, -1, 1), (2, 2, 0), (2, 2, 1, -1)])
def test_mlp_rejects_invalid_sizes(args):
    with pytest.raises(ValueError):
        MLP(*args)

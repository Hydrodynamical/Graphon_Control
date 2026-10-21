"""Scientific units and callable contracts in graphon visualizations."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from graphon_control.controls import TimeLabelGraphonControl
from graphon_control.plotting import plot_graphon_diagnostics, sample_graphon_on_fine_grid


def test_uniform_graphon_density_has_unit_row_and_column_mass():
    axes = plot_graphon_diagnostics(torch.ones(5, 5), show=False)
    try:
        row_mass, column_mass = axes[2].lines
        np.testing.assert_allclose(row_mass.get_ydata(), np.ones(5))
        np.testing.assert_allclose(column_mass.get_ydata(), np.ones(5))
    finally:
        plt.close(axes[0].figure)


def test_density_masses_integrate_receivers_and_senders_on_unit_interval():
    axes = plot_graphon_diagnostics(torch.tensor([[2.0, 0.0], [1.0, 1.0]]), show=False)
    try:
        row_mass, column_mass = axes[2].lines
        np.testing.assert_allclose(row_mass.get_ydata(), [1.0, 1.0])
        np.testing.assert_allclose(column_mass.get_ydata(), [1.5, 0.5])
    finally:
        plt.close(axes[0].figure)


def _controller():
    xi = (torch.arange(5, dtype=torch.float64) + 0.5) / 5
    return TimeLabelGraphonControl(xi, 1, 3, 4, 4)


@pytest.mark.parametrize("explicit_grid", [False, True])
def test_controller_sampler_returns_density_on_its_own_grid(explicit_grid):
    model = _controller()
    kwargs = {"M": model.N} if explicit_grid else {}
    density = sample_graphon_on_fine_grid(model, t=0.3, **kwargs)
    expected = model.N * model(0.3)
    torch.testing.assert_close(density, expected)
    assert density.shape == (model.N, model.N)
    assert density.dtype == torch.float64
    assert density.device.type == "cpu"
    assert not density.requires_grad
    torch.testing.assert_close(density.mean(-1), torch.ones(model.N, dtype=density.dtype))


def test_controller_sampler_default_time_is_zero():
    model = _controller()
    torch.testing.assert_close(sample_graphon_on_fine_grid(model), model.N * model(0.0))


def test_controller_sampler_rejects_unsupported_label_grid():
    with pytest.raises(ValueError, match="arbitrary grids are unsupported"):
        sample_graphon_on_fine_grid(_controller(), t=0.2, M=11)


def test_controller_sampler_rejects_a_time_batch_even_with_one_entry():
    with pytest.raises(ValueError, match="scalar time"):
        sample_graphon_on_fine_grid(_controller(), t=torch.tensor([0.2]))


def test_callable_sampler_evaluates_midpoint_grid_as_density():
    density = sample_graphon_on_fine_grid(lambda xi, zeta: xi + 2 * zeta, M=2)
    torch.testing.assert_close(density, torch.tensor([[0.75, 1.75], [1.25, 2.25]]))


def test_callable_internal_typeerror_is_propagated_without_retry():
    calls = []
    def broken_kernel(t, x, xi, zeta):
        calls.append((t, x))
        raise TypeError("deliberate error within kernel")
    with pytest.raises(TypeError, match="deliberate error within kernel"):
        sample_graphon_on_fine_grid(broken_kernel, t=0.2, M=3)
    assert calls == [(0.2, None)]

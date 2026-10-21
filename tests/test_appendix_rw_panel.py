"""Appendix panel checks using handwritten arrays, without an experiment."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pytest

from graphon_control import artifacts, dynamics, training
from graphon_control.experiments.kuramoto_plots import (
    ARM_COLORS, ARM_LABELS, fig_rw_panel,
)


def _curves(include_ring=False):
    t = np.array([0.0, 1.0, 2.0, 3.0])
    r = {"meanfield": np.zeros(4), "trained": np.array([0.0, 0.3, 0.8, 1.0])}
    winding = {"meanfield": np.full(4, 3.0), "trained": np.array([3.0, 2.0, 1.0, 0.0])}
    if include_ring:
        r["ring"] = np.array([0.0, 0.1, 0.1, 0.1])
        winding["ring"] = np.full(4, 3.0)
    return t, r, winding


@pytest.mark.parametrize("include_ring", [False, True])
def test_rw_panel_preserves_both_diagnostics_and_all_arms(tmp_path, monkeypatch, include_ring):
    def forbidden(*args, **kwargs):
        raise AssertionError("An archived-array panel must not run an experiment or load a checkpoint")

    monkeypatch.setattr(training, "train_controller", forbidden)
    monkeypatch.setattr(dynamics, "rollout_with_Ws", forbidden)
    monkeypatch.setattr(artifacts, "load_controller", forbidden)
    t, r, winding = _curves(include_ring)
    before = [value.copy() for value in [t, *r.values(), *winding.values()]]
    captured = []
    original_savefig = Figure.savefig

    def savefig(figure, *args, **kwargs):
        captured.append(figure)
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", savefig)
    output = tmp_path / "kur_rw_panel.pdf"
    fig_rw_panel(t, r, winding, 3, output)

    assert output.read_bytes().startswith(b"%PDF-")
    assert len(captured) == 1
    figure = captured[0]
    assert len(figure.axes) == 2
    for ax, expected in zip(figure.axes, (r, winding)):
        curves = {line.get_label(): line for line in ax.lines if line.get_label() in ARM_LABELS.values()}
        assert set(curves) == {ARM_LABELS[arm] for arm in expected}
        if ax is figure.axes[0]:
            assert {text.get_text() for text in ax.get_legend().get_texts()} == set(curves)
        else:
            assert ax.get_legend() is None
        for arm, values in expected.items():
            line = curves[ARM_LABELS[arm]]
            np.testing.assert_array_equal(line.get_xdata(), t)
            np.testing.assert_array_equal(line.get_ydata(), values)
            assert line.get_color() == ARM_COLORS[arm]
    assert figure.axes[0].get_ylabel() == r"order parameter $r(t)$"
    assert figure.axes[1].get_ylabel() == r"winding number $\mathcal{W}(t)$"
    assert not plt.fignum_exists(figure.number)
    for actual, expected in zip([t, *r.values(), *winding.values()], before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("case", [
    "invalid_k", "short_grid", "nonmonotone_grid", "nonfinite_grid",
    "not_mapping", "missing_required_arm", "mismatched_arms", "unknown_arm",
    "wrong_shape", "nonfinite_curve", "invalid_order_parameter",
])
def test_rw_panel_rejects_invalid_inputs_before_writing(tmp_path, case):
    t, r, winding = _curves()
    k = 3
    if case == "invalid_k":
        k = True
    elif case == "short_grid":
        t = np.array([0.0])
    elif case == "nonmonotone_grid":
        t[2] = t[1]
    elif case == "nonfinite_grid":
        t[1] = np.nan
    elif case == "not_mapping":
        r = []
    elif case == "missing_required_arm":
        del r["meanfield"]
        del winding["meanfield"]
    elif case == "mismatched_arms":
        del winding["meanfield"]
    elif case == "unknown_arm":
        r["other"] = np.zeros(4)
        winding["other"] = np.zeros(4)
    elif case == "wrong_shape":
        r["trained"] = np.zeros((4, 1))
    elif case == "nonfinite_curve":
        winding["trained"][1] = np.inf
    elif case == "invalid_order_parameter":
        r["trained"][1] = 1.1
    output = tmp_path / "invalid.pdf"
    figures_before = plt.get_fignums()
    with pytest.raises(ValueError):
        fig_rw_panel(t, r, winding, k, output)
    assert not output.exists()
    assert plt.get_fignums() == figures_before

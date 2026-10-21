"""Optimization behavior and guards against corrupting a training run."""

import pytest
import torch

from graphon_control.training import train_controller


torch.set_num_threads(1)


class ScalarController(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))


def test_training_decreases_known_convex_objective_and_records_metrics():
    model = ScalarController()
    def objective():
        loss = (model.weight - 2).square()
        return {"total": loss, "squared_error": loss, "constant_metric": 4.0}
    history, last = train_controller(model, objective, 20, lr=0.1, grad_clip=None)
    assert history["loss"][-1] < history["loss"][0] / 10
    assert len(history["squared_error"]) == 20
    assert history["constant_metric"] == [4.0] * 20
    assert last["total"] == history["loss"][-1]
    assert all(isinstance(value, float) for value in last.values())


def test_nonfinite_loss_is_rejected_before_update():
    model = ScalarController()
    before = model.weight.detach().clone()
    def objective():
        return {"total": model.weight * float("nan")}
    with pytest.raises(FloatingPointError, match="total"):
        train_controller(model, objective, 1, lr=0.1)
    assert torch.equal(before, model.weight)


def test_nonfinite_gradient_is_rejected_before_update():
    model = ScalarController()
    # sqrt(0) has a finite value and an infinite derivative.
    def objective():
        return {"total": torch.sqrt(model.weight)}
    with pytest.raises(FloatingPointError, match="gradient"):
        train_controller(model, objective, 1, lr=0.1)
    assert model.weight.item() == 0


def test_disconnected_objective_is_reported():
    model = ScalarController()
    def objective():
        return {"total": torch.tensor(1.0, requires_grad=True)}
    with pytest.raises(ValueError, match="disconnected"):
        train_controller(model, objective, 1, lr=0.1)


@pytest.mark.parametrize("kwargs", [{"num_steps": 0}, {"lr": 0}, {"grad_clip": -1}])
def test_invalid_training_parameters(kwargs):
    model = ScalarController()
    settings = {"num_steps": 1, "lr": 0.1, "grad_clip": 1.0}
    settings.update(kwargs)
    with pytest.raises(ValueError):
        train_controller(model, lambda: {"total": model.weight.square()}, **settings)

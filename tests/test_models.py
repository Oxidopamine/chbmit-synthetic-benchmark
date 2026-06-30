"""Detector smoke tests: shapes, gradient flow, dynamic input sizing."""
import pytest
import torch

from models import available_models, build_model, count_parameters


@pytest.mark.parametrize("name", ["eegnet", "lct", "tcn"])
def test_forward_shape_and_logit(name):
    torch.manual_seed(0)
    model = build_model(name, n_channels=18, n_samples=1024)
    x = torch.randn(8, 18, 1024)
    out = model(x)
    assert out.shape == (8,)  # single logit per sample
    assert torch.isfinite(out).all()
    assert count_parameters(model) > 0


@pytest.mark.parametrize("name", ["eegnet", "lct", "tcn"])
def test_backward_flows(name):
    model = build_model(name, n_channels=18, n_samples=1024)
    x = torch.randn(4, 18, 1024)
    y = torch.tensor([0.0, 1.0, 1.0, 0.0])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("name", ["eegnet", "lct", "tcn"])
def test_dynamic_input_length(name):
    model = build_model(name, n_channels=18, n_samples=512)
    out = model(torch.randn(2, 18, 512))
    assert out.shape == (2,)


def test_registry_lists_required_detectors():
    assert set(["eegnet", "lct", "tcn"]).issubset(set(available_models()))

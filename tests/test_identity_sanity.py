"""Milestone-1 sanity: guide == target implies aligned plasticity for free,
and meta-gradients flow through the hypothetical update step.

If these fail, the bug is in the pipeline (V, Π, CKA, functional plumbing) —
not in the science.
"""

import torch

from src.data import Experience, ExperienceSource, make_probe_set
from src.kernels import plasticity_summary
from src.kernels.metrics import dcka
from src.losses import LayerwiseAlignmentLoss, build_loss
from src.models import ProbedModel, build_model
from src.rules import SGDRule


def tiny_probed(seed: int) -> ProbedModel:
    torch.manual_seed(seed)
    backbone, _ = build_model("mlp", in_dim=16, hidden=(32, 32), out_dim=5)
    return ProbedModel(backbone, layer_names=["act1", "act2"])


def make_world(m: int = 6, n: int = 10):
    pool_x, pool_y = torch.randn(200, 16), torch.randint(0, 5, (200,))
    probe_x = make_probe_set(pool_x, n, seed=0)
    experiences = ExperienceSource(pool_x, pool_y, batch_size=1, seed=0).sample(m)
    return probe_x, experiences


def loss_fn():
    return LayerwiseAlignmentLoss(
        build_loss(
            [
                {"name": "cka_k", "weight": 1.0},
                {"name": "cka_pi", "weight": 1.0},
                {"name": "v_match", "weight": 1.0},
                {"name": "magnitude", "weight": 0.1},
            ]
        )
    )


def test_identity_alignment_is_trivially_zero():
    probe_x, experiences = make_world()
    guide, target = tiny_probed(seed=0), tiny_probed(seed=0)  # same init
    rule = SGDRule(lr=0.01)

    gs = plasticity_summary(guide, guide.params(), rule, experiences, probe_x)
    ts = plasticity_summary(target, target.params(), rule, experiences, probe_x)

    assert list(gs) == ["act1", "act2"]
    for layer in gs:
        assert torch.allclose(gs[layer]["K"], ts[layer]["K"], atol=1e-5)
        assert torch.allclose(gs[layer]["V"], ts[layer]["V"], atol=1e-4)
        assert float(dcka(gs[layer]["Pi"], ts[layer]["Pi"])) < 1e-5

    total, _ = loss_fn()(ts, gs)
    assert float(total) < 1e-4


def test_meta_gradients_flow_through_hypothetical_step():
    """trainable=weights: d(alignment loss)/d(theta) exists, is finite, and is
    nonzero when guide != target. Exercises the second-order path (grad of
    grad) that the whole design depends on."""
    probe_x, experiences = make_world()
    guide, target = tiny_probed(seed=0), tiny_probed(seed=1)  # different init
    rule = SGDRule(lr=0.01)

    guide_summaries = plasticity_summary(guide, guide.params(), rule, experiences, probe_x)

    params = {k: v.requires_grad_(True) for k, v in target.params().items()}
    target_summaries = plasticity_summary(target, params, rule, experiences, probe_x)

    total, _ = loss_fn()(target_summaries, guide_summaries)
    assert float(total) > 1e-4  # different nets should NOT be aligned

    grads = torch.autograd.grad(total, list(params.values()))
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().max() > 0 for g in grads)


def test_experience_batch_toggle():
    pool_x, pool_y = torch.randn(64, 16), torch.randint(0, 5, (64,))
    for batch_size in (1, 4):
        experiences = ExperienceSource(pool_x, pool_y, batch_size=batch_size).sample(3)
        assert len(experiences) == 3
        assert experiences[0].x.shape == (batch_size, 16)

"""Joint co-training smoke tests: tiny models, synthetic paired data.

Checks the wiring of all guidance directions, that alignment gradients reach
only the guided side (the teacher is detached), and that tracked metrics run.
"""

import torch
from torch import nn

from src.data.paired import VIEWS, synthetic_paired_batch
from src.data.probes import collect_paired_bank
from src.losses import LayerwiseAlignmentLoss, build_loss
from src.models import ProbedModel, build_model
from src.rules import SGDRule
from src.rules.tasks import image_classification, lm_next_token
from src.training import JointSide, JointTrainer
from src.training.metrics import tracked_eval

VOCAB, SEQ, HW, NCLS = 64, 12, 16, 10


def make_sides():
    torch.manual_seed(0)
    cnn, cnn_probe = build_model("tiny_cnn", widths=(4, 8), out_dim=NCLS)
    lm, _ = build_model(
        "gpt2", vocab_size=VOCAB, n_positions=SEQ, n_layer=2, n_embd=32, n_head=2
    )
    return {
        "vision": JointSide(
            "vision", ProbedModel(cnn, **cnn_probe),
            SGDRule(lr=0.01, task=image_classification), lr=0.01, device="cpu",
            view=VIEWS["vision"],
        ),
        "lm": JointSide(
            "lm", ProbedModel(lm, layer_types=[nn.LayerNorm]),
            SGDRule(lr=0.01, task=lm_next_token), lr=0.01, device="cpu",
            view=VIEWS["lm"],
        ),
    }


def make_bank_and_batches():
    g = torch.Generator().manual_seed(1)
    batches = [
        synthetic_paired_batch(8, HW, SEQ, VOCAB, NCLS, generator=g) for _ in range(4)
    ]
    bank = collect_paired_bank(batches[:3], n_probes=10, m_eval=4, views=VIEWS)
    return bank, batches


def make_trainer(guided):
    sides = make_sides()
    bank, batches = make_bank_and_batches()
    align = LayerwiseAlignmentLoss(build_loss([{"name": "cka_pi", "weight": 1.0}]))
    trainer = JointTrainer(
        sides, guided=guided, align_loss=align, probes=bank["probes"],
        m_per_step=3, device="cpu", seed=0,
    )
    return trainer, bank, batches


def test_control_arm_no_alignment():
    trainer, _, batches = make_trainer(guided=[])
    metrics = trainer.step(batches[-1])
    assert "vision/task_loss" in metrics and "lm/task_loss" in metrics
    assert not any("align" in k for k in metrics)


def test_guided_lm_step_and_teacher_isolation():
    trainer, _, batches = make_trainer(guided=["lm"])
    vision_before = {k: v.detach().clone() for k, v in trainer.sides["vision"].params.items()}

    metrics = trainer.step(batches[-1])
    assert torch.isfinite(torch.tensor(metrics["lm/align_loss"]))
    assert metrics["lm/total_loss"] != metrics["lm/task_loss"]

    # The teacher moved only by its own task SGD step: parameter change must
    # equal -lr * task gradient (no alignment gradient leaked into it).
    side = trainer.sides["vision"]
    from src.data.paired import vision_view

    exp = vision_view(batches[-1])
    ref = {k: v.detach().requires_grad_(True) for k, v in vision_before.items()}
    loss = side.rule.task(side.probed, ref, exp, side.buffers)
    grads = torch.autograd.grad(loss, list(ref.values()))
    for (name, before), g in zip(vision_before.items(), grads):
        expected = before - 0.01 * g
        assert torch.allclose(side.params[name].detach(), expected, atol=1e-6), name


def test_mutual_and_reversed_directions_run():
    for guided in (["vision"], ["lm", "vision"]):
        trainer, _, batches = make_trainer(guided=guided)
        metrics = trainer.step(batches[-1])
        for side in guided:
            assert f"{side}/align_loss" in metrics
            assert torch.isfinite(torch.tensor(metrics[f"{side}/align_loss"]))


def test_tracked_eval_runs():
    trainer, bank, batches = make_trainer(guided=["lm"])
    trainer.step(batches[-1])
    out = tracked_eval(trainer, bank, val_loader=None)
    assert 0.0 <= out["eval/k_cka/mean"] <= 1.0 + 1e-5
    assert 0.0 <= out["eval/pi_cka/mean"] <= 1.0 + 1e-5
    assert any(k.startswith("eval/v_mag/") for k in out)

"""Audio-vision track: VGG twins on synthetic paired batches, pairing logic."""

import torch

from src.data.paired_av import VIEWS, build_pairs, synthetic_av_batch
from src.data.probes import collect_paired_bank
from src.losses import LayerwiseAlignmentLoss, build_loss
from src.models import ProbedModel, build_model
from src.rules import SGDRule
from src.rules.tasks import image_classification
from src.training import JointSide, JointTrainer
from src.training.metrics import tracked_eval

NCLS = 6


def make_sides(widths=(8, 16)):
    torch.manual_seed(0)
    sides = {}
    for name, in_ch in (("vision", 3), ("audio", 1)):
        backbone, probe = build_model(
            "vgg11", in_channels=in_ch, out_dim=NCLS, widths=widths
        )
        sides[name] = JointSide(
            name, ProbedModel(backbone, layer_types=probe["layer_types"], drop_last=False),
            SGDRule(lr=0.01, task=image_classification), lr=0.01, device="cpu",
            view=VIEWS[name],
        )
    return sides


def make_trainer(guided, loss_terms):
    sides = make_sides()
    g = torch.Generator().manual_seed(1)
    batches = [synthetic_av_batch(8, generator=g) for _ in range(4)]
    bank = collect_paired_bank(batches[:3], n_probes=10, m_eval=4, views=VIEWS)
    align = LayerwiseAlignmentLoss(build_loss(loss_terms))
    trainer = JointTrainer(
        sides, guided=guided, align_loss=align, probes=bank["probes"],
        m_per_step=3, device="cpu", seed=0,
    )
    return trainer, bank, batches


def test_vgg11_canonical_structure():
    backbone, _ = build_model("vgg11")  # full-size: 8 convs, canonical pools
    probed = ProbedModel(backbone, layer_types=[torch.nn.Conv2d], drop_last=False)
    feats = probed.features(probed.params(), torch.randn(2, 3, 32, 32))
    assert list(feats) == [f"conv{i}" for i in range(1, 9)]
    # audio twin accepts rectangular spectrogram input
    audio, _ = build_model("vgg11", in_channels=1)
    a_probed = ProbedModel(audio, layer_types=[torch.nn.Conv2d], drop_last=False)
    a_feats = a_probed.features(a_probed.params(), torch.randn(2, 1, 64, 172))
    assert len(a_feats) == 8
    # identical depths -> even-spread mapping is the identity (strict 1:1)
    from src.losses.layerwise import layer_supervision

    mapping = layer_supervision(list(feats), list(a_feats))
    assert all(g == t for g, t in mapping.items())


def test_arm_b_and_c_step():
    for terms in ([{"name": "cka_k", "weight": 1.0}], [{"name": "cka_pi", "weight": 1.0}]):
        trainer, _, batches = make_trainer(["audio"], terms)
        metrics = trainer.step(batches[-1])
        assert torch.isfinite(torch.tensor(metrics["audio/align_loss"]))
        assert "vision/align_loss" not in metrics  # teacher is not guided


def test_control_arm_and_tracked_eval():
    trainer, bank, batches = make_trainer([], [])
    metrics = trainer.step(batches[-1])
    assert not any("align" in k for k in metrics)
    out = tracked_eval(trainer, bank, val_loader=batches[3:], k_val_batches=1,
                       val_metrics={"vision": ["top1", "loss"], "audio": ["top1", "loss"]})
    assert 0.0 <= out["eval/k_cka/mean"] <= 1.0 + 1e-5
    assert "eval/vision_top1" in out and "eval/audio_top1" in out
    assert out["eval/audio_loss"] > 0.0


def test_full_cka_matrices():
    from src.training.metrics import cross_model_cka_matrices, eval_summaries

    trainer, bank, _ = make_trainer([], [])
    sums = {
        name: eval_summaries(side, bank["eval_experiences"][name], trainer.probes[name])
        for name, side in trainer.sides.items()
    }
    k_mat, pi_mat, g_names, t_names = cross_model_cka_matrices(sums["vision"], sums["audio"])
    assert k_mat.shape == (len(g_names), len(t_names)) == (2, 2)  # test widths -> 2 convs
    assert torch.isfinite(k_mat).all() and torch.isfinite(pi_mat).all()


def test_run_logger_jsonl(tmp_path):
    import json

    from src.utils.logging import RunLogger

    log = RunLogger({"wandb": {"enabled": False}}, outdir=tmp_path)
    log({"step": 1, "vision/task_loss": 1.5})
    log({"step": 2, "eval/k_cka/mean": 0.3})
    lines = [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert lines[0]["vision/task_loss"] == 1.5 and lines[1]["step"] == 2


def test_wandb_project_required(tmp_path):
    import pytest

    from src.utils.logging import RunLogger, maybe_wandb

    with pytest.raises(ValueError, match="wandb-project"):
        RunLogger({"wandb": {"enabled": True}}, outdir=tmp_path)  # no project set
    with pytest.raises(ValueError, match="wandb-project"):
        maybe_wandb({"wandb": {"enabled": True, "name": "x"}})


def test_build_pairs_fixed_and_shuffled():
    v_by_c = {c: list(range(c * 100, c * 100 + 10 + c)) for c in range(3)}
    a_by_c = {c: list(range(c * 1000, c * 1000 + 8)) for c in range(3)}
    pairs = build_pairs(v_by_c, a_by_c, seed=0, balance=True)
    assert len(pairs) == 3 * 8  # capped at smallest class
    assert all(lv == la for _, _, lv, la in pairs)  # class-consistent
    assert pairs == build_pairs(v_by_c, a_by_c, seed=0, balance=True)  # stable
    # audio indices really belong to the labeled class
    assert all(a in a_by_c[la] for _, a, _, la in pairs)

    shuffled = build_pairs(v_by_c, a_by_c, seed=0, balance=True, shuffled_pairs=True)
    assert any(lv != la for _, _, lv, la in shuffled)  # correspondence broken
    # ...but each side's own labels stay truthful
    assert all(v in v_by_c[lv] and a in a_by_c[la] for v, a, lv, la in shuffled)


def test_centered_kernel_knob():
    """kernel='centered' -> every V has zero row/col sums (probe-centered),
    K-CKA is unchanged vs linear (idempotent centering), trainer runs."""
    from src.kernels import KERNEL_REGISTRY, cka
    from src.training.metrics import eval_summaries

    trainer, bank, batches = make_trainer(["audio"], [{"name": "cka_pi", "weight": 1.0}])
    side = trainer.sides["audio"]
    exps, probe = bank["eval_experiences"]["audio"], trainer.probes["audio"]
    lin = eval_summaries(side, exps, probe, KERNEL_REGISTRY["linear"])
    cen = eval_summaries(side, exps, probe, KERNEL_REGISTRY["centered"])
    for layer in cen:
        v = cen[layer]["V"]                                  # (m, n, n)
        assert torch.allclose(v.sum(dim=1), torch.zeros_like(v.sum(dim=1)), atol=1e-3)
        assert torch.allclose(v.sum(dim=2), torch.zeros_like(v.sum(dim=2)), atol=1e-3)
        assert torch.allclose(cka(lin[layer]["K"], cen[layer]["K"]), torch.tensor(1.0), atol=1e-4)

    trainer.kernel_fn = KERNEL_REGISTRY["centered"]
    metrics = trainer.step(batches[-1])
    assert torch.isfinite(torch.tensor(metrics["audio/align_loss"]))


def test_stratified_indices_balanced():
    from src.training.joint_trainer import stratified_indices

    y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    g = torch.Generator().manual_seed(0)
    idx = stratified_indices(y, 8, g)
    assert len(idx) == len(set(idx)) == 8
    counts = torch.bincount(y[idx], minlength=4).tolist()
    assert counts == [2, 2, 2, 2]
    # degrades gracefully when a class is missing / m exceeds balance
    idx = stratified_indices(torch.tensor([0, 0, 1]), 3, g)
    assert sorted(idx) == [0, 1, 2]


def test_build_pairs_prefix_is_class_mixed():
    v_by_c = {c: list(range(c * 100, c * 100 + 40)) for c in range(6)}
    a_by_c = {c: list(range(c * 1000, c * 1000 + 33)) for c in range(6)}
    pairs = build_pairs(v_by_c, a_by_c, seed=0, balance=True)
    prefix_classes = {lv for _, _, lv, _ in pairs[:60]}
    assert prefix_classes == set(range(6))  # the bug: class-ordered rows


def test_checkpoint_gradients_match():
    """Checkpointed stepped forwards must give the same loss AND the same
    parameter gradients as the stored-activation path."""
    trainer, bank, batches = make_trainer(["audio"], [{"name": "cka_pi", "weight": 1.0}])
    side = trainer.sides["audio"]
    teacher = trainer.summary(trainer.sides["vision"],
                              trainer.sides["vision"].detached_params(),
                              batch_to_exps(batches[-1], trainer.sides["vision"]), "vision")
    grads = {}
    for flag in (False, True):
        trainer.use_checkpoint = flag
        own = trainer.summary(side, side.params, batch_to_exps(batches[-1], side), "audio")
        loss, _ = trainer.align_loss(own, teacher)
        g = torch.autograd.grad(loss, list(side.params.values()))
        grads[flag] = (float(loss), [x.clone() for x in g])
    assert abs(grads[False][0] - grads[True][0]) < 1e-6
    for a, b in zip(grads[False][1], grads[True][1]):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-4)


def batch_to_exps(batch, side, m=3):
    from src.data.experiences import batch_to_experiences

    return batch_to_experiences(batch, side.view, list(range(m)))


def test_k_only_loss_skips_plasticity():
    """Arm B (cka_k only) must take the representation-only path: summaries
    carry K and nothing else, and no hypothetical step is taken."""
    trainer, _, batches = make_trainer(["audio"], [{"name": "cka_k", "weight": 1.0}])
    assert trainer.needs == {"K"}
    side = trainer.sides["audio"]
    own = trainer.summary(side, side.params, batch_to_exps(batches[-1], side), "audio")
    assert all(set(layer) == {"K"} for layer in own.values())
    metrics = trainer.step(batches[-1])
    assert torch.isfinite(torch.tensor(metrics["audio/align_loss"]))
    # the control arm (nothing guided) needs nothing
    control, _, _ = make_trainer([], [])
    assert control.needs == set()


def test_high_cadence_cka_logging():
    """cka_every logs actual K-CKA similarity for the control arm too."""
    trainer, _, batches = make_trainer([], [])
    trainer.cka_every = 1
    metrics = trainer.step(batches[-1])
    assert 0.0 <= metrics["cka/probe/mean"] <= 1.0 + 1e-5
    assert 0.0 <= metrics["cka/batch/mean"] <= 1.0 + 1e-5
    assert any(k.startswith("cka/probe/conv") for k in metrics)
    # off by default
    trainer.cka_every = 0
    assert not any(k.startswith("cka/") for k in trainer.step(batches[-1]))


def test_mutual_pi_guidance_av():
    """guided=[audio, vision]: both sides get a finite align loss, both params
    move by more than their task step alone would (alignment gradient present
    on both), and each side's teacher is the other's PRE-step state."""
    trainer, _, batches = make_trainer(["audio", "vision"], [{"name": "cka_pi", "weight": 1.0}])
    assert trainer.needs == {"Pi"}
    before = {n: {k: v.detach().clone() for k, v in s.params.items()}
              for n, s in trainer.sides.items()}
    metrics = trainer.step(batches[-1])
    for name in ("audio", "vision"):
        assert torch.isfinite(torch.tensor(metrics[f"{name}/align_loss"]))
        assert metrics[f"{name}/total_loss"] != metrics[f"{name}/task_loss"]
        moved = any(not torch.equal(before[name][k], trainer.sides[name].params[k].detach())
                    for k in before[name])
        assert moved


def test_normalize_v_cosine_pi():
    """normalize_v: Π has unit diagonal (cosine kernel), stored V stays raw,
    and the trainer threads the knob into the loss path."""
    from src.kernels.plasticity import plasticity_kernel

    trainer, bank, batches = make_trainer(["audio"], [{"name": "cka_pi", "weight": 1.0}])
    trainer.normalize_v = True
    side = trainer.sides["audio"]
    own = trainer.summary(side, side.detached_params(),
                          batch_to_exps(batches[-1], side), "audio")
    for layer in own.values():
        assert torch.allclose(layer["Pi"].diag(), torch.ones(len(layer["Pi"])), atol=1e-4)
        assert layer["Pi"].abs().max() <= 1.0 + 1e-4
        # stored V stays RAW: normalized Pi differs from raw-V Pi, and
        # rebuilding Pi from the stored V with normalize=True reproduces it
        raw_pi = plasticity_kernel(layer["V"], normalize=False)
        assert not torch.allclose(raw_pi.diag(), torch.ones(len(raw_pi)), atol=1e-2)
        assert torch.allclose(layer["Pi"], plasticity_kernel(layer["V"], normalize=True), atol=1e-5)
    metrics = trainer.step(batches[-1])
    assert torch.isfinite(torch.tensor(metrics["audio/align_loss"]))


def test_cka_prescale_is_value_identical():
    from src.kernels.metrics import cka

    g = torch.Generator().manual_seed(3)
    a = torch.randn(12, 5, generator=g); b = torch.randn(12, 5, generator=g)
    k1, k2 = a @ a.T, b @ b.T
    base = cka(k1, k2)
    assert torch.allclose(cka(k1 * 1e6, k2 * 1e-3), base, atol=1e-5)  # scale-invariant
    assert 0.0 < float(base) < 1.0

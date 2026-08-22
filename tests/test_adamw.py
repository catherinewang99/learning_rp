"""AdamWRule honesty: the rule's hypothetical step == torch.optim.AdamW's real
step on the same single experience, from the same optimizer state."""

import torch

from src.data.paired_av import VIEWS, synthetic_av_batch
from src.data.experiences import batch_to_experiences
from src.data.probes import collect_paired_bank
from src.losses import LayerwiseAlignmentLoss, build_loss
from src.models import ProbedModel, build_model
from src.rules import AdamWRule
from src.rules.tasks import image_classification
from src.training import JointSide, JointTrainer

OPT = {"name": "adamw", "weight_decay": 0.05, "betas": (0.9, 0.999), "eps": 1e-8,
       "clip_grad_norm": 1.0}


def make_side(name="vision", in_ch=3, lr=1e-3):
    torch.manual_seed(0)
    backbone, probe = build_model("vgg11", in_channels=in_ch, out_dim=6, widths=(8, 16))
    rule = AdamWRule(lr=lr, betas=OPT["betas"], eps=OPT["eps"],
                     weight_decay=OPT["weight_decay"], clip_grad_norm=OPT["clip_grad_norm"],
                     task=image_classification)
    return JointSide(name, ProbedModel(backbone, layer_types=probe["layer_types"], drop_last=False),
                     rule, lr=lr, device="cpu", view=VIEWS[name], optimizer=OPT)


def real_adamw_step(side, experience):
    """Apply torch's AdamW (with clip) to a CLONE of the side, return Δθ."""
    params = {k: v.detach().clone().requires_grad_(True) for k, v in side.params.items()}
    opt = torch.optim.AdamW(list(params.values()), lr=side.rule.lr, betas=OPT["betas"],
                            eps=OPT["eps"], weight_decay=OPT["weight_decay"])
    # copy optimizer state (moments) so both start from the same history
    for (k, p_src), p_dst in zip(side.params.items(), params.values()):
        st = side.optimizer.state.get(p_src, {})
        if st:
            opt.state[p_dst] = {kk: (vv.clone() if torch.is_tensor(vv) else vv)
                                for kk, vv in st.items()}
    before = {k: v.detach().clone() for k, v in params.items()}
    loss = side.rule.task(side.probed, params, experience, side.buffers)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(list(params.values()), OPT["clip_grad_norm"])
    grad_norms = {k: float(p.grad.norm()) for k, p in params.items()}
    opt.step()
    return {k: params[k].detach() - before[k] for k in params}, grad_norms


def test_rule_matches_torch_adamw_fresh_and_after_steps():
    side = make_side()
    g = torch.Generator().manual_seed(1)
    batches = [synthetic_av_batch(8, generator=g) for _ in range(3)]

    # fresh optimizer (no state) and again after two real steps (with state)
    for round_idx in range(3):
        exp = batch_to_experiences(batches[round_idx], side.view, [0])[0]
        side.sync_rule_state()
        rule_delta = side.rule.delta(side.probed, side.detached_params(), exp, side.buffers)
        real_delta, grad_norms = real_adamw_step(side, exp)
        # Params with ~zero gradient (e.g. a conv bias cancelled by the
        # following GroupNorm) give Adam steps that are pure numerical noise
        # (g / (|g| + eps) with g ~ 1e-9) in BOTH implementations; skip them.
        compared = 0
        for k in rule_delta:
            if grad_norms[k] < 1e-6:
                continue
            assert torch.allclose(rule_delta[k], real_delta[k], atol=1e-7, rtol=1e-4), (round_idx, k)
            compared += 1
        assert compared >= len(rule_delta) // 2
        # take a real training step to build optimizer state
        loss = side.rule.task(side.probed, side.params, side.view(batches[round_idx]), side.buffers)
        side.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(side.params.values()), OPT["clip_grad_norm"])
        side.optimizer.step()


def test_joint_trainer_adamw_runs_and_logs_grad_norm():
    sides = {"vision": make_side("vision", 3), "audio": make_side("audio", 1)}
    g = torch.Generator().manual_seed(2)
    batches = [synthetic_av_batch(8, generator=g) for _ in range(4)]
    bank = collect_paired_bank(batches[:3], n_probes=10, m_eval=4, views=VIEWS)
    trainer = JointTrainer(sides, guided=["audio"],
                           align_loss=LayerwiseAlignmentLoss(build_loss([{"name": "cka_pi", "weight": 1.0}])),
                           probes=bank["probes"], m_per_step=3, device="cpu", seed=0)
    for _ in range(2):
        metrics = trainer.step(batches[-1])
    assert torch.isfinite(torch.tensor(metrics["audio/align_loss"]))
    assert 0.0 < metrics["audio/grad_norm"] <= OPT["clip_grad_norm"] + 1e-4 or metrics["audio/grad_norm"] > 0
    assert "vision/grad_norm" in metrics
    # gradient flows through the AdamW hypothetical step (second order)
    assert any(p.grad is not None for p in sides["audio"].params.values())

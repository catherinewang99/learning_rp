"""Tracked metrics for joint experiments (the probe-experiment readouts).

Computed on FIXED banks (probes + eval experiences) so curves are comparable
across training. Everything here is measurement only — detached params, no
gradients — and shared identically by all arms:

  * cross-model layerwise K-CKA  — do the representations converge? (the
    headline curve; the control arm gives the baseline "platonic" drift)
  * cross-model layerwise Π-CKA — do the learning dynamics converge?
  * per-side task validation (top-1 accuracy or perplexity)
"""

from __future__ import annotations

import math

import torch

from ..kernels.gram import linear_gram
from ..kernels.metrics import cka, response_magnitudes
from ..kernels.plasticity import plasticity_summary
from ..losses.layerwise import layer_supervision


@torch.no_grad()
def _val_batches(loader, k: int, device):
    for i, batch in enumerate(loader):
        if i >= k:
            break
        yield {key: v.to(device) for key, v in batch.items()}


def cross_model_alignment(
    guide_summaries: dict, target_summaries: dict, upper_half: bool = False
) -> dict[str, float]:
    """Layerwise K-CKA and Π-CKA between two models' summaries (higher =
    more aligned; these are similarities, not the loss's dissimilarities)."""
    mapping = layer_supervision(list(guide_summaries), list(target_summaries), upper_half)
    out: dict[str, float] = {}
    k_vals, pi_vals = [], []
    for g_layer, t_layer in mapping.items():
        k_sim = float(cka(guide_summaries[g_layer]["K"], target_summaries[t_layer]["K"]))
        pi_sim = float(cka(guide_summaries[g_layer]["Pi"], target_summaries[t_layer]["Pi"]))
        out[f"k_cka/{g_layer}->{t_layer}"] = k_sim
        out[f"pi_cka/{g_layer}->{t_layer}"] = pi_sim
        k_vals.append(k_sim)
        pi_vals.append(pi_sim)
    out["k_cka/mean"] = sum(k_vals) / len(k_vals)
    out["pi_cka/mean"] = sum(pi_vals) / len(pi_vals)
    return out


def cross_model_k_cka(
    guide_summaries: dict, target_summaries: dict, upper_half: bool = False
) -> dict[str, float]:
    """Layerwise K-CKA only (works on representation_summary output). Cheap
    enough to log at high cadence during training, for every arm."""
    mapping = layer_supervision(list(guide_summaries), list(target_summaries), upper_half)
    out = {
        f"{g}->{t}": float(cka(guide_summaries[g]["K"], target_summaries[t]["K"]))
        for g, t in mapping.items()
    }
    out["mean"] = sum(out.values()) / len(out)
    return out


def eval_summaries(side, eval_experiences, probe_x, kernel_fn=linear_gram,
                   normalize_v: bool = False) -> dict:
    """Plasticity summary at the side's current (detached) params on the fixed
    eval bank. Detached params -> no outer graph retained."""
    return plasticity_summary(
        side.probed, side.detached_params(), side.rule,
        eval_experiences, probe_x, side.buffers, kernel_fn,
        normalize_v=normalize_v,
    )


@torch.no_grad()
def val_top1(side, batches) -> float:
    """Classification accuracy through the side's own view (exp.y = labels)."""
    correct = total = 0
    for batch in batches:
        exp = side.view(batch)
        logits = side.probed.forward_output(side.detached_params(), exp.x, side.buffers)
        correct += int((logits.argmax(-1) == exp.y).sum())
        total += exp.y.shape[0]
    return correct / max(total, 1)


@torch.no_grad()
def val_task_loss(side, batches) -> float:
    """Mean task loss on val batches (the crossmodal 'val_loss' analog)."""
    losses = [
        float(side.rule.task(side.probed, side.detached_params(), side.view(b), side.buffers))
        for b in batches
    ]
    return sum(losses) / max(len(losses), 1)


def val_ppl(side, batches) -> float:
    """exp(mean task loss) — perplexity for LM sides."""
    return math.exp(val_task_loss(side, batches))


VAL_METRIC_FNS = {"top1": val_top1, "ppl": val_ppl, "loss": val_task_loss}


def cross_model_cka_matrices(guide_summaries: dict, target_summaries: dict):
    """FULL L_guide x L_target CKA matrices (not just the mapped pairs) for
    heatmap visualization — off-diagonal structure shows whether e.g. deep
    audio layers align with shallow vision layers. Returns
    (k_mat, pi_mat, guide_layer_names, target_layer_names)."""
    g_names, t_names = list(guide_summaries), list(target_summaries)
    k_mat = torch.zeros(len(g_names), len(t_names))
    pi_mat = torch.zeros(len(g_names), len(t_names))
    for i, g in enumerate(g_names):
        for j, t in enumerate(t_names):
            k_mat[i, j] = float(cka(guide_summaries[g]["K"], target_summaries[t]["K"]))
            pi_mat[i, j] = float(cka(guide_summaries[g]["Pi"], target_summaries[t]["Pi"]))
    return k_mat, pi_mat, g_names, t_names


def tracked_eval(
    trainer,
    bank: dict,             # from data.probes.collect_paired_bank
    val_loader=None,
    k_val_batches: int = 8,
    upper_half: bool = False,
    guide_name: str | None = None,
    val_metrics: dict[str, str | list[str]] | None = None,
    #   side name -> metric kind(s): "top1" | "ppl" | "loss"
) -> dict[str, float]:
    """The full periodic measurement pass; keys are wandb-ready.

    Cross-model CKA is reported as guide->target. Default guide: the teacher
    of the first guided side; for the control arm (nothing guided), the first
    side in the sides dict.
    """
    device = trainer.device
    sums = {}
    for name, side in trainer.sides.items():
        experiences = [e.to(device) for e in bank["eval_experiences"][name]]
        sums[name] = eval_summaries(side, experiences, trainer.probes[name],
                                    getattr(trainer, "kernel_fn", linear_gram),
                                    getattr(trainer, "normalize_v", False))

    if guide_name is None:
        guide_name = (
            trainer._teacher_of(trainer.guided[0]) if trainer.guided else list(trainer.sides)[0]
        )
    target_name = next(n for n in trainer.sides if n != guide_name)

    out = {
        f"eval/{k}": v
        for k, v in cross_model_alignment(sums[guide_name], sums[target_name], upper_half).items()
    }
    for name in trainer.sides:
        for layer, s in sums[name].items():
            out[f"eval/v_mag/{name}/{layer}"] = float(response_magnitudes(s["V"]).mean())

    if val_loader is not None and val_metrics:
        batches = list(_val_batches(val_loader, k_val_batches, device))
        for name, kinds in val_metrics.items():
            for kind in [kinds] if isinstance(kinds, str) else kinds:
                out[f"eval/{name}_{kind}"] = VAL_METRIC_FNS[kind](trainer.sides[name], batches)
    return out

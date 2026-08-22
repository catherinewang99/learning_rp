"""V(X|e): the representational response to a single experience, per layer.

    V_l(X|e) = ( K_l(θ + Δ_R θ(e)) − K_l(θ) ) / η        for each probed layer l

"How does one hypothetical update on experience e change the geometry of the
probe set, at every depth of the network?" This is the atom everything else
(Π, all loss terms) is built from.

Cost note (why layerwise is cheap): the hypothetical step (one backward) and
the stepped probe forward are shared across ALL layers — extra layers only add
n x n kernel algebra, not extra forwards/backwards.
"""

from __future__ import annotations

import torch

from .gram import linear_gram


def grams(features: dict[str, torch.Tensor], kernel_fn=linear_gram) -> dict[str, torch.Tensor]:
    """Per-layer kernels {layer: (n, n)} from per-layer features {layer: (n, d)}."""
    return {name: kernel_fn(h) for name, h in features.items()}


def stepped_grams(probed, stepped, probe_x, buffers, kernel_fn, use_checkpoint=False):
    """Kernels of the probe set under hypothetical params ``stepped``.

    With use_checkpoint (and params that require grad), the probe forward is
    NOT stored for backward — it is recomputed during backward (non-reentrant
    torch.utils.checkpoint). This is THE memory lever for Π: per experience
    only the update graph + the n x n kernels stay resident instead of a full
    forward's activations. ~1.5x compute on that forward; gradients identical
    (pinned by tests/test_av_smoke.py::test_checkpoint_gradients_match).
    """
    names = list(stepped)
    layer_order = probed.layer_order

    def run(*values):
        feats = probed.features(dict(zip(names, values)), probe_x, buffers)
        return tuple(kernel_fn(feats[n]) for n in layer_order)

    values = [stepped[n] for n in names]
    if use_checkpoint and any(v.requires_grad for v in values):
        from torch.utils.checkpoint import checkpoint

        outs = checkpoint(run, *values, use_reentrant=False)
    else:
        outs = run(*values)
    return dict(zip(layer_order, outs))


def representational_response(
    probed,
    params: dict[str, torch.Tensor],
    rule,
    experience,
    probe_x,                      # tensor or dict (modality-specific probe input)
    buffers=None,
    kernel_fn=linear_gram,
    k0: dict[str, torch.Tensor] | None = None,
    use_checkpoint: bool = False,
) -> dict[str, torch.Tensor]:
    """Returns {layer: V (n, n)}. Differentiable w.r.t. params and rule params.

    ``k0`` (pre-update kernels) can be passed in to avoid recomputing them for
    every experience in a batch — see plasticity.response_batch.
    """
    if k0 is None:
        k0 = grams(probed.features(params, probe_x, buffers), kernel_fn)
    delta = rule.delta(probed, params, experience, buffers)
    stepped = {k: params[k] + delta[k] for k in params}
    k1 = stepped_grams(probed, stepped, probe_x, buffers, kernel_fn, use_checkpoint)
    return {name: (k1[name] - k0[name]) / rule.step_size for name in k0}

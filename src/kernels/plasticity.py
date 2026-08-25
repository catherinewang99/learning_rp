"""The plasticity kernel Π and the per-state, per-layer summary dict.

Π_l(e_i, e_j) = ⟨vec V_l(X|e_i), vec V_l(X|e_j)⟩ — an (m, m) kernel over
experiences at layer l: which experiences move that layer's representation in
similar ways. A representation of the learning dynamics themselves.
"""

from __future__ import annotations

import torch

from .gram import linear_gram
from .response import grams, representational_response


def response_batch(
    probed,
    params,
    rule,
    experiences: list,
    probe_x,
    buffers=None,
    kernel_fn=linear_gram,
    use_checkpoint: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """V for each experience at every layer.

    Returns (K0: {layer: (n,n)}, V: {layer: (m,n,n)}).

    Plain loop over experiences: m hypothetical steps, K0 computed once.
    TODO(scale): chunk the loop if m or n grow large (hooks preclude vmap
    over the probe forward — chunk, don't vmap).
    """
    k0 = grams(probed.features(params, probe_x, buffers), kernel_fn)
    vs = [
        representational_response(
            probed, params, rule, e, probe_x, buffers, kernel_fn, k0=k0,
            use_checkpoint=use_checkpoint,
        )
        for e in experiences
    ]
    v = {name: torch.stack([vi[name] for vi in vs]) for name in k0}
    return k0, v


def plasticity_kernel(v: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """Π = (vec V)(vec V)ᵀ over experiences. v: (m, n, n) -> (m, m).

    normalize=True builds Π from unit-Frobenius V̂ (the TTU-style per-sample
    row normalization applied to plasticity's "samples" = experiences):
    Π becomes a cosine kernel — unit diagonal, entries in [-1, 1] — so no
    single large-response experience dominates the CKA. Project default for
    Π-guidance arms (plasticity.normalize_v); magnitude info lives in the
    separate MagnitudeTerm, which reads the RAW V."""
    flat = v.flatten(1)
    if normalize:
        flat = flat / (flat.norm(dim=1, keepdim=True) + 1e-12)
    return flat @ flat.T


def representation_summary(
    probed, params, probe_x, buffers=None, kernel_fn=linear_gram
) -> dict[str, dict[str, torch.Tensor]]:
    """K only: {layer: {"K": (n,n)}}. One probe forward, no hypothetical steps.
    Used when every loss term needs only K (e.g. the K-CKA control arm) —
    same interface as plasticity_summary, a fraction of the cost."""
    return {name: {"K": k} for name, k in grams(probed.features(params, probe_x, buffers), kernel_fn).items()}


def plasticity_summary(
    probed, params, rule, experiences, probe_x, buffers=None, kernel_fn=linear_gram,
    use_checkpoint: bool = False, normalize_v: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    """Everything the loss terms need at one network state, per probed layer:

        {layer_name: {"K": (n,n), "V": (m,n,n), "Pi": (m,m)}}

    Layer names are in execution order (dicts preserve it). This nested dict is
    the sole interface between the measurement stack and losses/ — loss terms
    consume one layer's inner dict; losses/layerwise.py handles the mapping
    between two models' layer sets. Add fields to the inner dict when new loss
    terms need new quantities.
    """
    k0, v = response_batch(probed, params, rule, experiences, probe_x, buffers, kernel_fn,
                           use_checkpoint)
    return {
        name: {"K": k0[name], "V": v[name],
               "Pi": plasticity_kernel(v[name], normalize=normalize_v)}
        for name in k0
    }

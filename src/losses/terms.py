"""Composable alignment loss terms.

Every term maps (target_summary, guide_summary) -> scalar, where a summary is
the dict produced by kernels.plasticity_summary: {"K", "V", "Pi"}.
New objective idea == new class here + a registry entry; nothing else changes.
"""

from __future__ import annotations

import torch

from ..kernels.metrics import dcka, response_magnitudes, EPS


class CKAPiTerm:
    """d_CKA(Π_target, Π_guide): match the global organization of plastic
    responses — which experiences move the network in similar ways."""

    needs = {"Pi"}   # summary fields this term reads (trainer skips the rest)

    def __call__(self, target: dict, guide: dict) -> torch.Tensor:
        return dcka(target["Pi"], guide["Pi"])


class CKAKTerm:
    """d_CKA(K_target, K_guide): the 0th-order Optimization-Prior term.
    Under trajectory matching this doubles as target-state placement."""

    needs = {"K"}

    def __call__(self, target: dict, guide: dict) -> torch.Tensor:
        return dcka(target["K"], guide["K"])


class VMatchTerm:
    """E_e ‖V̂_t(e) − V̂_g(e)‖²_F on unit-normalized responses: match each
    experience's response *direction* (the PDF's second loss component)."""

    needs = {"V"}

    def __call__(self, target: dict, guide: dict) -> torch.Tensor:
        vt, vg = target["V"].flatten(1), guide["V"].flatten(1)
        vt = vt / (vt.norm(dim=1, keepdim=True) + EPS)
        vg = vg / (vg.norm(dim=1, keepdim=True) + EPS)
        return ((vt - vg) ** 2).sum(dim=1).mean()


class MagnitudeTerm:
    """E_e log²(‖V_t(e)‖ / ‖V_g(e)‖): preserve response scale, which CKA and
    V̂ deliberately discard. Log-ratio so 1000x-too-slow and 1000x-too-fast
    are penalized symmetrically."""

    needs = {"V"}

    def __call__(self, target: dict, guide: dict) -> torch.Tensor:
        mt = response_magnitudes(target["V"])
        mg = response_magnitudes(guide["V"])
        return (torch.log((mt + EPS) / (mg + EPS)) ** 2).mean()


TERM_REGISTRY = {
    "cka_pi": CKAPiTerm,
    "cka_k": CKAKTerm,
    "v_match": VMatchTerm,
    "magnitude": MagnitudeTerm,
    # "diffknn_pi": parked — neighborhood-overlap objective on Π (see metrics.py)
}

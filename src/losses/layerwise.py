"""Layerwise alignment: map guide layers onto target layers, sum loss terms.

Ported conventions from crossmodal-prior / training-the-untrainable:
even-spread mapping of guide layers across target layers (by depth order),
with an upper_half option restricting supervision to the deeper half of the
target. Layer lists come from plasticity_summary key order (execution order).
"""

from __future__ import annotations

from collections import Counter

import torch


def layer_supervision(
    guide_layers: list[str],
    target_layers: list[str],
    upper_half: bool = False,
) -> dict[str, str]:
    """Map guide layer names -> target layer names by even spreading."""
    if upper_half:
        target_layers = target_layers[len(target_layers) // 2:]
    if not guide_layers or not target_layers:
        raise ValueError("no layers to map (check layer_types / probe config)")
    n_t, n_g = len(target_layers), len(guide_layers)
    step = (n_t - 1) / (n_g - 1) if n_g > 1 else 1
    return {
        g: target_layers[min(round(i * step), n_t - 1)]
        for i, g in enumerate(guide_layers)
    }


class LayerwiseAlignmentLoss:
    """Apply a CompositeLoss to every mapped (guide layer, target layer) pair
    and average, so total scale is independent of network depth.

    __call__(target_summaries, guide_summaries) -> (total, parts) where the
    summaries are plasticity_summary dicts {layer: {"K","V","Pi"}} and parts
    is {f"{g}->{t}/{term}": value} for logging. When several guide layers map
    to one target layer, each pair still contributes once (multiplicity is
    reported for diagnostics, matching crossmodal's accounting).
    """

    def __init__(self, composite, upper_half: bool = False):
        self.composite = composite
        self.upper_half = upper_half

    @property
    def needs(self) -> set[str]:
        return self.composite.needs

    def __call__(self, target_summaries: dict, guide_summaries: dict):
        mapping = layer_supervision(
            list(guide_summaries), list(target_summaries), self.upper_half
        )
        multiplicity = Counter(mapping.values())
        total = 0.0
        parts: dict[str, torch.Tensor] = {}
        for g_layer, t_layer in mapping.items():
            pair_total, pair_parts = self.composite(
                target_summaries[t_layer], guide_summaries[g_layer]
            )
            total = total + pair_total
            for term, value in pair_parts.items():
                parts[f"{g_layer}->{t_layer}/{term}"] = value
        parts["n_pairs"] = torch.tensor(float(len(mapping)))
        parts["max_multiplicity"] = torch.tensor(float(max(multiplicity.values())))
        return total / max(len(mapping), 1), parts

"""Weighted sum of loss terms, assembled from config."""

from __future__ import annotations

import torch

from .terms import TERM_REGISTRY


class CompositeLoss:
    """loss(target_summary, guide_summary) -> (total, {name: value}).

    The per-term dict is returned unweighted for logging/diagnostics.
    """

    def __init__(self, terms: list[tuple[str, float, object]]):
        self.terms = terms  # (name, weight, callable)

    @property
    def needs(self) -> set[str]:
        """Union of summary fields the terms read ({"K"} alone => cheap path)."""
        out: set[str] = set()
        for _, _, fn in self.terms:
            out |= getattr(fn, "needs", {"K", "V", "Pi"})
        return out

    def __call__(self, target: dict, guide: dict):
        parts: dict[str, torch.Tensor] = {}
        total = 0.0
        for name, weight, fn in self.terms:
            value = fn(target, guide)
            parts[name] = value.detach()
            total = total + weight * value
        return total, parts


def build_loss(loss_cfg: list[dict]) -> CompositeLoss:
    """loss_cfg: [{"name": "cka_pi", "weight": 1.0}, ...] (see configs/)."""
    terms = []
    for entry in loss_cfg:
        name = entry["name"]
        terms.append((name, float(entry.get("weight", 1.0)), TERM_REGISTRY[name]()))
    return CompositeLoss(terms)

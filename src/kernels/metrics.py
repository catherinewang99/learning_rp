"""Similarity metrics between kernels. Pure functions.

Used both on representation kernels K (n, n) and plasticity kernels Π (m, m).
"""

from __future__ import annotations

import torch

from .gram import center_gram

EPS = 1e-12


def cka(k1: torch.Tensor, k2: torch.Tensor) -> torch.Tensor:
    """Linear CKA between two square kernels of the same size. Scale-invariant.

    Inputs are pre-scaled to unit Frobenius norm before centering: the CKA
    value is mathematically unchanged (scale invariance), but intermediate
    sums stay O(1) instead of e.g. 1e24 for raw plasticity kernels — a
    float32 precision guard, not a modeling choice."""
    k1 = k1 / (k1.norm() + EPS)
    k2 = k2 / (k2.norm() + EPS)
    k1c, k2c = center_gram(k1), center_gram(k2)
    hsic = (k1c * k2c).sum()
    return hsic / (k1c.norm() * k2c.norm() + EPS)


def dcka(k1: torch.Tensor, k2: torch.Tensor) -> torch.Tensor:
    """CKA distance in [0, 2] (≈ [0, 1] for PSD kernels)."""
    return 1.0 - cka(k1, k2)


def response_magnitudes(v: torch.Tensor) -> torch.Tensor:
    """Per-experience Frobenius norms ‖V(X|e)‖_F. v: (m, n, n) -> (m,).

    CKA removes scale, but a rule that moves in the right directions 1000x
    too slowly is not dynamically equivalent (PDF §1) — magnitudes are
    matched by a separate loss term.
    """
    return v.flatten(1).norm(dim=1)


# Parked for later: DiffKNN-style neighborhood overlap on Π ("which experiences
# are local neighbors in plasticity space"), per the PDF's second objective.

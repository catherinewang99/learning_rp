"""Representation kernels K(H) and feature preprocessing. Pure functions.

H is (n, d) — n probe stimuli, d (pooled) features. K is (n, n).
"""

from __future__ import annotations

import torch


def linear_gram(h: torch.Tensor) -> torch.Tensor:
    """K = H Hᵀ (the PDF's choice)."""
    return h @ h.T


def center_gram(k: torch.Tensor) -> torch.Tensor:
    """Double-centered kernel: J K J with J = I - 11ᵀ/n (used by CKA/HSIC)."""
    n = k.shape[0]
    j = torch.eye(n, device=k.device, dtype=k.dtype) - 1.0 / n
    return j @ k @ j


def normalize_rows(h: torch.Tensor, center: bool = False) -> torch.Tensor:
    """crossmodal-prior preprocessing: optionally center features over the
    batch, then scale each sample vector to unit norm. Composing with
    linear_gram gives a cosine-style kernel. NOTE: applying this changes what
    V measures (change of *directional* geometry, raw scale discarded) — keep
    it off when the magnitude term matters."""
    if center:
        h = h - h.mean(dim=0, keepdim=True)
    return h / (h.norm(dim=1, keepdim=True) + 1e-8)


def row_normalized_gram(h: torch.Tensor) -> torch.Tensor:
    """Drop-in kernel_fn variant matching crossmodal-prior's K preprocessing."""
    return linear_gram(normalize_rows(h))


def centered_gram(h: torch.Tensor) -> torch.Tensor:
    """K = J H Hᵀ J: the Gram of mean-centered features.

    Using this as kernel_fn makes V = ΔK/η PROBE-CENTERED (J commutes with the
    difference), so Π is built only from changes in the *relative* geometry of
    the probes — the constant (mean-norm) and row/column (per-probe offset)
    components of ΔK are removed before the experience-space inner products.
    K-CKA is unaffected (CKA's own centering is idempotent); Π-CKA and the
    magnitude diagnostics change. See README "centered vs linear".
    """
    return center_gram(linear_gram(h))


# Config-selectable kernels (plasticity.kernel in configs/)
KERNEL_REGISTRY = {
    "linear": linear_gram,            # PDF's literal K = HHᵀ (default)
    "centered": centered_gram,        # probe-centered V / relational Π
    "row_normalized": row_normalized_gram,  # crossmodal-prior preprocessing
}


# Ideas parked for later (drop-in kernel_fn replacements): rbf_gram(h, sigma)

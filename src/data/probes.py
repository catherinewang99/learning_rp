"""Probe sets and fixed evaluation banks.

The probe set X is the fixed input set on which all kernels are computed. It
must be identical (same samples, same order) for every model being compared
and fixed for the lifetime of an experiment — K, V, Π are only comparable on
a shared X. For paired experiments X is a set of paired samples: each model
sees its own modality view of the same n concepts.

Measurement stability (project decision): the training loss may use each
step's minibatch as experiences, but *tracked metrics* use the fixed banks
built here, so CKA-at-step-1k is comparable to CKA-at-step-50k.
"""

from __future__ import annotations

import torch

from .experiences import Experience, batch_to_experiences


def make_probe_set(x: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
    """Draw n fixed probe inputs from a tensor pool. Deterministic given seed."""
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(x), generator=generator)[:n]
    return x[idx].clone()


def collect_paired_bank(val_loader, n_probes: int, m_eval: int, views: dict) -> dict:
    """Accumulate the first n_probes + m_eval val samples into fixed banks.

    ``views`` maps side name -> view fn (e.g. paired.VIEWS or paired_av.VIEWS).
    Returns {
      "probes":   {side: probe inputs (n, ...) per that side's modality},
      "eval_experiences": {side: [Experience] * m_eval},
    }
    Uses the val split (deterministic transform, held out from training).
    """
    batches = []
    total = 0
    for batch in val_loader:
        batches.append(batch)
        total += next(iter(batch.values())).shape[0]
        if total >= n_probes + m_eval:
            break
    if total < n_probes + m_eval:
        raise ValueError(f"val split too small: {total} < {n_probes + m_eval}")
    merged = {k: torch.cat([b[k] for b in batches])[: n_probes + m_eval] for k in batches[0]}

    probe_slice = {k: v[:n_probes] for k, v in merged.items()}
    eval_slice = {k: v[n_probes:] for k, v in merged.items()}
    return {
        "probes": {name: view(probe_slice).x for name, view in views.items()},
        "probe_labels": {name: view(probe_slice).y for name, view in views.items()},
        "eval_experiences": {
            name: batch_to_experiences(eval_slice, view) for name, view in views.items()
        },
    }

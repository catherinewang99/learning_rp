"""Downstream evaluation (phase 4) — the success criterion.

STUBS — implement at milestone 4.

The claim to test (phase B): a target trained from a plasticity-aligned init
learns *more like the guide* than controls do. Conditions:
  1. plasticity-aligned init (K + Π matched)      <- ours
  2. K-aligned init only (Optimization Prior)     <- key control
  3. random init                                  <- baseline
Metrics logged from day one so runs are comparable:
  * task learning curves (loss/acc vs steps)
  * kernel-trajectory divergence: d_CKA(K_target(t), K_guide(t)) over training
  * plasticity persistence: d_CKA(Π_target(t), Π_guide(t)) over training —
    measures how fast the 1st-order prior decays under real SGD (an actual
    empirical question, not a nuisance).
"""

from __future__ import annotations

import torch


def train_and_track(
    probed_target,
    init_params: dict[str, torch.Tensor],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    guide_bank,
    experiences: list,
    lr: float,
    steps: int,
    eval_every: int = 100,
) -> list[dict]:
    """Train normally from ``init_params``; periodically log the metrics above."""
    raise NotImplementedError("milestone 4")


def compare_inits(results_by_condition: dict[str, list[dict]]) -> dict:
    """Aggregate/plot the three-condition comparison."""
    raise NotImplementedError("milestone 4")

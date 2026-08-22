"""Train the guide and checkpoint its trajectory (phase 1).

STUB — standard supervised training; implement at milestone 2/3.

Design notes:
  * Returns/saves a list of parameter dicts θ_g(t) at checkpoint times
    t = 0, c, 2c, ..., T (t=0 included: the guide's init is a checkpoint).
  * Checkpoint *times* should be logged in optimizer steps, so that guide
    "experience time" can be compared across guides.
  * PINNED (per discussion): where the guide starts — pretrained vs scratch —
    is an open experimental question; both should be supported via config.
"""

from __future__ import annotations

import torch


def train_guide(
    probed_guide,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    lr: float,
    steps: int,
    checkpoint_every: int,
    batch_size: int = 128,
    seed: int = 0,
) -> list[dict[str, torch.Tensor]]:
    """Plain SGD + cross-entropy training loop; returns checkpoint param dicts.

    (Ordinary stateful training is fine here — functional purity is only
    required where meta-gradients flow, i.e. on the target side.)
    """
    raise NotImplementedError("milestone 2: standard supervised loop + checkpointing")

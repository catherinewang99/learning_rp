"""LearningRule interface.

A rule is a *functional* map: (params, experience) -> parameter update Δθ.

Contract (this is what makes the whole project work — do not break it):
  * ``delta`` is pure: no in-place mutation of params/module, no optimizer state.
  * ``delta`` is differentiable w.r.t. BOTH
      (a) ``params``            -> enables trainable=weights (phase B), and
      (b) the rule's own learnable parameters -> enables trainable=rule (phase A).
    Second-order gradients flow through the hypothetical step in
    kernels/response.py; anything stateful or in-place silently breaks that.

The rule ladder (see ROADMAP): L0 SGDRule -> L1 PerLayerLRRule ->
L2 gated/preconditioned -> L3 blackbox local (bio-plausible rules slot in here).
Every rung implements this same interface; nothing else in the codebase changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class LearningRule(ABC):
    @abstractmethod
    def delta(
        self,
        probed,
        params: dict[str, torch.Tensor],
        experience,
        buffers: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return Δθ as a name -> tensor dict matching ``params``."""

    @property
    @abstractmethod
    def step_size(self) -> float:
        """Nominal step size η, used to normalize V = ΔK / η."""

    def learnable_parameters(self) -> list[torch.Tensor]:
        """Rule parameters φ to optimize under trainable=rule. L0 rules: none."""
        return []

"""Levels 1-2 of the rule ladder: backprop-anchored learnable rules (phase A).

STUBS — signatures and semantics are settled; implement when phase A starts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import LearningRule


class PerLayerLRRule(LearningRule):
    """Level 1: Δθ_l = -exp(φ_l) * g_l, one log-learning-rate per param tensor.

    Contains SGDRule exactly (all φ_l = log lr), giving the known-solution
    test for the phase-A meta-optimization pipeline: with guide == target,
    initializing at SGD must already put the alignment loss ≈ 0 and the
    meta-gradient ≈ 0.
    """

    def __init__(self, param_names: list[str], init_lr: float = 0.01, loss_fn=F.cross_entropy):
        self.loss_fn = loss_fn
        self.log_lrs = {
            name: torch.tensor(float(torch.log(torch.tensor(init_lr))), requires_grad=True)
            for name in param_names
        }
        self._init_lr = init_lr

    @property
    def step_size(self) -> float:
        return self._init_lr

    def learnable_parameters(self):
        return list(self.log_lrs.values())

    def delta(self, probed, params, experience, buffers=None):
        raise NotImplementedError(
            "Phase A: like SGDRule.delta but scale each grads[k] by "
            "-exp(self.log_lrs[k]). Keep everything functional."
        )


class GatedGradRule(LearningRule):
    """Level 2: Δθ = -M_φ(local signals) ⊙ g.

    M_φ is a small network reading e.g. |g|, activation stats, layer depth.
    TODO(phase A): define the feature set and the gate parameterization.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

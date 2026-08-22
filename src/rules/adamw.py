"""AdamW as a functional LearningRule (honesty-preserving).

The hypothetical update for one experience e is the EXACT update that
torch.optim.AdamW would apply if e were the whole batch, given the
optimizer's current moment estimates:

    g      = clip(∇_θ task(e))                      (global-norm clipping, optional)
    m'     = β1 m + (1-β1) g
    v'     = β2 v + (1-β2) g²
    Δθ     = -lr·wd·θ  -  (lr / (1-β1^t)) · m' / ( sqrt(v') / sqrt(1-β2^t) + eps )

which mirrors torch's implementation step-for-step (decoupled weight decay
applied to θ before the Adam step; bias corrections with t = step + 1).
The moments (m, v, step) are READ from the live optimizer each training
step via ``sync_state`` and never mutated here, so V stays a pure function
of (θ, e, optimizer state). Differentiable w.r.t. θ through g (second order).

Note: V = ΔK/η uses η = lr as the nominal step size; Adam's effective
per-parameter step differs, which CKA on Π is invariant to (scale), while the
magnitude diagnostics will reflect it.
"""

from __future__ import annotations

import math

import torch
from torch.func import grad

from .base import LearningRule
from .tasks import image_classification


def clip_by_global_norm(grads: dict[str, torch.Tensor], max_norm: float | None):
    """Differentiable analogue of nn.utils.clip_grad_norm_ on a grad dict."""
    if not max_norm:
        return grads, None
    # tiny floor inside the sqrt: d/dx sqrt(x) is infinite at 0, which would
    # NaN the second-order backward for an all-zero gradient
    total = torch.sqrt(sum((g ** 2).sum() for g in grads.values()) + 1e-24)
    scale = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    return {k: g * scale for k, g in grads.items()}, total


class AdamWRule(LearningRule):
    def __init__(
        self,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        clip_grad_norm: float | None = None,
        task=image_classification,
    ):
        self.lr, self.betas, self.eps = lr, tuple(betas), eps
        self.weight_decay, self.clip_grad_norm = weight_decay, clip_grad_norm
        self.task = task
        # name -> (exp_avg, exp_avg_sq, step); empty == fresh optimizer
        self.state: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}

    @property
    def step_size(self) -> float:
        return self.lr

    def sync_state(self, optimizer: torch.optim.Optimizer, params: dict[str, torch.Tensor]):
        """Snapshot the live optimizer's moments (detached) for the NEXT step."""
        self.state = {}
        for name, p in params.items():
            st = optimizer.state.get(p, {})
            if st:
                self.state[name] = (st["exp_avg"].detach(), st["exp_avg_sq"].detach(),
                                    int(st["step"]))

    def delta(self, probed, params, experience, buffers=None):
        def task_loss(p):
            return self.task(probed, p, experience, buffers)

        grads = grad(task_loss)(params)
        grads, _ = clip_by_global_norm(grads, self.clip_grad_norm)
        b1, b2 = self.betas
        out = {}
        for name, g in grads.items():
            if name in self.state:
                m, v, step = self.state[name]
            else:
                m, v, step = torch.zeros_like(g), torch.zeros_like(g), 0
            t = step + 1
            m_new = b1 * m + (1 - b1) * g
            v_new = b2 * v + (1 - b2) * g * g
            # Floor v at eps^2 before sqrt: sqrt'(0) = inf would NaN the
            # second-order backward for zero-gradient params (e.g. a conv bias
            # cancelled by the following GroupNorm). Forward value is identical
            # to torch's wherever |g| > eps; below that the step is noise anyway.
            denom = v_new.clamp_min(self.eps ** 2).sqrt() / math.sqrt(1 - b2 ** t) + self.eps
            adam_step = (self.lr / (1 - b1 ** t)) * m_new / denom
            out[name] = -self.lr * self.weight_decay * params[name] - adam_step
        return out

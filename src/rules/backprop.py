"""Level 0: plain SGD on a task loss.

Also usable as a stateless reference while actual task updates use AdamW.
Optional gradient clipping does not add momentum or weight decay.

Used (i) as both models' actual training rule in the joint experiment
(pure-SGD decision: the rule that defines V is the rule the model really
trains with), (ii) for guides in the offline track, and (iii) as the
known-solution anchor for phase-A sanity checks.
"""

from __future__ import annotations

from torch.func import grad

from .base import LearningRule
from .tasks import image_classification


class SGDRule(LearningRule):
    def __init__(self, lr: float = 0.01, task=image_classification,
                 clip_grad_norm: float | None = None):
        self.lr = lr
        self.clip_grad_norm = clip_grad_norm
        self.task = task  # (probed, params, experience, buffers) -> scalar

    @property
    def step_size(self) -> float:
        return self.lr

    def delta(self, probed, params, experience, buffers=None):
        def task_loss(p):
            return self.task(probed, p, experience, buffers)

        # torch.func.grad composes with autograd, so this stays differentiable
        # w.r.t. params (Hessian-vector products under the hood in phase B /
        # the joint trainer's alignment term).
        grads = grad(task_loss)(params)
        if self.clip_grad_norm is not None:
            from .adamw import clip_by_global_norm

            grads, _ = clip_by_global_norm(grads, self.clip_grad_norm)
        return {k: -self.lr * g for k, g in grads.items()}

"""The RL task loss: what one transition teaches (the rule's atom).

make_ppo_task returns a task fn with the standard (probed, params, experience,
buffers) signature, so it slots into SGDRule/AdamWRule unchanged: V(X|e) is
the exact optimizer update on THIS transition's PPO loss.

experience.x = observation (1, C, H, W)
experience.y = {"action": (1,A), "advantage": (1,), "logp_old": (1,),
                "value_target": (1,)}
The same fn evaluated over the whole window batch IS the real update's loss
(clipped surrogate + value + entropy), keeping the honesty invariant.
"""

from __future__ import annotations

import torch

from .policy import squashed_logp_entropy


def make_ppo_task(clip_coef: float = 0.2, value_coef: float = 0.5,
                  entropy_coef: float = 0.01):
    def ppo_task(probed, params, experience, buffers=None):
        mean, log_std, value = probed.forward_output(params, experience.x, buffers)
        y = experience.y
        logp, entropy = squashed_logp_entropy(mean, log_std, y["action"])
        ratio = torch.exp(logp - y["logp_old"])
        adv = y["advantage"]
        surrogate = -torch.min(
            ratio * adv, torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * adv
        )
        value_loss = (value - y["value_target"]) ** 2
        return (surrogate + value_coef * value_loss - entropy_coef * entropy).mean()

    return ppo_task

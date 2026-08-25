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
                  entropy_coef: float = 0.01, mean_reg: float = 1e-3):
    """mean_reg: L2 penalty on the PRE-tanh action mean. This — not the
    entropy bonus — is what counters executed-action saturation: the base
    Gaussian entropy sees only log_std and is blind to tanh(N(mu, sigma))
    collapsing onto +-1 as |mu| grows. The principled alternative (SAC-style
    executed-entropy with a fresh rsample) would put nondeterministic noise
    inside the task fn and break the V-honesty/determinism invariant; a
    stored-action entropy estimate has exactly zero expected gradient
    (E[score] = 0). Hence the deterministic mu regularizer."""

    def ppo_components(probed, params, experience, buffers=None):
        """(total, parts): total carries the graph; parts are detached floats
        of the UNWEIGHTED components for logging."""
        mean, log_std, value = probed.forward_output(params, experience.x, buffers)
        y = experience.y
        logp, entropy = squashed_logp_entropy(mean, log_std, y["action"])
        ratio = torch.exp(logp - y["logp_old"])
        adv = y["advantage"]
        surrogate = -torch.min(
            ratio * adv, torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * adv
        )
        value_loss = (value - y["value_target"]) ** 2
        saturation_penalty = mean.pow(2).sum(-1)
        total = (surrogate + value_coef * value_loss - entropy_coef * entropy
                 + mean_reg * saturation_penalty).mean()
        parts = {"policy_loss": float(surrogate.mean()),
                 "value_loss": float(value_loss.mean()),
                 "entropy": float(entropy.mean()),
                 "saturation_penalty": float(saturation_penalty.mean()),
                 "clip_frac": float(((ratio - 1).abs() > clip_coef).float().mean())}
        return total, parts

    def ppo_task(probed, params, experience, buffers=None):
        return ppo_components(probed, params, experience, buffers)[0]

    ppo_task.components = ppo_components   # trainer logs the breakdown
    return ppo_task

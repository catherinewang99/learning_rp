"""PPO loss for real updates and hypothetical plasticity steps."""

from __future__ import annotations

import torch

from .policy import squashed_logp_entropy


def make_ppo_task(clip_coef: float = 0.2,
                  value_coef: float = 0.5,
                  entropy_coef: float = 0.01,
                  mean_reg: float = 1e-3):

    def loss_terms(probed, params, experience, buffers=None):
        mean, log_std, value = probed.forward_output(
            params, experience.x, buffers)
        y = experience.y

        logp, entropy = squashed_logp_entropy(
            mean, log_std, y["action"],
            pre_tanh=y.get("pre_tanh"))

        log_ratio = logp - y["logp_old"]

        # Ordinary PPO clipping happens after exp.
        # Guard extreme values before exp to prevent overflow.
        ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))

        advantage = y["advantage"]
        surrogate = -torch.minimum(
            ratio * advantage,
            ratio.clamp(
                1 - clip_coef, 1 + clip_coef
            ) * advantage,
        )

        value_loss = (value - y["value_target"]).square()
        saturation_penalty = mean.square().sum(-1)

        total = (
            surrogate
            + value_coef * value_loss
            - entropy_coef * entropy
            + mean_reg * saturation_penalty
        ).mean()

        parts = {
            "policy_loss": surrogate.mean(),
            "value_loss": value_loss.mean(),
            "entropy": entropy.mean(),
            "saturation_penalty": saturation_penalty.mean(),
            "clip_frac": (
                (ratio - 1).abs() > clip_coef
            ).float().mean(),
            "ratio_guard_frac": (
                log_ratio.abs() > 20
            ).float().mean(),
        }
        return total, parts

    def ppo_components(probed, params, experience, buffers=None):
        total, parts = loss_terms(
            probed, params, experience, buffers)

        # Convert only when the outer trainer requests metrics.
        values = (
            torch.stack(list(parts.values()))
            .detach().cpu().tolist()
        )
        return total, dict(zip(parts, values))

    def ppo_task(probed, params, experience, buffers=None):
        # Hypothetical updates need only the differentiable loss.
        return loss_terms(
            probed, params, experience, buffers)[0]

    ppo_task.components = ppo_components
    return ppo_task

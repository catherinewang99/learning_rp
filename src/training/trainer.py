"""The offline alignment loop (track 1 phase 3: guide bank -> aligned target).

One generic outer loop; the config's ``train.trainable`` axis decides which
leaf tensors receive gradients:

  * "weights" (phase B): optimize target params θ; rule fixed (SGD).
      -> deliverable: a plasticity-aligned initialization.
  * "rule"    (phase A): optimize rule params φ; θ placed at reference states.
      -> deliverable: a portable learning rule.

The loss differentiates through the hypothetical update inside V (second
order). Alignment is layerwise: guide layers map onto target layers by
even spreading (losses/layerwise.py).
"""

from __future__ import annotations

import torch

from ..kernels.plasticity import plasticity_summary


class AlignmentTrainer:
    def __init__(
        self,
        probed_target,
        rule,
        layerwise_loss,          # LayerwiseAlignmentLoss
        guide_bank,              # GuideBank (>=1 entries; trajectory curriculum)
        experiences: list,       # SAME list/order used to build the bank
        trainable: str = "weights",
        opt_lr: float = 1e-3,
        opt_weight_decay: float = 0.0,
        clip_grad_norm: float | None = None,
        device: str = "cpu",
        log_fn=None,
    ):
        self.probed = probed_target.to(device)
        self.rule = rule
        self.loss = layerwise_loss
        self.bank = guide_bank
        self.experiences = [e.to(device) for e in experiences]
        self.trainable = trainable
        self.log_fn = log_fn
        self.device = device

        self.params = {
            k: v.to(device).requires_grad_(trainable == "weights")
            for k, v in self.probed.params().items()
        }
        self.buffers = {k: v.to(device) for k, v in self.probed.buffers_dict().items()}

        if trainable == "weights":
            leaves = list(self.params.values())
        elif trainable == "rule":
            leaves = list(rule.learnable_parameters())
            if not leaves:
                raise ValueError(f"trainable=rule but {type(rule).__name__} has no φ")
        else:
            raise ValueError(f"unknown trainable mode: {trainable}")
        self.leaves = leaves
        self.clip_grad_norm = clip_grad_norm
        self.optimizer = torch.optim.AdamW(leaves, lr=opt_lr, weight_decay=opt_weight_decay)

    def step(self, bank_index: int) -> dict:
        entry = self.bank[bank_index]
        target_summaries = plasticity_summary(
            self.probed, self.params, self.rule, self.experiences,
            self.bank.probe_to(self.device), self.buffers,
        )
        total, parts = self.loss(target_summaries, entry["layers"])
        self.optimizer.zero_grad()
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.leaves, self.clip_grad_norm or float("inf"))  # inf = measure only
        self.optimizer.step()
        metrics = {"loss/total": float(total), "bank_t": entry.get("t", 0),
                   "grad_norm": float(grad_norm)}
        metrics.update({f"loss/{k}": float(v) for k, v in parts.items()})
        return metrics

    def fit(self, steps: int) -> list[dict]:
        """v0 curriculum: cycle through checkpoints in order.

        TODO(milestone 3): smarter trajectory curricula — e.g. dwell at
        checkpoint t until the K terms drop below tol before advancing.
        """
        history = []
        for i in range(steps):
            metrics = self.step(bank_index=i % len(self.bank))
            history.append(metrics)
            if self.log_fn is not None:
                self.log_fn(metrics)
        return history

    def aligned_params(self) -> dict[str, torch.Tensor]:
        """The deliverable under trainable=weights: a plasticity-aligned init."""
        return {k: v.detach().clone() for k, v in self.params.items()}

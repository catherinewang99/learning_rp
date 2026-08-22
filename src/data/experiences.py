"""Experiences: the units the plasticity kernel is defined over.

An experience's ``x`` is whatever the model's task consumes: an image tensor
for vision, a dict (input_ids, attention_mask) for an LM, later an RL
transition's fields (o, a, r, o') for track 2. Rules/tasks consume the fields
they know about; kernels and losses never look inside.

In the joint experiment, experiences come from each step's own minibatch via
data/paired.batch_to_experiences (single samples, per design; the batch-size
toggle lives there as the ``indices`` subsample).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

InputType = torch.Tensor | dict[str, torch.Tensor]


@dataclass
class Experience:
    x: InputType                     # (b, ...) tensors, b >= 1 always present
    y: torch.Tensor | None = None    # task targets, if the task needs them

    def to(self, device) -> "Experience":
        def move(v):
            if v is None:
                return None
            if isinstance(v, dict):
                return {k: t.to(device) for k, t in v.items()}
            return v.to(device)

        return Experience(x=move(self.x), y=move(self.y))


def batch_to_experiences(batch: dict, view, indices=None) -> list[Experience]:
    """Split a paired batch into single-sample experiences via a modality view
    (per design: each step's minibatch provides the experiences). ``indices``
    selects the m_per_step subsample — pass the SAME indices for every side so
    Π rows stay paired across models."""
    full = view(batch)
    n = (full.x if isinstance(full.x, torch.Tensor) else next(iter(full.x.values()))).shape[0]
    indices = range(n) if indices is None else indices

    def take(value, i):
        if value is None:
            return None
        if isinstance(value, dict):
            return {k: v[i : i + 1] for k, v in value.items()}
        return value[i : i + 1]

    return [Experience(x=take(full.x, i), y=take(full.y, i)) for i in indices]


class ExperienceSource:
    """Samples experience batches from tensors (offline track / synthetic data).

    The guide bank and the target must see the *same* experiences, so the
    canonical usage is: draw a fixed bank once (``sample``), reuse it by
    reference everywhere.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1, seed: int = 0):
        assert len(x) == len(y)
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self, m: int) -> list[Experience]:
        idx = torch.randperm(len(self.x), generator=self.generator)
        idx = idx[: m * self.batch_size].reshape(m, self.batch_size)
        return [Experience(x=self.x[i], y=self.y[i]) for i in idx]

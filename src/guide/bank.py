"""Guide bank: precomputed teacher signal for the OFFLINE track.

For each guide checkpoint t, a fixed experience bank E, and probe set X,
compute and cache the per-layer summaries {layer: {K, V, Pi}}. After this the
guide never runs inside the alignment loop — it is pure cached data.

(The JOINT experiment does not use a bank: its teacher is live but detached,
recomputed each step. See training/joint_trainer.py.)
"""

from __future__ import annotations

import torch

from ..kernels.plasticity import plasticity_summary


class GuideBank:
    """entries[i] = {"layers": {name: {"K","V","Pi"}}, "t": int}, ordered by
    checkpoint time."""

    def __init__(self, entries: list[dict], probe_x):
        self.entries = entries
        self.probe_x = probe_x

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        return self.entries[i]

    def probe_to(self, device):
        if isinstance(self.probe_x, dict):
            return {k: v.to(device) for k, v in self.probe_x.items()}
        return self.probe_x.to(device)

    def save(self, path: str):
        torch.save({"entries": self.entries, "probe_x": self.probe_x}, path)

    @classmethod
    def load(cls, path: str) -> "GuideBank":
        blob = torch.load(path, weights_only=True)
        return cls(blob["entries"], blob["probe_x"])


@torch.no_grad()
def build_guide_bank(
    probed_guide,
    checkpoints: list[dict[str, torch.Tensor]],
    checkpoint_times: list[int],
    rule,
    experiences: list,
    probe_x,
) -> GuideBank:
    """Summaries at every checkpoint. The same ``experiences`` list (same
    order) must be used for target summaries — Π rows must correspond.
    TODO(scale): stream entries to disk if m * n^2 * layers * ckpts is large.
    """
    buffers = probed_guide.buffers_dict()
    entries = []
    for t, params in zip(checkpoint_times, checkpoints):
        layers = plasticity_summary(probed_guide, params, rule, experiences, probe_x, buffers)
        entries.append({"layers": layers, "t": t})
    return GuideBank(entries, probe_x)

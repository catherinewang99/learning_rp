"""Kernel montages: LOOK at K, Π, and ΔK as training proceeds.

One PNG per call and per chosen layer, laid out as one row per side:
    [ K (n x n, probe rows in a meaningful order) | Π (m x m, shared experience
      order) | V for two example experiences | mean |V| ]
Cross-side comparability is the point: Π panels share the experience ordering
(their visual similarity IS what arm C optimizes) and V panels share a color
scale per column. Consumed live by trainers (logged to wandb as images via
RunLogger.log_image) and post-hoc by scripts/visualize_kernels.py.
"""

from __future__ import annotations

from pathlib import Path

import torch


def _panel(ax, mat: torch.Tensor, title: str, vmin=None, vmax=None, cmap="viridis"):
    im = ax.imshow(mat.detach().cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def kernel_montage(
    summaries_by_side: dict[str, dict],   # side -> {layer: {"K","V","Pi"}}
    layer: str,
    out_path: str | Path,
    order_note: str = "",                 # e.g. "probes sorted by dist-to-goal"
    v_examples: tuple[int, ...] = (0, 1),
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sides = list(summaries_by_side)
    # resolve by suffix so "conv4" finds "encoder.conv4" etc.
    any_layers = list(summaries_by_side[sides[0]])
    matches = [ln for ln in any_layers if ln == layer or ln.endswith("." + layer)]
    if not matches:
        raise KeyError(f"layer {layer!r} not found; have {any_layers}")
    layer = matches[0]
    n_cols = 2 + len(v_examples) + 1      # K, Pi, V_e..., mean|V|
    fig, axes = plt.subplots(len(sides), n_cols,
                             figsize=(2.6 * n_cols, 2.8 * len(sides)), squeeze=False)

    # shared color scales per column so cross-side comparison is honest
    k_mats = {s: summaries_by_side[s][layer]["K"] for s in sides}
    pi_mats = {s: _norm_pi(summaries_by_side[s][layer]["Pi"]) for s in sides}
    v_stacks = {s: summaries_by_side[s][layer]["V"] for s in sides}
    v_lim = max(float(v_stacks[s].abs().max()) for s in sides) or 1.0

    for r, s in enumerate(sides):
        _panel(axes[r][0], k_mats[s], f"{s} K ({layer})")
        _panel(axes[r][1], pi_mats[s], f"{s} Π̂ (corr)", vmin=-1, vmax=1, cmap="coolwarm")
        for j, e in enumerate(v_examples):
            e = min(e, v_stacks[s].shape[0] - 1)
            _panel(axes[r][2 + j], v_stacks[s][e], f"{s} V(e{e})",
                   vmin=-v_lim, vmax=v_lim, cmap="coolwarm")
        _panel(axes[r][2 + len(v_examples)], v_stacks[s].abs().mean(0),
               f"{s} mean|V|", vmin=0, vmax=v_lim)

    if order_note:
        fig.suptitle(order_note, fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _norm_pi(pi: torch.Tensor) -> torch.Tensor:
    """Correlation-normalize Π for display (diag -> 1) so the structure is
    visible regardless of the raw response magnitudes."""
    d = pi.diag().clamp_min(1e-12).sqrt()
    return pi / (d[:, None] * d[None, :])

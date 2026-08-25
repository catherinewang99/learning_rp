"""Post-hoc visualization for AV runs (crossmodal-prior visualize_alignment analog).

Reads each run's <outdir>/metrics.jsonl (written by RunLogger) and produces:
  curves_task.png       train + val task losses, val top-1, both sides, per run
  curves_alignment.png  align loss + cross-model K-CKA / Π-CKA means, runs overlaid
  layer_time_<m>.png    layer x training-step heatmap of per-layer K-CKA / Π-CKA
  layer_matrix.png      full L_guide x L_target K-CKA and Π-CKA heatmaps at a
                        checkpoint (needs --checkpoint; rebuilds models + banks)

Usage:
    python scripts/visualize_av.py --run-dirs runs/av-armA runs/av-armB runs/av-armC
    python scripts/visualize_av.py --run-dirs runs/av-armC --checkpoint final.pt
Multi-run overlays label lines by run-dir name (use one dir per arm).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_history(run_dir: Path) -> list[dict]:
    with open(run_dir / "metrics.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def series(history: list[dict], key: str):
    xs = [h["step"] for h in history if key in h and "step" in h]
    ys = [h[key] for h in history if key in h and "step" in h]
    return xs, ys


def plot_curves(histories: dict[str, list[dict]], keys: list[tuple[str, str]], out_path: Path):
    """One subplot per (key, title); one line per run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.6), squeeze=False)
    for ax, (key, title) in zip(axes[0], keys):
        for name, hist in histories.items():
            xs, ys = series(hist, key)
            if xs:
                ax.plot(xs, ys, label=name, linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_layer_time(history: list[dict], metric: str, out_path: Path):
    """Heatmap: probed layers (rows, depth order) x eval steps (cols)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pattern = re.compile(rf"eval/{metric}/(.+)->(.+)")
    layer_keys = sorted(
        {k for h in history for k in h if pattern.fullmatch(k)},
        key=lambda k: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", k)],
    )
    if not layer_keys:
        print(f"no per-layer eval/{metric} keys found; skipping {out_path}")
        return
    steps = sorted({h["step"] for h in history if any(k in h for k in layer_keys)})
    grid = [[next((h[k] for h in history if h.get("step") == s and k in h), float("nan"))
             for s in steps] for k in layer_keys]

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(layer_keys) + 2))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                   extent=[min(steps), max(steps), len(layer_keys) - 0.5, -0.5])
    ax.set_yticks(range(len(layer_keys)))
    ax.set_yticklabels([pattern.fullmatch(k).group(1) for k in layer_keys], fontsize=7)
    ax.set_xlabel("step")
    ax.set_title(f"{metric} per mapped layer over training")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_layer_matrix(run_dir: Path, checkpoint: str, out_path: Path):
    """Full L x L cross-model CKA matrices at a checkpoint (rebuilds the run)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.training.metrics import cross_model_cka_matrices, eval_summaries
    from src.training.rebuild import load_run

    cfg, sides, bank, kernel_fn = load_run(run_dir, checkpoint)
    nv = bool(cfg["plasticity"].get("normalize_v", False))
    sums = {
        name: eval_summaries(side, bank["eval_experiences"][name], bank["probes"][name],
                             kernel_fn, nv)
        for name, side in sides.items()
    }
    k_mat, pi_mat, g_names, t_names = cross_model_cka_matrices(sums["vision"], sums["audio"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mat, title in ((axes[0], k_mat, "K-CKA"), (axes[1], pi_mat, "Π-CKA")):
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(t_names))); ax.set_xticklabels(t_names, rotation=90, fontsize=7)
        ax.set_yticks(range(len(g_names))); ax.set_yticklabels(g_names, fontsize=7)
        ax.set_xlabel("audio (target) layer"); ax.set_ylabel("vision (guide) layer")
        ax.set_title(f"{title} @ {checkpoint}")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--out", default=None, help="default: <first run dir>/viz")
    parser.add_argument("--checkpoint", default=None,
                        help="e.g. final.pt — adds the full layer x layer matrix plot")
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    out = Path(args.out) if args.out else run_dirs[0] / "viz"
    out.mkdir(parents=True, exist_ok=True)
    histories = {d.name: load_history(d) for d in run_dirs}

    plot_curves(histories, [
        ("vision/task_loss", "vision train loss"),
        ("audio/task_loss", "audio train loss"),
        ("eval/vision_top1", "vision val top-1"),
        ("eval/audio_top1", "audio val top-1"),
    ], out / "curves_task.png")

    plot_curves(histories, [
        ("audio/align_loss", "alignment loss (guided side)"),
        ("eval/k_cka/mean", "cross-model K-CKA (mean)"),
        ("eval/pi_cka/mean", "cross-model Π-CKA (mean)"),
    ], out / "curves_alignment.png")

    for metric in ("k_cka", "pi_cka"):
        for name, hist in histories.items():
            plot_layer_time(hist, metric, out / f"layer_time_{metric}_{name}.png")

    if args.checkpoint:
        for d in run_dirs:
            plot_layer_matrix(d, args.checkpoint, out / f"layer_matrix_{d.name}.png")


if __name__ == "__main__":
    main()

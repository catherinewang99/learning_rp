"""One example image + spectrogram from the same class of the AV experiment.

Run on manitoulin (needs data/us8k_cache.pt + CIFAR-100 under data/):
    python scripts/make_class_pair_figure.py                     # ALL six classes
    python scripts/make_class_pair_figure.py --pair siren --index 3   # one pair
All-classes mode writes figs/pairs_all_classes.png (image row + spectrogram
row, one column per class); single-pair mode writes
figs/pair_<audio>_<vision>{_image,_spec,}.png (separate + combined).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.paired_av import CLASS_PAIRS, US8K_CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default=None, choices=US8K_CLASSES,
                   help="one class pair (US8K name); omit for the all-classes grid")
    p.add_argument("--index", type=int, default=0, help="which example of the class")
    p.add_argument("--data-root", default="data")
    args = p.parse_args()

    if args.pair is None:
        return all_classes_grid(args)
    label = US8K_CLASSES.index(args.pair)
    audio_name, vision_name = CLASS_PAIRS[label]

    # spectrogram from the cache (already log-mel, z-normalized on train folds)
    cache = torch.load(Path(args.data_root) / "us8k_cache.pt", weights_only=True)
    idxs = (cache["labels"] == label).nonzero(as_tuple=True)[0]
    spec = cache["specs"][idxs[args.index % len(idxs)]][0].numpy()   # (64, T)

    # raw CIFAR image (no transform -> PIL, true colors)
    from torchvision.datasets import CIFAR100

    ds = CIFAR100(args.data_root, train=True, download=False)
    cifar_label = ds.class_to_idx[vision_name]
    img_idx = [i for i, t in enumerate(ds.targets) if t == cifar_label][args.index]
    img = np.asarray(ds[img_idx][0])                                  # (32, 32, 3)

    out = Path("figs"); out.mkdir(exist_ok=True)
    stem = f"pair_{audio_name}_{vision_name}"

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(img, interpolation="nearest")
    ax.set_title(f"CIFAR-100: {vision_name}", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(out / f"{stem}_image.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    im = ax.imshow(spec, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(f"UrbanSound8K: {audio_name} (log-mel)", fontsize=11)
    ax.set_xlabel("time frame (4 s)"); ax.set_ylabel("mel bin")
    fig.colorbar(im, ax=ax, shrink=0.85, label="normalized log power")
    fig.savefig(out / f"{stem}_spec.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), gridspec_kw={"width_ratios": [1, 1.5]})
    axes[0].imshow(img, interpolation="nearest")
    axes[0].set_title(f"vision input: {vision_name}", fontsize=11)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    im = axes[1].imshow(spec, origin="lower", aspect="auto", cmap="magma")
    axes[1].set_title(f"audio input: {audio_name}", fontsize=11)
    axes[1].set_xlabel("time frame"); axes[1].set_ylabel("mel bin")
    fig.colorbar(im, ax=axes[1], shrink=0.85)
    fig.suptitle(f"one paired class: {audio_name} ↔ {vision_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"wrote figs/{stem}.png, figs/{stem}_image.png, figs/{stem}_spec.png")


def _load_example(label: int, index: int, data_root: str):
    """(raw CIFAR image HWC uint8, log-mel spec (64, T)) for our class id."""
    from torchvision.datasets import CIFAR100

    cache = torch.load(Path(data_root) / "us8k_cache.pt", weights_only=True)
    idxs = (cache["labels"] == label).nonzero(as_tuple=True)[0]
    spec = cache["specs"][idxs[index % len(idxs)]][0].numpy()
    ds = CIFAR100(data_root, train=True, download=False)
    cifar_label = ds.class_to_idx[CLASS_PAIRS[label][1]]
    img_idx = [i for i, t in enumerate(ds.targets) if t == cifar_label][index]
    return np.asarray(ds[img_idx][0]), spec


def all_classes_grid(args):
    """2 x 6 grid: top row CIFAR images, bottom row US8K log-mel spectrograms,
    one column per paired class. Shared spectrogram color scale."""
    n = len(CLASS_PAIRS)
    examples = [_load_example(c, args.index, args.data_root) for c in range(n)]
    vmin = min(s.min() for _, s in examples)
    vmax = max(s.max() for _, s in examples)

    fig, axes = plt.subplots(2, n, figsize=(2.35 * n, 4.6),
                             gridspec_kw={"height_ratios": [1, 1.05]})
    for c, ((img, spec), (audio_name, vision_name)) in enumerate(zip(examples, CLASS_PAIRS)):
        axes[0][c].imshow(img, interpolation="nearest")
        axes[0][c].set_title(vision_name, fontsize=10)
        im = axes[1][c].imshow(spec, origin="lower", aspect="auto", cmap="magma",
                               vmin=float(vmin), vmax=float(vmax))
        axes[1][c].set_title(audio_name.replace("_", " "), fontsize=10)
        for ax in (axes[0][c], axes[1][c]):
            ax.set_xticks([]); ax.set_yticks([])
    axes[0][0].set_ylabel("image (CIFAR-100)", fontsize=10)
    axes[1][0].set_ylabel("log-mel (US8K)", fontsize=10)
    fig.colorbar(im, ax=axes[1], shrink=0.85, pad=0.012)
    out = Path("figs"); out.mkdir(exist_ok=True)
    fig.savefig(out / "pairs_all_classes.png", dpi=300, bbox_inches="tight",
                transparent=True)
    plt.close(fig)
    print("wrote figs/pairs_all_classes.png")


if __name__ == "__main__":
    main()

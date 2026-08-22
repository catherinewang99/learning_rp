"""Download UrbanSound8K (optional) and precompute the log-mel cache.

Usage:
    python scripts/prepare_us8k.py --root data [--download]

Expects (or downloads to) <root>/UrbanSound8K/ with the official layout:
    metadata/UrbanSound8K.csv, audio/fold1..fold10/*.wav

Writes <root>/us8k_cache.pt:
    specs  (N, 1, 64, T) float32 log-mel spectrograms, z-normalized with
           TRAIN-fold statistics (fold != --val-fold)
    labels (N,) int64 in CLASS_PAIRS order (only the 6 selected classes kept)
    folds  (N,) int64 official fold ids (source-file leakage: clips from one
           recording share a fold, so keep val = a whole fold)

Audio recipe: mono, resampled to 22050 Hz, padded/trimmed to 4 s, mel
n_fft=1024 hop=512 n_mels=64, log(x + 1e-6).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.paired_av import US8K_CLASSES

ZENODO_URL = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"
SAMPLE_RATE = 22050
CLIP_SECONDS = 4.0
N_MELS = 64


def load_clip(path: Path, resamplers: dict):
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0, keepdim=True)  # mono
    if sr != SAMPLE_RATE:
        if sr not in resamplers:
            resamplers[sr] = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        wav = resamplers[sr](wav)
    target = int(SAMPLE_RATE * CLIP_SECONDS)
    if wav.shape[1] < target:  # center-pad short clips
        pad = target - wav.shape[1]
        wav = torch.nn.functional.pad(wav, (pad // 2, pad - pad // 2))
    else:
        wav = wav[:, :target]
    return wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--val-fold", type=int, default=10)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    import torchaudio

    root = Path(args.root)
    us8k = root / "UrbanSound8K"
    if not us8k.exists():
        if not args.download:
            sys.exit(f"{us8k} not found; rerun with --download (~5.6GB from Zenodo)")
        import tarfile
        import urllib.request

        root.mkdir(parents=True, exist_ok=True)
        tar_path = root / "UrbanSound8K.tar.gz"
        print(f"downloading {ZENODO_URL} ...")
        urllib.request.urlretrieve(ZENODO_URL, tar_path)
        with tarfile.open(tar_path) as tar:
            tar.extractall(root)
        tar_path.unlink()

    class_to_label = {name: i for i, name in enumerate(US8K_CLASSES)}
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=N_MELS
    )
    resamplers: dict = {}
    specs, labels, folds = [], [], []

    with open(us8k / "metadata" / "UrbanSound8K.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["class"] in class_to_label]
    print(f"{len(rows)} clips across {len(class_to_label)} selected classes")

    for i, row in enumerate(rows):
        path = us8k / "audio" / f"fold{row['fold']}" / row["slice_file_name"]
        wav = load_clip(path, resamplers)
        specs.append(torch.log(melspec(wav) + 1e-6))
        labels.append(class_to_label[row["class"]])
        folds.append(int(row["fold"]))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(rows)}")

    specs = torch.stack(specs)
    labels = torch.tensor(labels, dtype=torch.long)
    folds = torch.tensor(folds, dtype=torch.long)

    train_mask = folds != args.val_fold
    mean, std = specs[train_mask].mean(), specs[train_mask].std()
    specs = (specs - mean) / (std + 1e-8)

    out = root / "us8k_cache.pt"
    torch.save(
        {"specs": specs, "labels": labels, "folds": folds,
         "norm": {"mean": float(mean), "std": float(std)},
         "recipe": {"sr": SAMPLE_RATE, "seconds": CLIP_SECONDS, "n_mels": N_MELS,
                    "n_fft": 1024, "hop": 512, "val_fold": args.val_fold}},
        out,
    )
    counts = torch.bincount(labels, minlength=len(class_to_label)).tolist()
    print(f"wrote {out}: specs {tuple(specs.shape)}, per-class counts {counts}")


if __name__ == "__main__":
    main()

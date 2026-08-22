"""Paired audio–vision data: CIFAR-100 subset <-> UrbanSound8K subset.

Recipe follows Pitzer & Mihai (arXiv:2601.22041): a small set of CIFAR-100
visual classes paired AT THE CLASS LEVEL with semantically corresponding
UrbanSound8K acoustic classes ("precise semantic alignment is not critical").
Consequence for kernels: "row i is the same concept" means the same CLASS —
within-class instance geometry is uncorrelated across modalities by
construction, so cross-modal CKA has a ceiling well below 1 and all claims
are relative to the control arm.

Pairing is a FIXED, seeded within-class assignment per split (stability
decision), with a shuffled_pairs control flag that permutes partners across
classes. Under shuffled_pairs each side keeps its own TRUE label for its task
loss (labels_vision / labels_audio) — only the kernel-row correspondence
breaks.

Audio: 4 s clips -> log-mel spectrograms (64 mels), precomputed into a cache
file by scripts/prepare_us8k.py (torchaudio only needed there). Official
UrbanSound8K folds; val fold is held out entirely (clips from the same source
recording never straddle train/val).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .experiences import Experience

# ---- class mapping (v0 decision; edit freely, ESC-50 additions later) ------
# us8k class name -> cifar100 class name
CLASS_PAIRS = [
    ("dog_bark", "wolf"),
    ("car_horn", "pickup_truck"),
    ("engine_idling", "tractor"),
    ("children_playing", "boy"),
    ("siren", "streetcar"),
    ("street_music", "house"),
]
US8K_CLASSES = [p[0] for p in CLASS_PAIRS]
CIFAR_CLASSES = [p[1] for p in CLASS_PAIRS]

DATA_ROOT = Path(os.environ.get("LRP_DATA_ROOT", "data"))
US8K_CACHE = DATA_ROOT / "us8k_cache.pt"      # written by scripts/prepare_us8k.py

# ---- views ------------------------------------------------------------------


def vision_view(batch: dict) -> Experience:
    return Experience(x=batch["images"], y=batch["labels_vision"])


def audio_view(batch: dict) -> Experience:
    return Experience(x=batch["audio"], y=batch["labels_audio"])


VIEWS = {"vision": vision_view, "audio": audio_view}


# ---- pairing ----------------------------------------------------------------


def build_pairs(
    vision_by_class: dict[int, list[int]],
    audio_by_class: dict[int, list[int]],
    seed: int = 0,
    balance: bool = True,
    shuffled_pairs: bool = False,
) -> list[tuple[int, int, int, int]]:
    """Fixed seeded pairing: [(vision_idx, audio_idx, label_v, label_a), ...].

    Per class: shuffle both sides (seeded), zip up to the shorter side;
    balance=True additionally caps every class at the smallest class's pair
    count (the paper balanced classes; US8K is imbalanced, e.g. car_horn).
    shuffled_pairs permutes the audio column across the whole list (seeded),
    breaking class correspondence while both label columns stay truthful.
    """
    generator = torch.Generator().manual_seed(seed)
    per_class = []
    for c in sorted(vision_by_class):
        v_idx = [vision_by_class[c][i] for i in torch.randperm(len(vision_by_class[c]), generator=generator)]
        a_idx = [audio_by_class[c][i] for i in torch.randperm(len(audio_by_class[c]), generator=generator)]
        n = min(len(v_idx), len(a_idx))
        per_class.append([(v_idx[i], a_idx[i], c, c) for i in range(n)])
    if balance:
        cap = min(len(rows) for rows in per_class)
        per_class = [rows[:cap] for rows in per_class]
    pairs = [row for rows in per_class for row in rows]

    if shuffled_pairs:
        perm = torch.randperm(len(pairs), generator=generator).tolist()
        pairs = [
            (pairs[i][0], pairs[j][1], pairs[i][2], pairs[j][3])
            for i, j in enumerate(perm)
        ]
    # Seeded shuffle of ROW ORDER: the probe/eval banks are prefixes of the val
    # split, so class-ordered rows would give class-skewed (even class-missing)
    # probes. Deterministic given seed.
    order = torch.randperm(len(pairs), generator=generator).tolist()
    return [pairs[i] for i in order]


# ---- datasets ---------------------------------------------------------------


class PairedAVDataset(Dataset):
    """Rows are fixed (image, spectrogram) pairs; images load lazily from the
    underlying CIFAR dataset, spectrograms index into the preloaded cache."""

    def __init__(self, cifar_ds, spectrograms: torch.Tensor, pairs: list):
        self.cifar_ds = cifar_ds
        self.spectrograms = spectrograms
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        v_idx, a_idx, label_v, label_a = self.pairs[i]
        image, _ = self.cifar_ds[v_idx]
        return image, self.spectrograms[a_idx], label_v, label_a


def _collate(batch):
    images, audio, labels_v, labels_a = zip(*batch)
    return {
        "images": torch.stack(images),
        "audio": torch.stack(audio),
        "labels_vision": torch.tensor(labels_v, dtype=torch.long),
        "labels_audio": torch.tensor(labels_a, dtype=torch.long),
    }


def _cifar_class_indices(cifar_ds, class_names: list[str]) -> dict[int, list[int]]:
    """Map our label space (0..5, CLASS_PAIRS order) -> CIFAR sample indices."""
    name_to_ours = {n: i for i, n in enumerate(class_names)}
    cifar_idx_to_ours = {
        cifar_ds.class_to_idx[n]: name_to_ours[n] for n in class_names
    }
    out: dict[int, list[int]] = {i: [] for i in range(len(class_names))}
    for sample_idx, target in enumerate(cifar_ds.targets):
        if target in cifar_idx_to_ours:
            out[cifar_idx_to_ours[target]].append(sample_idx)
    return out


def make_av_loaders(
    cifar_root: str | Path = DATA_ROOT,
    us8k_cache: str | Path = US8K_CACHE,
    batch_size: int = 64,
    num_workers: int = 4,
    val_fold: int = 10,
    seed: int = 0,
    balance: bool = True,
    shuffled_pairs: bool = False,
    augment_train_images: bool = True,
):
    """Train/val DataLoaders over paired batches. Val pairing uses seed+1 so
    train/val pairings are independent draws; probe/eval banks come from val
    (deterministic image transform)."""
    from torchvision import transforms
    from torchvision.datasets import CIFAR100

    cache = torch.load(us8k_cache, weights_only=True)
    # cache: {"specs": (N,1,64,T) float32 (already log-mel + normalized),
    #         "labels": (N,) int64 in CLASS_PAIRS order, "folds": (N,) int64}
    specs, labels, folds = cache["specs"], cache["labels"], cache["folds"]

    normalize = transforms.Normalize(
        [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]  # CIFAR-100 stats
    )
    train_tf = (
        transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), normalize,
        ])
        if augment_train_images
        else transforms.Compose([transforms.ToTensor(), normalize])
    )
    val_tf = transforms.Compose([transforms.ToTensor(), normalize])

    cifar_train = CIFAR100(str(cifar_root), train=True, download=True, transform=train_tf)
    cifar_val = CIFAR100(str(cifar_root), train=False, download=True, transform=val_tf)

    def audio_by_class(fold_mask):
        idx = torch.nonzero(fold_mask, as_tuple=True)[0]
        return {
            c: [int(i) for i in idx if int(labels[i]) == c]
            for c in range(len(CLASS_PAIRS))
        }

    train_pairs = build_pairs(
        _cifar_class_indices(cifar_train, CIFAR_CLASSES),
        audio_by_class(folds != val_fold),
        seed=seed, balance=balance, shuffled_pairs=shuffled_pairs,
    )
    val_pairs = build_pairs(
        _cifar_class_indices(cifar_val, CIFAR_CLASSES),
        audio_by_class(folds == val_fold),
        seed=seed + 1, balance=balance, shuffled_pairs=shuffled_pairs,
    )

    def loader(cifar_ds, pairs, shuffle, drop_last):
        return DataLoader(
            PairedAVDataset(cifar_ds, specs, pairs),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            collate_fn=_collate, drop_last=drop_last,
            persistent_workers=num_workers > 0,
        )

    return (
        loader(cifar_train, train_pairs, True, True),
        loader(cifar_val, val_pairs, False, False),
    )


def synthetic_av_batch(
    b: int = 8, image_hw: int = 16, mels: int = 16, frames: int = 20, n_classes: int = 6,
    generator: torch.Generator | None = None,
) -> dict:
    """Random paired AV batch for tests/local dev — no datasets needed."""
    labels = torch.randint(0, n_classes, (b,), generator=generator)
    return {
        "images": torch.randn(b, 3, image_hw, image_hw, generator=generator),
        "audio": torch.randn(b, 1, mels, frames, generator=generator),
        "labels_vision": labels,
        "labels_audio": labels.clone(),
    }

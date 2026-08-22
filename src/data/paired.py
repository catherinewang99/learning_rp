"""Paired image–caption data for the joint (env-free) experiment.

Adapted from crossmodal-prior src/data/paired_dataset.py. Rows are paired
samples: kernels over the two models' activations compare the same n
underlying concepts. caption_index.py is copied verbatim from crossmodal-prior.

Default paths are the manitoulin/CSAIL locations used by crossmodal-prior;
override with env vars or config.
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .caption_index import CaptionIndex, PairRecord
from .experiences import Experience

IMAGENET_ROOT = os.environ.get(
    "IMAGENET_ROOT", "/storage/dmayo2/datasets/imagenet_pytorch/train"
)
CAPTIONS_JSON = os.environ.get(
    "IMAGENET_CAPTIONS_JSON",
    "/storage/catherinewang99/imagenet-captions/imagenet_captions.json",
)


# ---- batch -> per-model experience views ----------------------------------
# A PairedBatch is the collate output dict. Each model consumes its own view
# of the same underlying samples; row i is the same concept in both views.


def vision_view(batch: dict) -> Experience:
    return Experience(x=batch["images"], y=batch["labels"])


def lm_view(batch: dict) -> Experience:
    return Experience(
        x={"input_ids": batch["caption_ids"], "attention_mask": batch["attention_mask"]},
        y=None,  # next-token targets are derived from input_ids in tasks.lm_next_token
    )


VIEWS = {"vision": vision_view, "lm": lm_view}


# ---- datasets --------------------------------------------------------------


class PairedImageCaptionDataset(Dataset):
    """Map-style dataset over PairRecords: (caption_str, image_tensor, label)."""

    def __init__(
        self,
        records: Sequence[PairRecord],
        image_transform: Callable,
        wnid_to_label: Optional[dict] = None,
    ) -> None:
        from PIL import Image  # lazy; not needed for synthetic/local tests

        self._image = Image
        self.records = list(records)
        self.image_transform = image_transform
        if wnid_to_label is None:
            wnids = sorted({r.wnid for r in self.records})
            wnid_to_label = {w: i for i, w in enumerate(wnids)}
        self.wnid_to_label = wnid_to_label

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        with self._image.open(r.image_path) as img:
            image = self.image_transform(img.convert("RGB"))
        return r.caption, image, self.wnid_to_label[r.wnid]


class PairedCollate:
    """Tokenizes captions and stacks images into the PairedBatch dict:
    caption_ids (B,L), attention_mask (B,L), images (B,3,H,W), labels (B,)."""

    def __init__(self, tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT-2 standard fix

    def __call__(self, batch):
        captions, images, labels = zip(*batch)
        tok = self.tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "caption_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "images": torch.stack(images),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def make_paired_loaders(
    index: CaptionIndex,
    tokenizer,
    image_transform: Callable,
    batch_size: int = 64,
    max_length: int = 64,
    num_workers: int = 8,
):
    """Train/val DataLoaders. drop_last=True on train (kernel losses degrade on
    ragged small batches); never on val."""
    collate = PairedCollate(tokenizer, max_length=max_length)
    wnids = sorted({r.wnid for r in index.train + index.val})
    wnid_to_label = {w: i for i, w in enumerate(wnids)}

    def loader(records, shuffle, drop_last):
        return DataLoader(
            PairedImageCaptionDataset(records, image_transform, wnid_to_label),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=True,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
        )

    return loader(index.train, True, True), loader(index.val, False, False)


def synthetic_paired_batch(
    b: int = 8, image_hw: int = 32, seq_len: int = 12, vocab_size: int = 64, n_classes: int = 10,
    generator: torch.Generator | None = None,
) -> dict:
    """Random PairedBatch for tests/local dev — no ImageNet, no tokenizer."""
    g = generator
    mask = torch.ones(b, seq_len, dtype=torch.long)
    mask[:, seq_len // 2 :] = (torch.rand(b, seq_len - seq_len // 2, generator=g) > 0.3).long()
    return {
        "caption_ids": torch.randint(0, vocab_size, (b, seq_len), generator=g),
        "attention_mask": mask,
        "images": torch.randn(b, 3, image_hw, image_hw, generator=g),
        "labels": torch.randint(0, n_classes, (b,), generator=g),
    }

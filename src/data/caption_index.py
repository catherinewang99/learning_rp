"""Torch-free index over ImageNet-Captions pairs.

Loads imagenet_captions.json, builds caption strings, joins filenames against a
local ImageNet train directory, and produces deterministic train/val splits and
(optionally) a shuffled image<->caption pairing for the control condition.

Everything here is pure Python so it can be unit-tested without torch/PIL.
"""

from __future__ import annotations

import hashlib
import html
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Caption recipe version, recorded in each run's config for provenance.
# v1-tier2: HTML unescape/strip + whitespace collapse; machine tags (= or :)
# dropped; case-insensitive tag dedup; tags capped (default 10); field order
# title, description, tags so truncation cuts tags first.
# v2: v1 plus URLs stripped, camera-filename tokens dropped (DSC0388,
# IMG_1234, ...), immediately-repeated words collapsed.
CAPTION_CLEAN = "v2"

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"(https?://|www\.)\S+", re.I)
# Camera-generated filename tokens: short letter prefix + >=3 digits, e.g.
# DSC0388, IMG_1234, DSCN00721, P1010203, CIMG42x. Requires the letter prefix
# so real numbers in prose ("ca. 1530") are untouched.
_CAMERA_TOKEN_RE = re.compile(
    r"^_*(dsc|dscn|dscf|img|imgp|pict|cimg|gopr|sdc|stc|sam|p10|mvc)[_\-]?\d{2,}[a-z]?$",
    re.I,
)
_REPEAT2_RE = re.compile(r"\b(\w+\s+\w+)(\s+\1\b)+", re.I)  # repeated bigrams
_REPEAT_RE = re.compile(r"\b(\w+)(\s+\1\b)+", re.I)  # repeated single words


def _clean_text(s: str) -> str:
    """HTML-unescape, strip markup/URLs/camera tokens, collapse repeats+space."""
    s = html.unescape(s)
    s = _HTML_TAG_RE.sub(" ", s)
    s = _URL_RE.sub(" ", s)
    s = " ".join(w for w in s.split() if not _CAMERA_TOKEN_RE.match(w))
    s = _REPEAT2_RE.sub(r"\1", s)
    s = _REPEAT_RE.sub(r"\1", s)
    return _WS_RE.sub(" ", s).strip()


@dataclass(frozen=True)
class PairRecord:
    caption: str
    image_path: str  # absolute path to the JPEG
    wnid: str
    filename: str


def build_caption(
    entry: dict,
    fields: Sequence[str] = ("title", "description", "tags"),
    max_tags: Optional[int] = 10,
) -> str:
    """Build a cleaned caption from metadata fields (see CAPTION_CLEAN).

    Empty fields are skipped. Tags are cleaned individually: machine tags
    (containing '=' or ':') are dropped, duplicates are removed
    case-insensitively, and at most max_tags survive.
    """
    parts: List[str] = []
    for field in fields:
        value = entry.get(field)
        if not value:
            continue
        if field == "tags":
            tags: List[str] = []
            seen = set()
            for tag in value:
                tag = _clean_text(tag)
                if not tag or "=" in tag or ":" in tag:
                    continue
                key = tag.lower()
                if key in seen:
                    continue
                seen.add(key)
                tags.append(tag)
                if max_tags is not None and len(tags) >= max_tags:
                    break
            if tags:
                parts.append(" ".join(tags))
        else:
            cleaned = _clean_text(value)
            if cleaned:
                parts.append(cleaned)
    return " ".join(parts).strip()


def _val_hash(filename: str) -> float:
    """Deterministic [0,1) value per image, stable across runs/machines."""
    h = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


class CaptionIndex:
    """Index of (caption, image path) pairs with splits and pairing controls.

    Args:
        json_path: path to imagenet_captions.json.
        imagenet_root: ImageNet train root laid out as root/<wnid>/<filename>.
        caption_fields: which metadata fields to concatenate.
        max_tags: cap on number of tags used in the caption (None = all).
        val_fraction: fraction of pairs held out (split by filename hash, so it
            is stable regardless of ordering or subsetting).
        min_caption_words: drop pairs whose caption is shorter than this.
        subset_size: optional cap on total pairs (applied after filtering,
            deterministic), for debug runs.
        shuffled_pairs: if True, permute the image assignment across records
            (seeded) while captions stay in place -- the semantic-pairing
            control. The permutation is applied within each split so train/val
            stay disjoint.
        seed: seed for subsetting and the shuffled-pairs permutation.
        verify_files: if True, drop records whose image file does not exist on
            disk (requires imagenet_root to be populated).
    """

    def __init__(
        self,
        json_path: str,
        imagenet_root: str,
        caption_fields: Sequence[str] = ("title", "description", "tags"),
        max_tags: Optional[int] = 10,
        val_fraction: float = 0.02,
        min_caption_words: int = 1,
        require_description: bool = False,
        subset_size: Optional[int] = None,
        shuffled_pairs: bool = False,
        seed: int = 0,
        verify_files: bool = True,
    ) -> None:
        self.imagenet_root = Path(imagenet_root)
        with open(json_path) as f:
            entries = json.load(f)

        records: List[PairRecord] = []
        self.num_missing_files = 0
        for e in entries:
            if require_description and not _clean_text(e.get("description") or ""):
                continue
            caption = build_caption(e, fields=caption_fields, max_tags=max_tags)
            if len(caption.split()) < min_caption_words:
                continue
            path = self.imagenet_root / e["wnid"] / e["filename"]
            if verify_files and not path.is_file():
                self.num_missing_files += 1
                continue
            records.append(
                PairRecord(
                    caption=caption,
                    image_path=str(path),
                    wnid=e["wnid"],
                    filename=e["filename"],
                )
            )

        # Deterministic order regardless of JSON order.
        records.sort(key=lambda r: r.filename)

        if subset_size is not None and subset_size < len(records):
            rng = random.Random(seed)
            records = rng.sample(records, subset_size)
            records.sort(key=lambda r: r.filename)

        train, val = [], []
        for r in records:
            (val if _val_hash(r.filename) < val_fraction else train).append(r)

        if shuffled_pairs:
            train = self._shuffle_images(train, seed)
            val = self._shuffle_images(val, seed + 1)

        self.train: List[PairRecord] = train
        self.val: List[PairRecord] = val
        self.shuffled_pairs = shuffled_pairs

    @staticmethod
    def _shuffle_images(records: List[PairRecord], seed: int) -> List[PairRecord]:
        """Reassign images to captions by a seeded permutation (control condition)."""
        rng = random.Random(seed)
        perm = list(range(len(records)))
        rng.shuffle(perm)
        return [
            PairRecord(
                caption=r.caption,
                image_path=records[j].image_path,
                wnid=records[j].wnid,  # wnid follows the image, not the caption
                filename=r.filename,
            )
            for r, j in zip(records, perm)
        ]

    def summary(self) -> str:
        n = len(self.train) + len(self.val)
        return (
            f"CaptionIndex: {n} pairs ({len(self.train)} train / {len(self.val)} val), "
            f"{self.num_missing_files} missing files, shuffled_pairs={self.shuffled_pairs}"
        )

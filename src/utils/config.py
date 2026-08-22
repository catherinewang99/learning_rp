"""Plain-YAML config loading (matching crossmodal-prior's no-framework style).

Supports a single composition mechanism: a top-level ``_base_: <relative
path>`` key deep-merges the file over its base (child wins). That is all —
no hydra, no interpolation. Experiment arms are small override files.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop("_base_", None)
    if base_ref is not None:
        base = load_config(path.parent / base_ref)
        cfg = _deep_merge(base, cfg)
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

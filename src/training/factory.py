"""Build a side's rule + optimizer from its model config (shared by scripts).

Per-model config:
    optimizer: {name: adamw|sgd, lr, weight_decay, betas, eps, clip_grad_norm}
(legacy: a bare ``lr`` key => plain SGD). The rule that defines V and the
torch optimizer that takes the real step are built from the SAME dict, which
is what keeps "V describes the update the model takes" true.
"""

from __future__ import annotations

from ..rules import AdamWRule, SGDRule
from ..rules.tasks import TASK_REGISTRY

DEFAULT_ADAMW = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0, "clip_grad_norm": None}


def optimizer_cfg(model_cfg: dict) -> dict:
    opt = dict(model_cfg.get("optimizer") or {"name": "sgd", "lr": model_cfg["lr"]})
    opt.setdefault("name", "sgd")
    if opt["name"] == "adamw":
        for k, v in DEFAULT_ADAMW.items():
            opt.setdefault(k, v)
        opt["betas"] = tuple(opt["betas"])
    return opt


def make_rule(model_cfg: dict):
    opt = optimizer_cfg(model_cfg)
    task = TASK_REGISTRY[model_cfg["task"]]
    if opt["name"] == "sgd":
        return SGDRule(lr=opt["lr"], task=task)
    if opt["name"] == "adamw":
        return AdamWRule(lr=opt["lr"], betas=opt["betas"], eps=opt["eps"],
                         weight_decay=opt["weight_decay"],
                         clip_grad_norm=opt.get("clip_grad_norm"), task=task)
    raise ValueError(f"unknown optimizer {opt['name']}")

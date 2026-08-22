"""Offline-track entry point: align a target to a frozen guide bank.

Currently wired for the synthetic sanity path (data.synthetic: true).
Usage:
    python scripts/align.py --config configs/sanity_identity.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ExperienceSource, make_probe_set
from src.guide.bank import build_guide_bank
from src.losses import LayerwiseAlignmentLoss, build_loss
from src.models import ProbedModel, build_model
from src.rules import RULE_REGISTRY
from src.rules.tasks import TASK_REGISTRY
from src.training import AlignmentTrainer
from src.utils import load_config, set_seed
from src.utils.logging import maybe_wandb


def build_probed(section: dict, seed: int) -> ProbedModel:
    torch.manual_seed(seed)
    model_cfg = dict(section["model"])
    backbone, default_probe = build_model(model_cfg.pop("name"), **model_cfg)
    probe_kwargs = (
        {"layer_names": section["layers"]} if "layers" in section else default_probe
    )
    return ProbedModel(backbone, **probe_kwargs)


def build_rule(section: dict):
    rule_cfg = dict(section["rule"])
    task = TASK_REGISTRY[rule_cfg.pop("task", "image_classification")]
    return RULE_REGISTRY[rule_cfg.pop("name")](task=task, **rule_cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    data_cfg = cfg["data"]
    if not data_cfg.get("synthetic", False):
        raise NotImplementedError("real datasets for the offline track: see ROADMAP")

    guide = build_probed(cfg["guide"], cfg["seed"])
    in_dim = cfg["guide"]["model"]["in_dim"]
    pool_x = torch.randn(2048, in_dim)
    pool_y = torch.randint(0, cfg["guide"]["model"]["out_dim"], (2048,))

    probe_x = make_probe_set(pool_x, data_cfg["probe_size"], cfg["seed"])
    source = ExperienceSource(
        pool_x, pool_y, batch_size=data_cfg["experience_batch_size"], seed=cfg["seed"]
    )
    experiences = source.sample(data_cfg["n_experiences"])  # fixed bank, reused everywhere

    bank = build_guide_bank(
        guide, [guide.params()], [0], build_rule(cfg["guide"]), experiences, probe_x
    )

    target = build_probed(cfg["target"], cfg["target"].get("init_seed", cfg["seed"]))
    trainer = AlignmentTrainer(
        target,
        build_rule(cfg["target"]),
        LayerwiseAlignmentLoss(build_loss(cfg["loss"])),
        bank,
        experiences,
        trainable=cfg["train"]["trainable"],
        opt_lr=cfg["train"]["opt_lr"],
        opt_weight_decay=cfg["train"].get("opt_weight_decay", 0.0),
        clip_grad_norm=cfg["train"].get("clip_grad_norm"),
        device=cfg["device"],
        log_fn=maybe_wandb(cfg),
    )
    history = trainer.fit(cfg["train"]["steps"])
    print(f"step 0:  {history[0]}")
    print(f"step -1: {history[-1]}")


if __name__ == "__main__":
    main()

"""Audio-vision joint co-training entry point (one arm per run).

Prereqs: scripts/prepare_us8k.py once (CIFAR-100 auto-downloads).
Usage:
    python scripts/av_train.py --config configs/av/arm_c_pi_only.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import paired_av
from src.data.probes import collect_paired_bank
from src.kernels import KERNEL_REGISTRY
from src.losses import LayerwiseAlignmentLoss, build_loss
from src.models import ProbedModel, build_model
from src.training.factory import make_rule, optimizer_cfg
from src.training import JointSide, JointTrainer
from src.training.metrics import tracked_eval
from src.utils import load_config, set_seed
from src.utils.logging import RunLogger

PROBE_TYPES = {"conv2d": [nn.Conv2d], "layernorm": [nn.LayerNorm]}


def build_side(name: str, cfg: dict, device: str) -> JointSide:
    model_cfg = cfg["models"][name]
    backbone, default_probe = build_model(model_cfg["name"], **model_cfg.get("kwargs", {}))
    probe_cfg = model_cfg.get("probe", {})
    if "layer_types" in probe_cfg:
        probe_kwargs = {
            "layer_types": PROBE_TYPES[probe_cfg["layer_types"]],
            "drop_last": probe_cfg.get("drop_last", True),
        }
    else:
        probe_kwargs = default_probe
    probed = ProbedModel(backbone, **probe_kwargs)
    opt = optimizer_cfg(model_cfg)
    return JointSide(name, probed, make_rule(model_cfg), lr=opt["lr"], device=device,
                     view=paired_av.VIEWS[name], optimizer=opt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--wandb-project", default=None,
                        help="required when wandb is enabled; name the project yourself")
    parser.add_argument("--wandb-entity", default="cwang99-duke-university")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.get("wandb", {}).get("enabled", False) and not args.wandb_project:
        parser.error("this config has wandb enabled: --wandb-project <name> is required "
                     "(no default, by convention)")
    if args.wandb_project:
        cfg.setdefault("wandb", {})["project"] = args.wandb_project
        cfg["wandb"]["entity"] = args.wandb_entity
    set_seed(cfg["seed"])
    device = cfg["device"]

    data_cfg = cfg["data"]
    train_loader, val_loader = paired_av.make_av_loaders(
        cifar_root=data_cfg["root"],
        us8k_cache=data_cfg["us8k_cache"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        val_fold=data_cfg["val_fold"],
        seed=cfg["seed"],
        balance=data_cfg["balance"],
        shuffled_pairs=data_cfg["shuffled_pairs"],
    )
    plast = cfg["plasticity"]
    bank = collect_paired_bank(
        val_loader, plast["probe_size"], plast["m_eval"], views=paired_av.VIEWS
    )

    train_cfg = cfg["train"]
    outdir = Path(train_cfg["outdir"])
    log = RunLogger(cfg, outdir)   # wandb (named run) + outdir/metrics.jsonl
    import yaml

    (outdir / "config.yaml").write_text(yaml.safe_dump(cfg))  # for visualize_av.py

    sides = {name: build_side(name, cfg, device) for name in cfg["models"]}
    guidance = cfg["guidance"]
    align = LayerwiseAlignmentLoss(
        build_loss(guidance["loss"]), upper_half=guidance["upper_half"]
    )
    trainer = JointTrainer(
        sides, guided=list(guidance["guided"]), align_loss=align,
        probes=bank["probes"], m_per_step=plast["m_per_step"],
        device=device, seed=cfg["seed"], log_fn=log,
        kernel_fn=KERNEL_REGISTRY[plast.get("kernel", "linear")],
        stratify_by=plast.get("stratify_by"),
        use_checkpoint=plast.get("checkpoint", False),
        cka_every=plast.get("cka_every", 0),
        normalize_v=plast.get("normalize_v", False),
    )

    val_metrics = {name: ["top1", "loss"] for name in sides}  # both sides classify
    step = 0
    while step < train_cfg["steps"]:
        for batch in train_loader:
            trainer.step(batch)   # logs step-level metrics itself (includes "step")
            step += 1
            if step % train_cfg["eval_every"] == 0:
                log({"step": step,
                     **tracked_eval(trainer, bank, val_loader, train_cfg["k_val_batches"],
                                    upper_half=guidance["upper_half"], val_metrics=val_metrics)})
            if step % train_cfg["checkpoint_every"] == 0:
                torch.save(trainer.state_dicts(), outdir / f"step{step}.pt")
            if step >= train_cfg["steps"]:
                break
    torch.save(trainer.state_dicts(), outdir / "final.pt")
    print(f"done: {train_cfg['steps']} steps -> {outdir}/final.pt")


if __name__ == "__main__":
    main()

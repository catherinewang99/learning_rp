"""Joint co-training entry point (env-free experiment; one arm per run).

Usage (manitoulin):
    python scripts/joint_train.py --config configs/joint/arm_c_pi_only.yaml
Local dev: add data.subset_size + device=cpu/mps overrides in a copy of the
arm file, or run the smoke tests instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.caption_index import CaptionIndex
from src.data.paired import VIEWS, make_paired_loaders
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
    probe_cfg = model_cfg.get("probe")
    if probe_cfg and "layer_types" in probe_cfg:
        probe_kwargs = {"layer_types": PROBE_TYPES[probe_cfg["layer_types"]]}
    elif probe_cfg and "layer_names" in probe_cfg:
        probe_kwargs = {"layer_names": probe_cfg["layer_names"]}
    else:
        probe_kwargs = default_probe
    probed = ProbedModel(backbone, **probe_kwargs)
    opt = optimizer_cfg(model_cfg)
    return JointSide(name, probed, make_rule(model_cfg), lr=opt["lr"], device=device,
                     view=VIEWS[name], optimizer=opt)


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

    # ---- data ----
    from torchvision import transforms
    from transformers import GPT2TokenizerFast

    data_cfg = cfg["data"]
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    # NOTE: probe/eval banks come from the val loader -> deterministic transform.
    index = CaptionIndex(
        json_path=data_cfg["captions_json"],
        imagenet_root=data_cfg["imagenet_root"],
        val_fraction=data_cfg["val_fraction"],
        subset_size=data_cfg.get("subset_size"),
        seed=cfg["seed"],
    )
    print(index.summary())
    train_loader, val_loader = make_paired_loaders(
        index, tokenizer, train_tf,
        batch_size=data_cfg["batch_size"],
        max_length=data_cfg["max_caption_length"],
        num_workers=data_cfg["num_workers"],
    )
    # TODO: val split should use the eval transform (Resize/CenterCrop); using
    # train_tf for both is a known simplification of v0 — fix before papering.

    plast = cfg["plasticity"]
    bank = collect_paired_bank(val_loader, plast["probe_size"], plast["m_eval"], views=VIEWS)

    # ---- models + trainer ----
    sides = {name: build_side(name, cfg, device) for name in ("vision", "lm")}
    guidance = cfg["guidance"]
    align = LayerwiseAlignmentLoss(
        build_loss(guidance["loss"]), upper_half=guidance["upper_half"]
    )
    train_cfg = cfg["train"]
    outdir = Path(train_cfg["outdir"])
    log = RunLogger(cfg, outdir)
    import yaml

    (outdir / "config.yaml").write_text(yaml.safe_dump(cfg))

    trainer = JointTrainer(
        sides,
        guided=list(guidance["guided"]),
        align_loss=align,
        probes=bank["probes"],
        m_per_step=plast["m_per_step"],
        device=device,
        seed=cfg["seed"],
        log_fn=log,
        kernel_fn=KERNEL_REGISTRY[plast.get("kernel", "linear")],
        stratify_by=plast.get("stratify_by"),
        use_checkpoint=plast.get("checkpoint", False),
        cka_every=plast.get("cka_every", 0),
        normalize_v=plast.get("normalize_v", False),
    )

    # ---- loop ----
    step = 0
    while step < train_cfg["steps"]:
        for batch in train_loader:
            trainer.step(batch)
            step += 1
            if step % train_cfg["eval_every"] == 0:
                log({"step": step,
                     **tracked_eval(trainer, bank, val_loader, train_cfg["k_val_batches"],
                                    upper_half=guidance["upper_half"],
                                    val_metrics={"vision": ["top1", "loss"], "lm": ["ppl", "loss"]})})
            if step % train_cfg["checkpoint_every"] == 0:
                torch.save(trainer.state_dicts(), outdir / f"step{step}.pt")
            if step >= train_cfg["steps"]:
                break
    torch.save(trainer.state_dicts(), outdir / "final.pt")


if __name__ == "__main__":
    main()

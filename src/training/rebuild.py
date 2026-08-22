"""Rebuild a finished/running AV run from its outdir: config + checkpoint ->
sides with loaded params, plus the fixed probe/eval banks. Shared by the
post-hoc scripts (visualize_av.py, measure_cka.py)."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch import nn

from ..data import paired_av
from ..data.probes import collect_paired_bank
from ..kernels import KERNEL_REGISTRY
from ..models import ProbedModel, build_model
from .factory import make_rule, optimizer_cfg
from .joint_trainer import JointSide


def list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """[(step, path)] sorted by step; final.pt is labelled with train.steps."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    out = []
    for p in run_dir.glob("step*.pt"):
        out.append((int(p.stem[4:]), p))
    if (run_dir / "final.pt").exists():
        out.append((cfg["train"]["steps"], run_dir / "final.pt"))
    return sorted(out)


def load_run(run_dir: str | Path, checkpoint: str | None = None, device: str = "cpu"):
    """Returns (cfg, sides, bank, kernel_fn). checkpoint=None -> fresh init."""
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    d, plast = cfg["data"], cfg["plasticity"]
    _, val_loader = paired_av.make_av_loaders(
        cifar_root=d["root"], us8k_cache=d["us8k_cache"], batch_size=d["batch_size"],
        num_workers=0, val_fold=d["val_fold"], seed=cfg["seed"],
        balance=d["balance"], shuffled_pairs=d["shuffled_pairs"],
    )
    bank = collect_paired_bank(val_loader, plast["probe_size"], plast["m_eval"],
                               views=paired_av.VIEWS)
    state = (torch.load(run_dir / checkpoint, map_location="cpu", weights_only=True)
             if checkpoint else None)
    sides = {}
    for name, mcfg in cfg["models"].items():
        backbone, _ = build_model(mcfg["name"], **mcfg.get("kwargs", {}))
        probed = ProbedModel(backbone, layer_types=[nn.Conv2d],
                             drop_last=mcfg.get("probe", {}).get("drop_last", True))
        opt = optimizer_cfg(mcfg)
        side = JointSide(name, probed, make_rule(mcfg), lr=opt["lr"], device=device,
                         view=paired_av.VIEWS[name], optimizer=opt)
        if state is not None:
            for k, v in side.params.items():
                v.data.copy_(state[name][k].to(device))
        sides[name] = side
    return cfg, sides, bank, KERNEL_REGISTRY[plast.get("kernel", "linear")]

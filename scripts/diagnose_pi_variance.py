"""Is the noisy Π align_loss sampling noise, or is the target itself flat?

At FIXED parameters (a saved checkpoint) recompute the alignment loss many
times, varying ONLY the experience draw. θ never moves and the teacher never
moves, so any spread is pure sampling noise from the m-experience subsample.

For each (kernel, sampler, m) it reports the loss over R draws, plus a
SHUFFLED control in which the audio experiences are permuted against the
vision ones — breaking the row correspondence that Π-alignment depends on.
That control is the "no cross-modal signal" floor.

How to read it:
  * spread shrinks with m, and real mean sits clearly BELOW the shuffled mean
      -> sampling noise; a bigger m (chunked accumulation) buys real signal.
  * real mean ~= shuffled mean at every m
      -> Π-CKA carries no measurable cross-modal signal at this scale; m is
         not the bottleneck (look at the kernel choice / the loss target).

Usage:
    python scripts/diagnose_pi_variance.py --run-dir runs/av-armC
    python scripts/diagnose_pi_variance.py --run-dir runs/av-armC \
        --checkpoint step2000.pt --m-values 4,8,12,24 --repeats 8
    python scripts/diagnose_pi_variance.py --synthetic     # no data needed
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import paired_av
from src.data.experiences import batch_to_experiences
from src.data.probes import collect_paired_bank
from src.kernels import KERNEL_REGISTRY
from src.kernels.plasticity import plasticity_summary
from src.losses import LayerwiseAlignmentLoss, build_loss
from src.models import ProbedModel, build_model
from src.rules import SGDRule
from src.rules.tasks import TASK_REGISTRY
from src.training import JointSide
from src.training.joint_trainer import stratified_indices


def build_sides(cfg, device, state=None):
    sides = {}
    for name, mcfg in cfg["models"].items():
        backbone, _ = build_model(mcfg["name"], **mcfg.get("kwargs", {}))
        probed = ProbedModel(backbone, layer_types=[nn.Conv2d],
                             drop_last=mcfg.get("probe", {}).get("drop_last", True))
        rule = SGDRule(lr=mcfg["lr"], task=TASK_REGISTRY[mcfg["task"]])
        side = JointSide(name, probed, rule, lr=mcfg["lr"], device=device,
                         view=paired_av.VIEWS[name])
        if state is not None:
            for k, v in side.params.items():
                v.data.copy_(state[name][k].to(device))
        sides[name] = side
    return sides


def one_draw(sides, probes, align_loss, batch, idx_v, idx_a, kernel_fn) -> float:
    """Alignment loss for one experience draw at the current (fixed) params."""
    sums = {}
    for name, idx in (("vision", idx_v), ("audio", idx_a)):
        side = sides[name]
        exps = batch_to_experiences(batch, side.view, idx)
        sums[name] = plasticity_summary(
            side.probed, side.detached_params(), side.rule,
            exps, probes[name], side.buffers, kernel_fn,
        )
    total, _ = align_loss(sums["audio"], sums["vision"])   # target, guide
    return float(total)


def summarize(values: list[float]) -> str:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f} +- {std:.4f}  [{min(values):.4f}, {max(values):.4f}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=None)
    p.add_argument("--checkpoint", default="final.pt")
    p.add_argument("--m-values", default="4,8,12,24")
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--kernels", default="linear,centered")
    p.add_argument("--samplers", default="uniform,stratified")
    p.add_argument("--device", default=None)
    p.add_argument("--synthetic", action="store_true", help="tiny models, no data")
    args = p.parse_args()

    m_values = [int(x) for x in args.m_values.split(",")]
    generator = torch.Generator().manual_seed(0)

    if args.synthetic:
        device = args.device or "cpu"
        cfg = {"models": {
            n: {"name": "vgg11", "kwargs": {"in_channels": c, "out_dim": 6, "widths": (8, 16)},
                "probe": {"drop_last": False}, "task": "image_classification", "lr": 0.05}
            for n, c in (("vision", 3), ("audio", 1))}}
        sides = build_sides(cfg, device)
        batches = [paired_av.synthetic_av_batch(64, generator=generator) for _ in range(4)]
        bank = collect_paired_bank(batches[:3], 16, 8, views=paired_av.VIEWS)
        pool, terms = batches, [{"name": "cka_pi", "weight": 1.0}]
    else:
        import yaml

        run_dir = Path(args.run_dir)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        device = args.device or cfg.get("device", "cpu")
        state = torch.load(run_dir / args.checkpoint, map_location="cpu", weights_only=True)
        sides = build_sides(cfg, device, state)
        d = cfg["data"]
        train_loader, val_loader = paired_av.make_av_loaders(
            cifar_root=d["root"], us8k_cache=d["us8k_cache"], batch_size=d["batch_size"],
            num_workers=0, val_fold=d["val_fold"], seed=cfg["seed"],
            balance=d["balance"], shuffled_pairs=d["shuffled_pairs"],
        )
        plast = cfg["plasticity"]
        bank = collect_paired_bank(val_loader, plast["probe_size"], plast["m_eval"],
                                   views=paired_av.VIEWS)
        pool = [b for i, b in zip(range(args.repeats), train_loader)]
        terms = cfg["guidance"]["loss"] or [{"name": "cka_pi", "weight": 1.0}]
        print(f"checkpoint: {run_dir/args.checkpoint}   terms: {terms}")

    probes = {k: ({kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict)
                  else v.to(device)) for k, v in bank["probes"].items()}
    align_loss = LayerwiseAlignmentLoss(build_loss(terms))
    print(f"device={device}  repeats={args.repeats}  probes n={len(next(iter(probes.values())))}")
    print(f"{'kernel':9} {'sampler':11} {'m':>3}  {'REAL mean +- std [min, max]':38} "
          f"{'SHUFFLED (floor)':38}")

    for kernel in args.kernels.split(","):
        kernel_fn = KERNEL_REGISTRY[kernel]
        for sampler in args.samplers.split(","):
            for m in m_values:
                real, shuf = [], []
                for r in range(args.repeats):
                    batch = {k: v.to(device) for k, v in pool[r % len(pool)].items()}
                    y = sides["vision"].view(batch).y.cpu()
                    if sampler == "stratified":
                        idx = stratified_indices(y, m, generator)
                    else:
                        idx = torch.randperm(len(y), generator=generator)[:m].tolist()
                    real.append(one_draw(sides, probes, align_loss, batch, idx, idx, kernel_fn))
                    # break row correspondence: audio rows permuted vs vision rows
                    perm = torch.randperm(len(idx), generator=generator).tolist()
                    idx_a = [idx[j] for j in perm]
                    shuf.append(one_draw(sides, probes, align_loss, batch, idx, idx_a, kernel_fn))
                print(f"{kernel:9} {sampler:11} {m:>3}  {summarize(real):38} {summarize(shuf):38}")


if __name__ == "__main__":
    main()

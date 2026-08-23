"""Render K / Π / ΔK montages from a saved RL-run checkpoint (post hoc).

Usage (on a machine with GL, since probe banks re-render):
    MUJOCO_GL=egl python scripts/visualize_kernels.py \
        --run-dir runs/rl-armC-pi-guide-vision --checkpoint final.pt
Writes <run-dir>/kernels/post_<checkpoint>_<layer>.png per configured layer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from rl_train import build_sides  # noqa: E402

from src.kernels import KERNEL_REGISTRY  # noqa: E402
from src.losses import LayerwiseAlignmentLoss, build_loss  # noqa: E402
from src.rl.cross_render import build_eval_transitions, build_probe_bank  # noqa: E402
from src.rl.rl_trainer import RLTrainer  # noqa: E402
from src.training.kernel_viz import kernel_montage  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    sides = build_sides(cfg, args.device)
    state = torch.load(run_dir / args.checkpoint, map_location="cpu", weights_only=True)
    for name, side in sides.items():
        for k, v in side.params.items():
            v.data.copy_(state[name][k].to(args.device))

    plast = cfg["plasticity"]
    trainer = RLTrainer(
        sides, guided=list(cfg["guidance"]["guided"]),
        align_loss=LayerwiseAlignmentLoss(build_loss(cfg["guidance"]["loss"])),
        window_len=plast["window_len"], m_per_window=plast["m_per_window"],
        gamma=cfg["ppo"]["gamma"], gae_lambda=cfg["ppo"]["gae_lambda"],
        probe_bank=build_probe_bank(sides, plast["probe_size"], cfg["seed"] + 11),
        eval_bank=build_eval_transitions(sides, next(iter(sides.values())).arena.cfg,
                                         plast["eval_transitions"], cfg["seed"] + 12),
        kernel_fn=KERNEL_REGISTRY[plast.get("kernel", "linear")],
        device=args.device, seed=cfg["seed"],
    )
    out, sums = trainer.tracked_eval()
    from src.rl.traj_viz import plot_matched_paths

    path_metrics, records = trainer.path_eval(max_steps=cfg["arena"]["horizon"])
    print({k: round(v, 4) for k, v in path_metrics.items()})
    board = plot_matched_paths(records, next(iter(sides.values())).arena.cfg,
                               run_dir / "paths" / f"post_{args.checkpoint}.png")
    print(f"wrote {board}")
    print({k: round(v, 4) for k, v in out.items() if k.endswith("mean") or "behavior" in k})
    for layer in cfg["train"]["kernel_viz_layers"]:
        path = kernel_montage(sums, layer,
                              run_dir / "kernels" / f"post_{args.checkpoint}_{layer}.png",
                              order_note="probes sorted by dist-to-goal")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

"""RL track entry point (one arm per run).

Usage (manitoulin; EGL headless rendering):
    MUJOCO_GL=egl python scripts/rl_train.py \
        --config configs/rl/arm_c_pi_guide_vision.yaml --wandb-project rl_audiovis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kernels import KERNEL_REGISTRY
from src.losses import LayerwiseAlignmentLoss, build_loss
from src.models import ProbedModel
from src.rl.arena import ArenaConfig, VecArena
from src.rl.audio_sensor import AudioConfig, AudioSensor
from src.rl.cross_render import build_eval_transitions, build_probe_bank
from src.rl.policy import ActorCritic
from src.rl.rl_trainer import RLSide, RLTrainer
from src.rl.traj_viz import plot_matched_paths
from src.training.kernel_viz import kernel_montage
from src.utils import load_config, set_seed
from src.utils.logging import RunLogger
from torch import nn


def build_sides(cfg: dict, device: str) -> dict[str, RLSide]:
    mask_cfg = cfg.get("vision_mask")   # state-keyed occlusion, VISION ONLY
    arena_cfg_v = ArenaConfig(**cfg["arena"], seed=cfg["seed"], state_mask=mask_cfg)
    arena_cfg_a = ArenaConfig(**cfg["arena"], seed=cfg["seed"] + 1000)
    audio_cfg = AudioConfig(**{k: v for k, v in cfg["audio"].items()})
    sensor = AudioSensor(audio_cfg, seed=cfg["seed"] + 5)
    widths = tuple(cfg["model"]["widths"])

    hw = cfg["arena"]["camera_hw"]
    sides = {}
    for idx, (name, in_ch, input_hw, arena_cfg, sens) in enumerate((
        ("vision", 3, (hw, hw), arena_cfg_v, None),
        ("audio", sensor.channels, (audio_cfg.n_mels, sensor.frames),
         arena_cfg_a, sensor),   # 2 channels if binaural
    )):
        net = ActorCritic(in_channels=in_ch, input_hw=input_hw, widths=widths,
                          trunk_dim=cfg["model"]["trunk_dim"],
                          stats_bypass=cfg["model"].get("stats_bypass", True))
        probed = ProbedModel(net, layer_types=[nn.Conv2d], drop_last=False)
        sides[name] = RLSide(name, name, VecArena(arena_cfg), probed,
                             optimizer_cfg=cfg["optimizer"], ppo_cfg=cfg["ppo"],
                             device=device, sensor=sens, seed=cfg["seed"],
                             # cue geometry from the SHARED audio config on
                             # both sides, so either side can teach: full
                             # W-row histories + honest clip phases
                             cue_steps=audio_cfg.window_steps,
                             phase_step=sensor.step_samples,
                             # disjoint per-side noise-id namespaces
                             noise_id_base=1_000_000 + idx * 100_000_000)
    return sides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--wandb-project", default=None,
                        help="required when wandb is enabled (use: rl_audiovis)")
    parser.add_argument("--wandb-entity", default="cwang99-duke-university")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.get("wandb", {}).get("enabled", False) and not args.wandb_project:
        parser.error("wandb enabled: --wandb-project <name> is required")
    if args.wandb_project:
        cfg.setdefault("wandb", {})["project"] = args.wandb_project
        cfg["wandb"]["entity"] = args.wandb_entity
    set_seed(cfg["seed"])
    device = cfg["device"]

    train_cfg = cfg["train"]
    outdir = Path(train_cfg["outdir"])
    log = RunLogger(cfg, outdir)
    import yaml

    (outdir / "config.yaml").write_text(yaml.safe_dump(cfg))

    sides = build_sides(cfg, device)
    if cfg.get("vision_mask", {}) and cfg["vision_mask"].get("enabled", False):
        from src.rl.traj_viz import save_masked_examples

        path = save_masked_examples(sides["vision"], outdir / "vision_mask_examples.png")
        log.log_image("vision_mask/examples", path)
    plast = cfg["plasticity"]
    probe_bank = build_probe_bank(sides, plast["probe_size"], cfg["seed"] + 11)
    eval_bank = build_eval_transitions(
        sides, next(iter(sides.values())).arena.cfg, plast["eval_transitions"],
        cfg["seed"] + 12)

    guidance = cfg["guidance"]
    trainer = RLTrainer(
        sides, guided=list(guidance["guided"]),
        align_loss=LayerwiseAlignmentLoss(build_loss(guidance["loss"]),
                                          upper_half=guidance["upper_half"]),
        window_len=plast["window_len"], m_per_window=plast["m_per_window"],
        gamma=cfg["ppo"]["gamma"], gae_lambda=cfg["ppo"]["gae_lambda"],
        probe_bank=probe_bank, eval_bank=eval_bank,
        kernel_fn=KERNEL_REGISTRY[plast.get("kernel", "linear")],
        use_checkpoint=plast.get("checkpoint", True),
        device=device, seed=cfg["seed"], log_fn=log,
        stats_horizon=train_cfg.get("stats_horizon", 2000),
        n_matched_layouts=train_cfg.get("path_layouts", 8),
    )

    order_note = "probe rows sorted by distance-to-goal (near -> far)"
    for w in range(1, train_cfg["windows"] + 1):
        trainer.window()
        if w % train_cfg["eval_every"] == 0:
            step = trainer.window_count * trainer.window_len
            out, sums = trainer.tracked_eval()
            path_metrics, records = trainer.path_eval(
                max_steps=next(iter(sides.values())).arena.cfg.horizon)
            log({"step": step, **out,
                 **{f"eval/{k}": v for k, v in path_metrics.items()}})
            if w % train_cfg["kernel_viz_every"] == 0:
                for layer in train_cfg["kernel_viz_layers"]:
                    path = kernel_montage(sums, layer,
                                          outdir / "kernels" / f"step{step}_{layer}.png",
                                          order_note=order_note)
                    log.log_image(f"kernels/{layer}", path, step=step)
                board = plot_matched_paths(records, next(iter(sides.values())).arena.cfg,
                                           outdir / "paths" / f"step{step}.png")
                log.log_image("paths/board", board, step=step)
        if w % train_cfg["checkpoint_every"] == 0:
            torch.save({n: s.detached_params() for n, s in sides.items()},
                       outdir / f"window{w}.pt")
    torch.save({n: s.detached_params() for n, s in sides.items()}, outdir / "final.pt")
    print(f"done: {train_cfg['windows']} windows -> {outdir}/final.pt")


if __name__ == "__main__":
    main()

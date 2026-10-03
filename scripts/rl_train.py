"""Audio-vision RL training and configurable reference-response experiments.

Real task updates stay AdamW. plasticity.reference controls only the virtual
update used to measure V. --dump-config resolves all overrides without rendering.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kernels import KERNEL_REGISTRY
from src.kernels.response_geometry import precise_kernel
from src.losses import LayerwiseAlignmentLoss, build_loss
from src.models import ProbedModel
from src.rl.arena import ArenaConfig, VecArena
from src.rl.audio_sensor import AudioConfig, AudioSensor
from src.rl.cross_render import build_eval_transitions, build_probe_bank
from src.rl.policy import ActorCritic
from src.rl.response_alignment import ResponseAlignmentLoss
from src.rl.rl_trainer import RLSide, RLTrainer
from src.rl.traj_viz import plot_matched_paths
from src.training.kernel_viz import kernel_montage
from src.utils import load_config, set_seed
from src.utils.logging import RunLogger


def build_sides(cfg: dict, device: str) -> dict[str, RLSide]:
    mask_cfg = cfg.get('vision_mask')
    arena_cfg_v = ArenaConfig(**cfg['arena'], seed=cfg['seed'], state_mask=mask_cfg)
    arena_cfg_a = ArenaConfig(**cfg['arena'], seed=cfg['seed'] + 1000)
    audio_cfg = AudioConfig(**cfg['audio'])
    sensor = AudioSensor(audio_cfg, seed=cfg['seed'] + 5)
    widths = tuple(cfg['model']['widths'])
    hw = cfg['arena']['camera_hw']
    sides = {}
    for idx, (name, in_ch, input_hw, arena_cfg, sens) in enumerate((
        ('vision', 3, (hw, hw), arena_cfg_v, None),
        ('audio', sensor.channels, (audio_cfg.n_mels, sensor.frames), arena_cfg_a, sensor),
    )):
        net = ActorCritic(in_channels=in_ch, input_hw=input_hw, widths=widths,
                         trunk_dim=cfg['model']['trunk_dim'],
                         stats_bypass=cfg['model'].get('stats_bypass', True))
        layers = cfg['model'].get('probe_layers')
        if isinstance(layers, dict):
            layers = layers[name]  # permits future architecture-specific probe names
        if layers is None:
            probed = ProbedModel(net, layer_types=[nn.Conv2d], drop_last=False)
        else:
            if not isinstance(layers, list) or not layers:
                raise ValueError('model.probe_layers must be a nonempty list (or per-side lists).')
            probed = ProbedModel(net, layer_names=layers, drop_last=False)
        sides[name] = RLSide(
            name, name, VecArena(arena_cfg), probed,
            optimizer_cfg=cfg['optimizer'], ppo_cfg=cfg['ppo'], device=device,
            sensor=sens, seed=cfg['seed'], cue_steps=audio_cfg.window_steps,
            phase_step=sensor.step_samples, noise_id_base=1_000_000 + idx * 100_000_000,
            reference_cfg=cfg['plasticity'].get('reference'))
    return sides


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--wandb-project', default=None)
    parser.add_argument('--wandb-entity', default=None)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--windows', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--run-tag', default=None)
    parser.add_argument('--dump-config', action='store_true')
    return parser


def resolve_config(args):
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.windows is not None:
        if args.windows <= 0:
            raise ValueError('--windows must be positive.')
        cfg['train']['windows'] = args.windows
    if args.device:
        cfg['device'] = args.device
    if args.run_tag and any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.run_tag):
        raise ValueError('--run-tag may contain only letters, digits, hyphens and underscores.')
    wb = cfg.setdefault('wandb', {})
    if args.no_wandb:
        wb['enabled'] = False
    if args.wandb_project:
        wb['project'] = args.wandb_project
    if args.wandb_entity:
        wb['entity'] = args.wandb_entity
    if args.seed is not None or args.run_tag:
        suffix = f'-s{cfg["seed"]}' + (f'-{args.run_tag}' if args.run_tag else '')
        wb['name'] = wb.get('name', Path(args.config).stem) + suffix
        cfg['train']['outdir'] += suffix
    return cfg


def main():
    parser = make_parser()
    args = parser.parse_args()
    cfg = resolve_config(args)
    if args.dump_config:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return
    if cfg.get('wandb', {}).get('enabled', False) and not cfg['wandb'].get('project'):
        parser.error('wandb enabled: supply --wandb-project, or use --no-wandb.')
    set_seed(cfg['seed'])
    device, train_cfg = cfg['device'], cfg['train']
    outdir = Path(train_cfg['outdir'])
    if (outdir / 'metrics.jsonl').exists():
        parser.error(f'{outdir}/metrics.jsonl already exists; choose a new --run-tag. No resume is implied.')
    log = RunLogger(cfg, outdir)
    (outdir / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    sides = build_sides(cfg, device)
    if cfg.get('vision_mask', {}).get('enabled', False):
        from src.rl.traj_viz import save_masked_examples
        path = save_masked_examples(sides['vision'], outdir / 'vision_mask_examples.png')
        log.log_image('vision_mask/examples', path)

    plast, guidance = cfg['plasticity'], cfg['guidance']
    probe_bank = build_probe_bank(sides, plast['probe_size'], cfg['seed'] + 11)
    stable = plast.get('stable_comparison', False)
    eval_probe_bank = (build_probe_bank(sides, plast['probe_size'], cfg['seed'] + 1011)
                       if stable else probe_bank)
    eval_bank = build_eval_transitions(sides, next(iter(sides.values())).arena.cfg,
                                      plast['eval_transitions'], cfg['seed'] + 12)
    if stable:
        response_measurement = ResponseAlignmentLoss(
            objective=guidance.get('response_objective', 'pi'),
            upper_half=guidance.get('upper_half', False),
            **plast.get('response_geometry', {}))
        align_loss = response_measurement
    else:
        response_measurement = None
        align_loss = LayerwiseAlignmentLoss(build_loss(guidance['loss']),
                                            upper_half=guidance.get('upper_half', False))
    kernel_fn = KERNEL_REGISTRY[plast.get('kernel', 'linear')]
    dtype = plast.get('gram_dtype', 'native')
    if dtype == 'float64':
        kernel_fn = precise_kernel(kernel_fn)
    elif dtype != 'native':
        raise ValueError('plasticity.gram_dtype must be native or float64.')
    trainer = RLTrainer(
        sides, guided=list(guidance['guided']), align_loss=align_loss,
        window_len=plast['window_len'], m_per_window=plast['m_per_window'],
        align_every=plast.get('align_every', 10), gamma=cfg['ppo']['gamma'],
        gae_lambda=cfg['ppo']['gae_lambda'], probe_bank=probe_bank, eval_bank=eval_bank,
        eval_probe_bank=eval_probe_bank, response_measurement=response_measurement,
        kernel_fn=kernel_fn, use_checkpoint=plast.get('checkpoint', True),
        device=device, seed=cfg['seed'], log_fn=log,
        stats_horizon=train_cfg.get('stats_horizon', 2000),
        n_matched_layouts=train_cfg.get('path_layouts', 8),
        max_align_ratio=guidance.get('max_align_ratio', 0.1))
    torch.save({'train_probes': probe_bank, 'eval_probes': eval_probe_bank,
                'eval_transitions': eval_bank, 'matched_layouts': trainer.matched_layouts},
               outdir / 'measurement_banks.pt')
    print(f'Task optimizer: AdamW. Reference: {next(iter(sides.values())).reference_name}. '
          f'Objective: {guidance.get("response_objective", guidance.get("loss"))}. '
          f'Guided: {guidance["guided"]}.')
    print('Probes:', {name: list(side.probed._probe_modules) for name, side in sides.items()})

    start = time.monotonic()
    for w in range(1, train_cfg['windows'] + 1):
        trainer.window()
        if w % train_cfg['eval_every'] == 0:
            step = trainer.window_count * trainer.window_len
            out, sums = trainer.tracked_eval()
            path_metrics, records = trainer.path_eval(
                max_steps=next(iter(sides.values())).arena.cfg.horizon)
            log({'step': step, 'elapsed_seconds': time.monotonic() - start, **out,
                 **{f'eval/{k}': v for k, v in path_metrics.items()}})
            if w % train_cfg['kernel_viz_every'] == 0:
                for layer in train_cfg['kernel_viz_layers']:
                    path = kernel_montage(sums, layer,
                        outdir / 'kernels' / f'step{step}_{layer}.png',
                        order_note='Held-out probe rows: near -> far. Pi panel is within-model response cosine.')
                    log.log_image(f'kernels/{layer}', path, step=step)
                board = plot_matched_paths(records, next(iter(sides.values())).arena.cfg,
                                          outdir / 'paths' / f'step{step}.png')
                log.log_image('paths/board', board, step=step)
        if w % train_cfg['checkpoint_every'] == 0:
            # Keep weight-only files compatible with the existing notebooks.
            torch.save({n: s.detached_params() for n, s in sides.items()}, outdir / f'window{w}.pt')
            torch.save({n: s.optimizer.state_dict() for n, s in sides.items()},
                       outdir / f'optimizer_window{w}.pt')
    torch.save({n: s.detached_params() for n, s in sides.items()}, outdir / 'final.pt')
    print(f'done: {train_cfg["windows"]} windows -> {outdir}/final.pt')


if __name__ == '__main__':
    main()

"""Reference updates used to measure plasticity; real RL training stays AdamW."""

from __future__ import annotations

import math

from ..rules import AdamWRule, SGDRule


def build_reference_rule(task, optimizer_cfg: dict, reference_cfg: dict | None = None):
    # Omitted config preserves the old AdamW-reference behavior.
    cfg = dict(reference_cfg or {})
    name = cfg.get('name', 'adamw').lower()
    allowed = {'name', 'lr', 'clip_grad_norm'}
    if name == 'adamw':
        allowed |= {'betas', 'eps', 'weight_decay'}
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f'Unknown {name} reference options: {sorted(unknown)}')
    lr = float(cfg.get('lr', optimizer_cfg['lr']))
    clip = cfg.get('clip_grad_norm', optimizer_cfg.get('clip_grad_norm'))
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('Reference learning rate must be positive and finite.')
    if clip is not None and (not math.isfinite(clip) or clip <= 0):
        raise ValueError('Reference gradient clip must be positive or null.')
    if name == 'sgd':
        return SGDRule(lr=lr, task=task, clip_grad_norm=clip)
    if name == 'adamw':
        return AdamWRule(lr=lr, task=task, clip_grad_norm=clip,
                         betas=tuple(cfg.get('betas', optimizer_cfg.get('betas', (0.9, 0.999)))),
                         eps=cfg.get('eps', optimizer_cfg.get('eps', 1e-8)),
                         weight_decay=cfg.get('weight_decay', optimizer_cfg.get('weight_decay', 0.0)))
    raise ValueError(f'Unknown reference rule {name!r}; choose sgd or adamw.')

import math
from collections.abc import Mapping

import torch

def _norm(grads) -> torch.Tensor:
    terms = [g.detach().double().square().sum() for g in grads if g is not None]
    if not terms:
        raise RuntimeError("The task loss is disconnected from all parameters.")
    return torch.stack(terms).sum().sqrt()

def checked_step(params, optimizer,
                 task_loss, align_loss = None,
                 max_align_ratio = 0.1,
                 clip_norm = 0.5):
    if max_align_ratio is not None and (
            not math.isfinite(max_align_ratio) or max_align_ratio < 0):
        raise ValueError("max_align_ratio must be finite and nonnegative or None")
    if clip_norm is not None and (not math.isfinite(clip_norm) or clip_norm <= 0):
        raise ValueError("clip_norm must be positive and finite or None")
    tensors = list(params.values())
    optimizer.zero_grad(set_to_none=True)
    if not bool(torch.isfinite(task_loss.detach()).all()):
        raise FloatingPointError("Nonfinite task loss; optimizer was not stepped.")
    task_grads = torch.autograd.grad(task_loss, tensors, allow_unused=True,
                                    retain_graph=align_loss is not None)
    task_norm = _norm(task_grads)
    if not bool(torch.isfinite(task_norm)):
        raise FloatingPointError("Nonfinite task gradient; optimizer was not stepped.")

    align_grads = [None] * len(tensors)
    task_n = task_norm.item()
    align_n, scale, cosine, bad_alignment = 0.0, 0.0, 0.0, 0.0
    if align_loss is not None:
        if not bool(torch.isfinite(align_loss.detach()).all()):
            bad_alignment = 1.0
        else:
            if not align_loss.requires_grad:
                raise RuntimeError("Alignment loss is detached from the student.")
            align_grads = torch.autograd.grad(align_loss, tensors, allow_unused=True)
            if not any(g is not None for g in align_grads):
                raise RuntimeError("Alignment loss is disconnected from the student.")
            align_norm = _norm(align_grads)
            if not bool(torch.isfinite(align_norm)):
                bad_alignment = 1.0
            else:
                align_n = align_norm.item()
                if align_n > 0:
                    scale = (1.0 if max_align_ratio is None else
                             min(1.0, max_align_ratio * task_n / align_n))
                dot = sum((a.detach().double() * b.detach().double()).sum()
                          for a, b in zip(task_grads, align_grads)
                          if a is not None and b is not None)
                cosine = float(dot / (task_norm * align_norm + 1e-30))

    # REMINDER FOR this dumbass -- do not multiply invalid gradients by zero: zero * nan is still nan.
    for p, g_task, g_align in zip(tensors, task_grads, align_grads):
        g = None if g_task is None else g_task.detach().clone()
        if scale > 0 and g_align is not None:
            added = scale * g_align.detach()
            g = added.clone() if g is None else g + added
        p.grad = g

    combined_norm = _norm([p.grad for p in tensors])
    if not bool(torch.isfinite(combined_norm)):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Nonfinite combined gradient; optimizer not stepped.")
    # Same clipping convention as the existing rule, with a robust norm.
    if clip_norm is not None:
        clip_scale = min(1.0, clip_norm / (combined_norm.item() + 1e-6))
        for p in tensors:
            if p.grad is not None:
                p.grad.mul_(clip_scale)
    optimizer.step()
    align_value = (align_loss.detach().item()
                   if align_loss is not None and not bad_alignment else 0.0)
    metrics = {
        "task_grad_norm": task_n,
        "align_grad_norm": align_n,
        "align_scale": scale,
        "align_task_grad_ratio_raw": align_n / (task_n + 1e-30),
        "align_task_grad_ratio_applied": scale * align_n / (task_n + 1e-30),
        "align_task_grad_cosine": cosine,
        "align_skipped_nonfinite": bad_alignment,
        "grad_norm": combined_norm.item(),
        "total_loss": task_loss.detach().item() + scale * align_value,
    }

    if align_loss is None:
        return {key: value for key, value in metrics.items() if not key.startswith("align") }
    return metrics

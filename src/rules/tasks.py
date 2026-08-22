"""Task losses: what one "experience" teaches, per modality.

A task is a pure function (probed, params, experience, buffers) -> scalar loss.
Rules differentiate these to produce updates; keep each one small and obvious.
"""

from __future__ import annotations

import torch.nn.functional as F


def image_classification(probed, params, experience, buffers=None):
    """Cross-entropy on class labels. experience.x: images (B,3,H,W) or any
    tensor the backbone accepts; experience.y: (B,) int64 labels."""
    logits = probed.forward_output(params, experience.x, buffers)
    return F.cross_entropy(logits, experience.y)


def lm_next_token(probed, params, experience, buffers=None):
    """Next-token prediction on captions. experience.x is a dict with
    input_ids (B,L) and attention_mask (B,L); pad positions are excluded from
    the loss via the standard -100 label convention. Uses the HF LM head's
    built-in shift-by-one loss."""
    x = experience.x
    labels = x["input_ids"].masked_fill(x["attention_mask"] == 0, -100)
    out = probed.forward_output(params, {**x, "labels": labels}, buffers)
    return out.loss


TASK_REGISTRY = {
    "image_classification": image_classification,
    "lm_next_token": lm_next_token,
    # Track 2 will add: policy_gradient (see ROADMAP)
}

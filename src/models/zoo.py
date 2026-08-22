"""Model registry: name -> constructor + default probe spec.

Adding a guide/target = add an entry here; nothing downstream changes.
Probing conventions (per project decisions):
  * vision ResNets/CNNs: probe nn.Conv2d layers
  * GPT-2 / transformers: probe nn.LayerNorm layers
Heavy deps (torchvision, transformers) are imported lazily inside constructors.
"""

from __future__ import annotations

from collections import OrderedDict

from torch import nn


def mlp(in_dim: int = 3072, hidden: tuple[int, ...] = (256, 256), out_dim: int = 10):
    layers: OrderedDict[str, nn.Module] = OrderedDict()
    layers["flatten"] = nn.Flatten()
    d = in_dim
    for i, h in enumerate(hidden, start=1):
        layers[f"fc{i}"] = nn.Linear(d, h)
        layers[f"act{i}"] = nn.ReLU()
        d = h
    layers["head"] = nn.Linear(d, out_dim)
    return nn.Sequential(layers)


def tiny_cnn(in_channels: int = 3, widths: tuple[int, ...] = (8, 16, 32), out_dim: int = 10):
    """Small conv net for tests and local dev (probe: Conv2d)."""
    layers: OrderedDict[str, nn.Module] = OrderedDict()
    c = in_channels
    for i, w in enumerate(widths, start=1):
        layers[f"conv{i}"] = nn.Conv2d(c, w, kernel_size=3, padding=1)
        layers[f"act{i}"] = nn.ReLU()
        layers[f"pool{i}"] = nn.MaxPool2d(2)
        c = w
    layers["gap"] = nn.AdaptiveAvgPool2d(1)
    layers["flatten"] = nn.Flatten()
    layers["head"] = nn.Linear(c, out_dim)
    return nn.Sequential(layers)


VGG11_WIDTHS = (64, 128, 256, 256, 512, 512, 512, 512)
VGG11_POOL_AFTER = (1, 2, 4, 6, 8)  # canonical VGG-11 pool positions


def vgg11(
    in_channels: int = 3,
    out_dim: int = 6,
    widths: tuple[int, ...] = VGG11_WIDTHS,
    pool_after: tuple[int, ...] | None = None,
    norm: str = "group",          # group | batch | none
    groups: int = 8,
):
    """Canonical VGG-11 conv structure with a configurable norm layer and the
    standard small-input classifier (GAP + single linear).

    norm="group" (default, the headline choice): batch-independent, so
    train mode == eval mode and kernels are deterministic functions of
    (theta, X) — required for the honesty invariant (the update V describes is
    the update the model takes). norm="batch" is canonical VGG-11-BN but its
    train/eval duality breaks that invariant (see README gotchas); norm="none"
    is the 2014 original (fragile with pure SGD). Audio twin: in_channels=1,
    same stack; the adaptive pool absorbs rectangular spectrogram inputs.
    """
    if pool_after is None:
        pool_after = VGG11_POOL_AFTER if len(widths) == 8 else tuple(range(1, len(widths) + 1))
    layers: OrderedDict[str, nn.Module] = OrderedDict()
    c = in_channels
    for i, w in enumerate(widths, start=1):
        layers[f"conv{i}"] = nn.Conv2d(c, w, kernel_size=3, padding=1)
        if norm == "group":
            layers[f"norm{i}"] = nn.GroupNorm(min(groups, w), w)
        elif norm == "batch":
            layers[f"norm{i}"] = nn.BatchNorm2d(w)
        elif norm != "none":
            raise ValueError(f"unknown norm: {norm}")
        layers[f"act{i}"] = nn.ReLU()
        if i in pool_after:
            layers[f"pool{i}"] = nn.MaxPool2d(2)
        c = w
    layers["gap"] = nn.AdaptiveAvgPool2d(1)
    layers["flatten"] = nn.Flatten()
    layers["head"] = nn.Linear(c, out_dim)
    return nn.Sequential(layers)


def resnet(depth: int = 18, out_dim: int = 1000, pretrained: bool = False):
    """torchvision ResNet, from scratch by default (joint-experiment decision)."""
    from torchvision import models

    ctor = {18: models.resnet18, 34: models.resnet34, 50: models.resnet50}[depth]
    weights = "DEFAULT" if pretrained else None
    net = ctor(weights=weights)
    if out_dim != net.fc.out_features:
        net.fc = nn.Linear(net.fc.in_features, out_dim)
    return net


# GPT-2 student presets, ported from crossmodal-prior models/gpt2_student.py
GPT2_PRESETS = {
    "nano": dict(n_layer=4, n_embd=512, n_head=8),
    "tiny": dict(n_layer=8, n_embd=512, n_head=8),   # ~51M params w/ GPT-2 vocab
    "small": dict(n_layer=12, n_embd=768, n_head=12),
}


def gpt2(preset: str = "tiny", vocab_size: int = 50257, n_positions: int = 256, **overrides):
    """Randomly initialized GPT2LMHeadModel (from-scratch decision). The same
    object serves alignment (LayerNorm hooks) and next-token training (via
    the `labels` kwarg in rules/tasks.lm_next_token). ``overrides`` replace
    preset fields (n_layer, n_embd, n_head) for tiny test models.

    IMPORTANT: attention is forced to "eager". The default SDPA/flash kernels
    have no double-backward, and the alignment loss differentiates THROUGH a
    gradient step (second order) — flash attention breaks that with
    'derivative for ..._flash_attention_backward is not implemented'."""
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=vocab_size, n_positions=n_positions, **{**GPT2_PRESETS[preset], **overrides}
    )
    config._attn_implementation = "eager"
    return GPT2LMHeadModel(config)


def _conv_probe():
    return {"layer_types": [nn.Conv2d]}


def _layernorm_probe():
    return {"layer_types": [nn.LayerNorm]}


# name -> (constructor, default probe kwargs for ProbedModel)
REGISTRY = {
    "mlp": (mlp, {"layer_names": ["act2"]}),
    "tiny_cnn": (tiny_cnn, _conv_probe()),
    "vgg11": (vgg11, _conv_probe()),
    "resnet18": (lambda **kw: resnet(18, **kw), _conv_probe()),
    "resnet34": (lambda **kw: resnet(34, **kw), _conv_probe()),
    "gpt2": (gpt2, _layernorm_probe()),
}


def build_model(name: str, **kwargs):
    """Returns (backbone, default probe kwargs). kwargs go to the constructor."""
    ctor, probe_kwargs = REGISTRY[name]
    return ctor(**kwargs), dict(probe_kwargs)

"""ProbedModel: attach representation probes to an arbitrary nn.Module.

This is the entire adapter surface for "any guide, any target": everything
downstream (kernels, losses, trainers) only ever sees the dict
``H = features(params, x)`` mapping layer name -> pooled (B, D) activations.

Probing conventions (ported from crossmodal-prior / training-the-untrainable):
  * Layers are selected by MODULE TYPE (e.g. nn.Conv2d for ResNets,
    nn.LayerNorm for GPT-2/ViTs), captured in execution order via forward
    hooks, or by explicit qualified names for small custom models.
  * The last captured layer is dropped by default (crossmodal convention:
    the output-adjacent layer is trivially task-supervised).
  * Activations are pooled to (B, D): conv maps (B,C,H,W) -> spatial mean;
    token activations (B,T,D) -> mean over non-pad tokens when the input dict
    carries an ``attention_mask``; (B,D) passes through.

All forward passes are functional (torch.func.functional_call) so hypothetical
parameter updates never mutate the module and stay differentiable.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.func import functional_call

InputType = torch.Tensor | dict[str, torch.Tensor]


def pool_features(h: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Reduce activations to (B, D). See module docstring for the conventions."""
    if h.dim() == 2:
        return h
    if h.dim() == 4:  # conv maps
        return h.mean(dim=(2, 3))
    if h.dim() != 3:
        raise ValueError(f"unexpected feature shape {tuple(h.shape)}")
    if attention_mask is None or attention_mask.shape[1] != h.shape[1]:
        return h.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(h.dtype)
    return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class ProbedModel(nn.Module):
    """Wraps a backbone and exposes functional, multi-layer feature access.

    Args:
        backbone: any nn.Module.
        layer_types: module classes to probe (all instances, execution order).
        layer_names: explicit qualified module names to probe instead.
            Exactly one of layer_types / layer_names must be given.
        drop_last: drop the last captured layer (default True with layer_types;
            ignored for explicit layer_names).

    Note: caller controls train/eval mode. We set eval() at construction —
    BN/dropout must not inject noise or buffer mutation into kernels.
    """

    def __init__(
        self,
        backbone: nn.Module,
        layer_types: list[type] | None = None,
        layer_names: list[str] | None = None,
        drop_last: bool = True,
    ):
        super().__init__()
        if (layer_types is None) == (layer_names is None):
            raise ValueError("give exactly one of layer_types / layer_names")
        self.backbone = backbone
        self.drop_last = drop_last if layer_types is not None else False
        by_name = dict(backbone.named_modules())

        if layer_names is not None:
            missing = [n for n in layer_names if n not in by_name]
            if missing:
                raise KeyError(f"layers {missing} not found; have: {list(by_name)[:20]}")
            self._probe_modules = {n: by_name[n] for n in layer_names}
        else:
            types = tuple(layer_types)
            self._probe_modules = {
                name: mod
                for name, mod in by_name.items()
                if isinstance(mod, types) and not isinstance(mod, (nn.Sequential, nn.ModuleList))
            }
            if not self._probe_modules:
                raise ValueError(f"no modules of types {types} found in backbone")

        # Populated on first forward: probe layer names in EXECUTION order
        # (dict insertion order of the hook captures), post drop_last.
        self.layer_order: list[str] | None = None
        self.backbone.eval()

    # ---- functional parameter access -------------------------------------

    def params(self, clone: bool = True) -> dict[str, torch.Tensor]:
        it = self.backbone.named_parameters()
        return {k: (v.detach().clone() if clone else v) for k, v in it}

    def buffers_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.backbone.named_buffers()}

    # ---- functional forwards ---------------------------------------------

    def _call(self, params, x: InputType, buffers):
        merged = {**params, **(buffers if buffers is not None else {})}
        if isinstance(x, dict):
            return functional_call(self.backbone, merged, args=(), kwargs=x)
        return functional_call(self.backbone, merged, (x,))

    def forward_output(self, params, x: InputType, buffers=None):
        """Full backbone output (used by task losses in rules/tasks.py)."""
        return self._call(params, x, buffers)

    def features(self, params, x: InputType, buffers=None) -> dict[str, torch.Tensor]:
        """All probed layers' pooled activations: {layer_name: (B, D)}."""
        mask = x.get("attention_mask") if isinstance(x, dict) else None
        captured: dict[str, torch.Tensor] = {}
        handles = [
            mod.register_forward_hook(
                lambda _m, _i, out, name=name: captured.__setitem__(name, out)
                if isinstance(out, torch.Tensor)
                else captured.__setitem__(name, out[0])
            )
            for name, mod in self._probe_modules.items()
        ]
        try:
            self._call(params, x, buffers)
        finally:
            for h in handles:
                h.remove()

        names = list(captured)  # execution order
        if self.drop_last and len(names) > 1:
            names = names[:-1]
        if self.layer_order is None:
            self.layer_order = names
        return {name: pool_features(captured[name], mask) for name in names}

"""Notebook-friendly kernel visualizations: K, V (=ΔK), and Π as heatmaps.

Typical notebook session:

    from src.viz import RunView, plot_kernel, plot_delta, plot_plasticity
    rv = RunView("runs/av-armC-adamw")        # loads config + fixed val banks once
    rv.steps                                   # available checkpoints, e.g. [500, 1000, ...]
    b = rv.bundle(step=2000, side="audio", layer="conv4")   # mode="self" default
    plot_kernel(b); plot_delta(b, experience="mean"); plot_plasticity(b)
    rv.triptych(step=2000, layer="conv4")      # vision + audio, K | V | Π grid

Modes (the axes decision):
  * "self"  (default): one fixed set S of m_eval held-out samples is BOTH the
    probe set and the experience set — K (m x m over S), V (m x m), Π (m x m)
    all share the same class-sorted rows/columns. A lens for understanding.
  * "probe": what the loss sees — K and V on the n fixed probes (probe-sorted
    axes), Π over the m_eval experiences (experience-sorted axes).

All rows/columns are sorted by class with boundary ticks. V and Π use a
diverging colormap centered at 0. Everything is computed with detached params
(no graphs); AdamW hypothetical steps use the optimizer moments restored from
the checkpoint (rebuild.load_run warns if a checkpoint lacks them).
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..data.experiences import Experience
from ..kernels.plasticity import plasticity_summary
from .. import kernels as _k  # noqa: F401  (kernel registry via rebuild)
from ..training.rebuild import list_checkpoints, load_run

CLASS_NAMES = None  # filled lazily from data.paired_av.CLASS_PAIRS


def _class_names():
    global CLASS_NAMES
    if CLASS_NAMES is None:
        from ..data.paired_av import CLASS_PAIRS

        CLASS_NAMES = [f"{a}/{v}" for a, v in CLASS_PAIRS]
    return CLASS_NAMES


class RunView:
    """Load a run once; compute class-sorted kernel bundles at any checkpoint.

    Caches (step -> sides) and (step, side, mode) -> summaries, so plotting
    several layers/experiences at one checkpoint costs one computation.
    """

    def __init__(self, run_dir: str | Path, device: str = "cpu"):
        self.run_dir = Path(run_dir)
        self.device = device
        self._ckpts = dict(list_checkpoints(self.run_dir))  # step -> path
        self._sides_cache: dict = {}
        self._summary_cache: dict = {}
        # load once with no checkpoint just for cfg/bank (cheap fresh models)
        self.cfg, _, self.bank, self.kernel_fn = load_run(self.run_dir, None, device)

    @property
    def steps(self) -> list[int]:
        return sorted(self._ckpts)

    def _resolve(self, step):
        if step in ("init", 0):      # before training: seeded fresh init
            return None
        if step in ("final", None):
            return "final.pt"
        step = int(step)
        if step not in self._ckpts:
            raise KeyError(f"no checkpoint at step {step}; have {self.steps} (+ 'init', 'final')")
        return self._ckpts[step].name

    def sides(self, step):
        name = self._resolve(step)
        key = name or "init"
        if key not in self._sides_cache:
            _, sides, _, _ = load_run(self.run_dir, name, self.device)
            self._sides_cache[key] = sides
        return self._sides_cache[key]

    def _experiences_and_probes(self, side_name: str, mode: str):
        exps = [e.to(self.device) for e in self.bank["eval_experiences"][side_name]]
        exp_labels = torch.cat([e.y for e in exps])
        if mode == "self":  # probes := the experiences themselves
            probe_x = torch.cat([e.x for e in exps])
            probe_labels = exp_labels
        elif mode == "probe":
            probe_x = self.bank["probes"][side_name].to(self.device)
            probe_labels = self.bank["probe_labels"][side_name]
        else:
            raise ValueError(f"mode must be 'self' or 'probe', got {mode}")
        return exps, exp_labels, probe_x, probe_labels

    def summaries(self, step, side_name: str, mode: str = "self") -> dict:
        key = (self._resolve(step) or "init", side_name, mode)
        if key not in self._summary_cache:
            side = self.sides(step)[side_name]
            exps, exp_labels, probe_x, probe_labels = self._experiences_and_probes(side_name, mode)
            sums = plasticity_summary(
                side.probed, side.detached_params(), side.rule,
                exps, probe_x, side.buffers, self.kernel_fn,
                normalize_v=bool(self.cfg["plasticity"].get("normalize_v", False)),
            )
            self._summary_cache[key] = (sums, exp_labels.cpu(), probe_labels.cpu())
        return self._summary_cache[key]

    def bundle(self, step, side: str, layer: str, mode: str = "self") -> dict:
        """Class-sorted {K, V, Pi, probe_labels, exp_labels, meta} at one layer.
        step: a checkpoint step, 'final', or 'init' (before training)."""
        sums, exp_labels, probe_labels = self.summaries(step, side, mode)
        meta = {"run": self.run_dir.name, "step": step, "side": side,
                "layer": layer, "mode": mode}
        return sorted_bundle(sums, layer, probe_labels, exp_labels, meta)

    def triptych(self, step, layer: str, sides=("vision", "audio"), mode: str = "self",
                 experience="mean", figsize=(13, 8)):
        """len(sides) x 3 grid: K | V | Π per side, shared class-sorted axes."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(sides), 3, figsize=figsize, squeeze=False)
        for row, side in enumerate(sides):
            b = self.bundle(step, side, layer, mode)
            plot_kernel(b, ax=axes[row][0], title=f"{side} K (cosine)", normalize=True)
            plot_delta(b, experience=experience, ax=axes[row][1],
                       title=f"{side} V ({experience}, unit-norm)", normalize=True)
            plot_plasticity(b, ax=axes[row][2], title=f"{side} Π (cosine)", normalize=True)
        fig.suptitle(f"{self.run_dir.name}  step={step}  layer={layer}  mode={mode}")
        fig.tight_layout()
        return fig


def sorted_bundle(sums: dict, layer: str, probe_labels: torch.Tensor,
                  exp_labels: torch.Tensor, meta: dict | None = None) -> dict:
    """Class-sort one layer of a plasticity_summary into a plottable bundle."""
    if layer not in sums:
        raise KeyError(f"layer {layer!r} not in {list(sums)}")
    p_ord = torch.argsort(probe_labels, stable=True)
    e_ord = torch.argsort(exp_labels, stable=True)
    s = sums[layer]
    return {
        "K": s["K"].detach().cpu()[p_ord][:, p_ord],
        "V": s["V"].detach().cpu()[e_ord][:, p_ord][:, :, p_ord],
        "Pi": s["Pi"].detach().cpu()[e_ord][:, e_ord],
        "probe_labels": probe_labels[p_ord],
        "exp_labels": exp_labels[e_ord],
        "meta": meta or {"step": "?", "side": "?", "layer": layer, "mode": "?"},
    }


# ---- pure plot functions (take a bundle or a raw matrix) --------------------


def _boundaries(labels: torch.Tensor) -> list[int]:
    lab = labels.tolist()
    return [i for i in range(1, len(lab)) if lab[i] != lab[i - 1]]


def _heatmap(mat: torch.Tensor, row_labels, col_labels, ax=None, title=None,
             vmax=None, center=True):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.4))
    mat = torch.as_tensor(mat).detach().cpu()
    if vmax is None:
        vmax = float(mat.abs().max()) or 1.0
    kw = ({"cmap": "RdBu_r", "vmin": -vmax, "vmax": vmax} if center
          else {"cmap": "viridis", "vmin": 0.0, "vmax": vmax})
    im = ax.imshow(mat, **kw)
    for b in _boundaries(torch.as_tensor(row_labels)):
        ax.axhline(b - 0.5, color="k", linewidth=0.4, alpha=0.5)
    for b in _boundaries(torch.as_tensor(col_labels)):
        ax.axvline(b - 0.5, color="k", linewidth=0.4, alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    ax.figure.colorbar(im, ax=ax, shrink=0.85)
    return ax


def cosine_normalize(k: torch.Tensor) -> torch.Tensor:
    """C_ij = K_ij / sqrt(K_ii K_jj): unit diagonal, entries in [-1, 1].
    A VIEW for reading structure — the loss consumes the raw Gram."""
    d = k.diag().clamp_min(1e-12).sqrt()
    return k / d[:, None] / d[None, :]


def plot_kernel(bundle: dict, ax=None, title=None, vmax=None, normalize: bool = False):
    """1. The raw kernel K: sample x sample similarities (class-sorted).

    normalize=True plots the cosine-normalized view (unit diagonal) — usually
    far more readable, since raw diagonal norms ||h_i||^2 dominate the color
    scale and hide off-diagonal structure."""
    b = bundle
    k = cosine_normalize(b["K"]) if normalize else b["K"]
    tag = "K (cosine)" if normalize else "K"
    return _heatmap(k, b["probe_labels"], b["probe_labels"], ax=ax,
                    title=title or f"{tag}  ({b['meta']['side']} {b['meta']['layer']}, "
                                   f"step {b['meta']['step']})",
                    vmax=1.0 if normalize else vmax)


def plot_delta(bundle: dict, experience=0, ax=None, title=None, vmax=None,
               normalize: bool = False):
    """2. The delta kernel V = ΔK/η for one experience, or aggregated.

    experience: int index (in class-sorted order), "mean" (signed common
    drift), or "absmean" (where change concentrates, sign ignored).
    normalize=True divides by the matrix's Frobenius norm — V̂, the DIRECTION
    of change (exactly what the v_match loss term compares). Raw V's scale is
    huge by construction (ΔK of raw Grams, divided by η=lr); no normalization
    exists in the pipeline — CKA is simply scale-invariant downstream."""
    b = bundle
    v = b["V"]
    if experience == "mean":
        mat, tag = v.mean(dim=0), "mean over experiences"
    elif experience == "absmean":
        mat, tag = v.abs().mean(dim=0), "|V| mean"
    else:
        i = int(experience)
        mat, tag = v[i], f"experience {i} (class {int(b['exp_labels'][i])})"
    if normalize:
        mat = mat / (mat.norm() + 1e-12)
        tag += ", unit-norm"
    return _heatmap(mat, b["probe_labels"], b["probe_labels"], ax=ax,
                    title=title or f"V — {tag}", vmax=vmax,
                    center=experience != "absmean")


def plot_plasticity(bundle: dict, ax=None, title=None, vmax=None,
                    normalize: bool = False):
    """3. The plasticity kernel Π: experience x experience ⟨vec V, vec V⟩.

    normalize=True plots the cosine view Π_ij / sqrt(Π_ii Π_jj) — the angle
    between two experiences' response directions (unit diagonal, [-1, 1]).
    Raw Π entries are enormous (inner products of raw V's); the loss's CKA is
    scale-invariant, so no normalization exists (or is needed) in the pipeline."""
    b = bundle
    pi = cosine_normalize(b["Pi"]) if normalize else b["Pi"]
    tag = "Π (cosine)" if normalize else "Π"
    return _heatmap(pi, b["exp_labels"], b["exp_labels"], ax=ax,
                    title=title or f"{tag}  ({b['meta']['side']} {b['meta']['layer']}, "
                                   f"step {b['meta']['step']})",
                    vmax=1.0 if normalize else vmax)


def plot_k_pair(bundle_a: dict, bundle_b: dict, axes=None, normalize: bool = True,
                suptitle: str | None = None):
    """The training-the-untrainable comparison: the two networks' K = HHᵀ on
    the SAME class-sorted probes, side by side, with their CKA similarity in
    the title. 'Do these two heatmaps look alike?' quantified."""
    import matplotlib.pyplot as plt

    from ..kernels.metrics import cka

    sim = float(cka(bundle_a["K"], bundle_b["K"]))
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    else:
        fig = axes[0].figure
    for ax, b in zip(axes, (bundle_a, bundle_b)):
        plot_kernel(b, ax=ax, normalize=normalize,
                    title=f"{b['meta']['side']} K ({b['meta']['layer']})")
    fig.suptitle(suptitle or f"step {bundle_a['meta']['step']}   "
                             f"CKA(K_a, K_b) = {sim:.3f}")
    return fig, sim


def compare_arms(runviews: dict, step, layer: str, mode: str = "self",
                 sides=("vision", "audio"), normalize: bool = True):
    """Grid: one row per arm, columns = each side's K at (step, layer), with
    the cross-model CKA printed per row. All arms share seed => identical
    probes and identical 'init' state, so differences are pure training effect.

        compare_arms({'A': rv_a, 'B': rv_b, 'C': rv_c}, step=2000, layer='conv4')
    """
    import matplotlib.pyplot as plt

    from ..kernels.metrics import cka

    fig, axes = plt.subplots(len(runviews), len(sides),
                             figsize=(4.4 * len(sides), 3.9 * len(runviews)),
                             squeeze=False)
    for row, (arm, rv) in enumerate(runviews.items()):
        bundles = [rv.bundle(step, side, layer, mode) for side in sides]
        sim = float(cka(bundles[0]["K"], bundles[1]["K"]))
        for col, b in enumerate(bundles):
            plot_kernel(b, ax=axes[row][col], normalize=normalize,
                        title=f"arm {arm}: {b['meta']['side']} K")
        axes[row][0].set_ylabel(f"arm {arm}   CKA={sim:.3f}", fontsize=10)
    fig.suptitle(f"K = HHᵀ per network   layer={layer}  step={step}")
    fig.tight_layout()
    return fig


def shared_vmax(bundles: list[dict], key: str) -> float:
    """Fixed color scale across checkpoints: max |value| over bundles[key]."""
    return max(float(torch.as_tensor(b[key]).abs().max()) for b in bundles)

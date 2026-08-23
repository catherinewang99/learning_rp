"""Joint co-training with plasticity-kernel guidance (env-free experiment).

Two models train simultaneously on paired data, each on its own task.
Honesty invariant: the rule that defines V is the update the model actually
takes — the torch optimizer (SGD or AdamW + weight decay + grad clipping) and
the functional rule (rules/backprop.py, rules/adamw.py) are built from ONE
per-model ``optimizer`` config (training/factory.py); stateful rules read the
live optimizer moments each step. Guided side(s) additionally minimize a layerwise
alignment loss between their plasticity summaries and the other model's,
computed on the fixed probe set with experiences drawn from the current
minibatch.

Sides are arbitrary named modalities (vision/lm, vision/audio, ...); each
side carries its own ``view`` mapping a paired batch to its Experience.
Guidance direction is config: e.g. guided=["audio"] (vision teaches audio),
["vision"] (reversed), both names (mutual), [] (control arm A).
The teacher side's summaries are computed with detached parameters, so no
graph is built through the teacher and no alignment gradient reaches it —
a teacher is a moving target that evolves only through its own task loss.
"""

from __future__ import annotations

import torch

from ..data.experiences import batch_to_experiences
from ..kernels.gram import linear_gram
from ..kernels.plasticity import plasticity_summary, representation_summary
from .metrics import cross_model_k_cka


class JointSide:
    """One model's training state: functional params + its optimizer.

    ``optimizer``: {name: sgd|adamw, weight_decay, betas, eps, clip_grad_norm}
    (lr comes from the ``lr`` arg). The torch optimizer here and the rule's
    hypothetical step must describe the same update — build both from the
    same config via training/factory.py.
    """

    def __init__(self, name: str, probed, rule, lr: float, device: str, view,
                 optimizer: dict | None = None):
        self.name = name
        self.probed = probed.to(device)
        self.rule = rule
        self.params = {k: v.to(device).requires_grad_(True) for k, v in probed.params().items()}
        self.buffers = {k: v.to(device) for k, v in probed.buffers_dict().items()}
        opt = dict(optimizer or {"name": "sgd"})
        leaves = list(self.params.values())
        if opt.get("name", "sgd") == "sgd":
            self.optimizer = torch.optim.SGD(leaves, lr=lr)
        elif opt["name"] == "adamw":
            self.optimizer = torch.optim.AdamW(
                leaves, lr=lr, betas=tuple(opt.get("betas", (0.9, 0.999))),
                eps=opt.get("eps", 1e-8), weight_decay=opt.get("weight_decay", 0.0),
            )
        else:
            raise ValueError(f"unknown optimizer {opt['name']}")
        self.clip_grad_norm = opt.get("clip_grad_norm")  # None = no clipping
        self.view = view  # paired batch dict -> this side's Experience

    def sync_rule_state(self):
        """For stateful rules (AdamW): hand the live optimizer moments to the
        rule so the hypothetical step matches the real one this step."""
        if hasattr(self.rule, "sync_state"):
            self.rule.sync_state(self.optimizer, self.params)

    def detached_params(self) -> dict[str, torch.Tensor]:
        return {k: v.detach() for k, v in self.params.items()}


def stratified_indices(y: torch.Tensor, m: int, generator: torch.Generator) -> list[int]:
    """Pick m row indices spreading across the classes present in y: shuffle,
    group by class, then round-robin one per class until m are chosen. With m a
    multiple of the class count and all classes present, the pick is exactly
    balanced; otherwise as balanced as the batch allows."""
    perm = torch.randperm(len(y), generator=generator).tolist()
    by_class: dict[int, list[int]] = {}
    for i in perm:
        by_class.setdefault(int(y[i]), []).append(i)
    chosen: list[int] = []
    while len(chosen) < m and any(by_class.values()):
        for rows in by_class.values():
            if rows and len(chosen) < m:
                chosen.append(rows.pop())
    return chosen


class JointTrainer:
    def __init__(
        self,
        sides: dict[str, JointSide],       # {"vision": ..., "lm": ...}
        guided: list[str],                  # which sides receive alignment loss
        align_loss,                         # LayerwiseAlignmentLoss (term weights inside)
        probes: dict,                       # {"vision": tensor, "lm": dict} — FIXED probe set X
        m_per_step: int,                    # experiences subsampled from each minibatch
        #   for the plasticity term. THE scoped cost knob: bounds the number of
        #   retained autograd graphs per step (each holds one hypothetical
        #   step + one stepped probe forward of the guided model).
        device: str = "cpu",
        seed: int = 0,
        log_fn=None,
        kernel_fn=linear_gram,              # gram.KERNEL_REGISTRY choice; applies to
        #   teacher, guided side, and (via tracked_eval) measurement alike
        stratify_by: str | None = None,     # side whose view().y gives class labels;
        #   if set, the m experiences are drawn class-balanced (round-robin) so Π's
        #   class structure is consistent step to step. None = uniform subsample.
        use_checkpoint: bool = False,       # recompute stepped probe forwards in
        #   backward instead of storing them (kernels/response.stepped_grams)
        cka_every: int = 0,                 # if > 0: every k steps log the actual
        #   cross-model K-CKA SIMILARITY per layer (all arms, incl. control), on
        #   the fixed probes ("cka/probe/...") and the current minibatch
        #   ("cka/batch/..."). Detached, two cheap forwards per side.
    ):
        assert set(guided) <= set(sides), f"guided {guided} not in sides {list(sides)}"
        self.sides = sides
        self.guided = guided
        self.align_loss = align_loss
        self.device = device
        self.m_per_step = m_per_step
        self.log_fn = log_fn
        self.kernel_fn = kernel_fn
        self.stratify_by = stratify_by
        self.use_checkpoint = use_checkpoint
        self.cka_every = cka_every
        # What the loss actually reads. {"K"} alone => no hypothetical steps at
        # all (K-CKA control arm runs at ~control-arm cost).
        self.needs = getattr(align_loss, "needs", {"K", "V", "Pi"}) if guided else set()
        self.generator = torch.Generator().manual_seed(seed)
        self.probes = {
            name: (
                {k: v.to(device) for k, v in p.items()} if isinstance(p, dict) else p.to(device)
            )
            for name, p in probes.items()
        }
        self.step_count = 0

    def summary(self, side, params, experiences, name: str) -> dict:
        """Per-layer summary at ``params`` — cheap K-only path when the loss
        needs nothing else, full plasticity summary otherwise."""
        if self.needs <= {"K"}:
            return representation_summary(
                side.probed, params, self.probes[name], side.buffers, self.kernel_fn
            )
        return plasticity_summary(
            side.probed, params, side.rule, experiences, self.probes[name],
            side.buffers, self.kernel_fn, use_checkpoint=self.use_checkpoint,
        )

    def measure_k_cka(self, batch: dict | None = None) -> dict[str, float]:
        """Actual cross-model K-CKA (similarity, higher = more aligned) at the
        current params: per mapped layer + mean, on the fixed probes and, if a
        batch is given, on that minibatch. Guide side listed first."""
        names = list(self.sides)
        guide = self._teacher_of(self.guided[0]) if self.guided else names[0]
        target = next(n for n in names if n != guide)

        def k_only(name, x):
            side = self.sides[name]
            return representation_summary(
                side.probed, side.detached_params(), x, side.buffers, self.kernel_fn
            )

        out: dict[str, float] = {}
        with torch.no_grad():
            sims = cross_model_k_cka(k_only(guide, self.probes[guide]),
                                     k_only(target, self.probes[target]))
            out.update({f"cka/probe/{k}": v for k, v in sims.items()})
            if batch is not None:
                sims = cross_model_k_cka(k_only(guide, self.sides[guide].view(batch).x),
                                         k_only(target, self.sides[target].view(batch).x))
                out.update({f"cka/batch/{k}": v for k, v in sims.items()})
        return out

    def _teacher_of(self, side_name: str) -> str:
        others = [s for s in self.sides if s != side_name]
        assert len(others) == 1, "JointTrainer is a two-model loop"
        return others[0]

    def step(self, batch: dict) -> dict:
        batch = {k: v.to(self.device) for k, v in batch.items()}
        metrics: dict[str, float] = {}

        # Same subsample indices for every side: Π rows must stay paired.
        b = next(iter(batch.values())).shape[0]
        if self.stratify_by is not None:
            y = self.sides[self.stratify_by].view(batch).y
            idx = stratified_indices(y.cpu(), self.m_per_step, self.generator)
        else:
            idx = torch.randperm(b, generator=self.generator)[: self.m_per_step].tolist()

        # Stateful rules see the optimizer moments the real step will use.
        for side in self.sides.values():
            side.sync_rule_state()

        # Teacher summaries first (graph-free via detached params), so mutual
        # guidance uses both sides' pre-step state symmetrically.
        summaries_detached = {}
        for name in {self._teacher_of(g) for g in self.guided}:
            side = self.sides[name]
            experiences = batch_to_experiences(batch, side.view, idx)
            summaries_detached[name] = self.summary(
                side, side.detached_params(), experiences, name
            )

        # Per-side losses: own task CE always; + alignment if guided.
        for name, side in self.sides.items():
            task_loss = side.rule.task(
                side.probed, side.params, side.view(batch), side.buffers
            )
            metrics[f"{name}/task_loss"] = float(task_loss)
            total = task_loss

            if name in self.guided:
                experiences = batch_to_experiences(batch, side.view, idx)
                own_summaries = self.summary(side, side.params, experiences, name)
                teacher = summaries_detached[self._teacher_of(name)]
                align_total, parts = self.align_loss(own_summaries, teacher)
                metrics[f"{name}/align_loss"] = float(align_total)
                metrics.update(
                    {f"{name}/align/{k}": float(v) for k, v in parts.items()}
                )
                total = total + align_total

            side.optimizer.zero_grad()
            total.backward()
            # clip_grad_norm_ returns the pre-clip total norm; max_norm=inf
            # measures without clipping (torch 2.5 has no get_total_norm).
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(side.params.values()), side.clip_grad_norm or float("inf")
            )
            metrics[f"{name}/grad_norm"] = float(grad_norm)
            side.optimizer.step()
            metrics[f"{name}/total_loss"] = float(total)

        self.step_count += 1
        if self.cka_every and self.step_count % self.cka_every == 0:
            metrics.update(self.measure_k_cka(batch))
        if self.log_fn is not None:
            self.log_fn({"step": self.step_count, **metrics})
        return metrics

    def state_dicts(self) -> dict:
        """Per side: {"params": ..., "optimizer": ...}. Optimizer state is
        needed to recompute the TRAINING-time V post hoc (AdamW's hypothetical
        step depends on the live moments). rebuild.load_run also accepts the
        old params-only format (SGD-era checkpoints)."""
        out = {}
        for name, side in self.sides.items():
            opt_sd = side.optimizer.state_dict()
            out[name] = {
                "params": {k: v.detach().cpu() for k, v in side.params.items()},
                "optimizer": opt_sd,
            }
        return out

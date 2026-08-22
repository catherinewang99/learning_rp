"""Measure cross-model CKA from a run's saved checkpoints (post hoc, any arm).

For every checkpoint in <run-dir> (step*.pt and final.pt) rebuilds both
models, loads the params, and computes on the run's own fixed val banks:
  * per-layer and mean K-CKA (representational similarity, higher = aligned)
  * per-layer and mean Π-CKA (plasticity similarity; uses the eval bank)
  * the full L x L K-CKA matrix (off-diagonal structure)
Writes <run-dir>/cka_checkpoints.jsonl (one line per checkpoint) and prints a
table. Optionally logs to wandb as a separate run named <run>-cka.

Usage:
    python scripts/measure_cka.py --run-dir runs/av-armA
    python scripts/measure_cka.py --run-dir runs/av-armA --device cuda \
        --wandb-project audiovis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.metrics import (cross_model_alignment, cross_model_cka_matrices,
                                  eval_summaries)
from src.training.rebuild import list_checkpoints, load_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-pi", action="store_true", help="K-CKA only (faster)")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default="cwang99-duke-university")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    ckpts = list_checkpoints(run_dir)
    if not ckpts:
        sys.exit(f"no step*.pt / final.pt in {run_dir}")
    print(f"{run_dir}: {len(ckpts)} checkpoints")

    wb = None
    if args.wandb_project:
        import wandb

        wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                        name=f"{run_dir.name}-cka", job_type="measure")

    out_path = run_dir / "cka_checkpoints.jsonl"
    out_path.unlink(missing_ok=True)
    print(f"{'step':>7}  {'K-CKA mean':>10}  {'Π-CKA mean':>10}  per-layer K-CKA")
    for step, ckpt in ckpts:
        cfg, sides, bank, kernel_fn = load_run(run_dir, ckpt.name, args.device)
        guide, target = "vision", "audio"
        sums = {}
        for name, side in sides.items():
            exps = [e.to(args.device) for e in bank["eval_experiences"][name]]
            probe = bank["probes"][name].to(args.device)
            sums[name] = eval_summaries(side, exps, probe, kernel_fn)
        sims = cross_model_alignment(sums[guide], sums[target])
        k_mat, _, g_names, t_names = cross_model_cka_matrices(sums[guide], sums[target])
        row = {"step": step, "checkpoint": ckpt.name,
               **{f"cka/{k}": v for k, v in sims.items()},
               "k_matrix": k_mat.tolist(), "guide_layers": g_names, "target_layers": t_names}
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if wb is not None:
            wb.log({k: v for k, v in row.items() if isinstance(v, (int, float))}, step=step)
        per_layer = " ".join(f"{sims[f'k_cka/{g}->{t}']:.3f}" for g, t in zip(g_names, t_names))
        print(f"{step:>7}  {sims['k_cka/mean']:>10.4f}  {sims['pi_cka/mean']:>10.4f}  {per_layer}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

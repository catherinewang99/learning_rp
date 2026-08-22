"""Run logging: wandb + local JSONL.

Mirrors crossmodal-prior's structure: every metric dict goes to wandb AND to
<outdir>/metrics.jsonl, so curves can be re-plotted offline (and runs survive
wandb outages / preemption). scripts/visualize_av.py consumes the JSONL.

Convention (also from crossmodal-prior): there is NO default wandb project.
When wandb is enabled the project must be provided explicitly (train scripts
take --wandb-project), so every tracked run is a project you named yourself.
"""

from __future__ import annotations

import json
from pathlib import Path

_NO_PROJECT_MSG = (
    "wandb is enabled but no project is set. There is deliberately no default: "
    "pass --wandb-project <name> on the command line (or set wandb.enabled: false)."
)


class RunLogger:
    """log(metrics: dict) -> appends one JSONL line and forwards to wandb.

    Include a "step" key (training step) in every dict so all charts share
    one x-axis; wandb is given it as its step explicitly.
    """

    def __init__(self, cfg: dict, outdir: str | Path | None = None):
        self._wandb = None
        wandb_cfg = cfg.get("wandb", {})
        if wandb_cfg.get("enabled", False):
            if not wandb_cfg.get("project"):
                raise ValueError(_NO_PROJECT_MSG)
            import wandb

            run = wandb.init(
                project=wandb_cfg["project"],
                entity=wandb_cfg.get("entity"),   # None -> account default; set
                #   explicitly (crossmodal convention) so runs land where you look
                name=wandb_cfg.get("name"),
                config=cfg,
            )
            print(f"[lrp] wandb run: {run.url}", flush=True)
            self._wandb = wandb
        self._path = None
        if outdir is not None:
            Path(outdir).mkdir(parents=True, exist_ok=True)
            self._path = Path(outdir) / "metrics.jsonl"

    def __call__(self, metrics: dict):
        if self._path is not None:
            with open(self._path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        if self._wandb is not None:
            self._wandb.log(metrics, step=metrics.get("step"))

    def log_image(self, key: str, path: str | Path, step: int | None = None):
        if self._wandb is not None:
            self._wandb.log({key: self._wandb.Image(str(path))}, step=step)


def maybe_wandb(cfg: dict):
    """Back-compat helper (offline track): log_fn or None."""
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    if not wandb_cfg.get("project"):
        raise ValueError(_NO_PROJECT_MSG)
    import wandb

    run = wandb.init(project=wandb_cfg["project"], entity=wandb_cfg.get("entity"),
                     name=wandb_cfg.get("name"), config=cfg)
    print(f"[lrp] wandb run: {run.url}", flush=True)
    return wandb.log

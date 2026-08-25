"""Matched-layout path evaluation + 2D board visualization.

Both agents are dropped into the SAME layout (same spawn pose, same goal) and
run their deterministic policy (action = mean) to episode end. Their paths are
drawn on one board — the most direct picture of behavioral convergence. Also
yields the path-divergence metric: mean distance between the two paths at
matched timesteps.

Episodes are simulated with the PURE kinematics (kinematic_step/reward-free),
touching none of the live training envs: vision observes via the side's
arena.render_at (stateless scratch render), audio via its sensor from the
accumulated distance history.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .arena import ArenaConfig, bearing, kinematic_step
from .cross_render import render_state


def run_matched_episode(side, pose: np.ndarray, goal: np.ndarray,
                        max_steps: int = 200, noise_offset: int = 500_000) -> dict:
    """Deterministic episode from a fixed layout; returns the 2D path."""
    cfg: ArenaConfig = side.arena.cfg
    pose = np.array(pose, dtype=np.float64)
    goal = np.array(goal, dtype=np.float64)
    step_id = 0
    cue_hist = [[float(np.linalg.norm(pose[:2] - goal)), bearing(pose, goal),
                 float(noise_offset)]]
    phase = 0
    path = [pose[:2].copy()]
    reached = False
    with torch.no_grad():
        for _ in range(max_steps):
            obs = render_state(side.modality, side, pose, goal,
                               np.asarray(cue_hist), phase).unsqueeze(0).to(side.device)
            mean, _, _ = side.probed.forward_output(side.detached_params(), obs, side.buffers)
            action = torch.tanh(mean[0]).cpu().numpy()
            pose = kinematic_step(pose, action, cfg)
            d = float(np.linalg.norm(pose[:2] - goal))
            step_id += 1
            cue_hist.append([d, bearing(pose, goal), float(noise_offset + step_id)])
            cue_hist = cue_hist[-(side.window_steps + 1):]
            phase += side.sensor.step_samples if side.sensor else 1
            path.append(pose[:2].copy())
            if d < cfg.reach_threshold:
                reached = True
                break
    return {"path": np.stack(path), "reached": reached, "steps": len(path) - 1}


def path_divergence(path_a: np.ndarray, path_b: np.ndarray) -> float:
    """Mean distance between the two paths at matched timesteps (shorter path
    padded with its final position — a finished agent 'waits' at its endpoint)."""
    n = max(len(path_a), len(path_b))

    def pad(p):
        return np.concatenate([p, np.repeat(p[-1:], n - len(p), axis=0)]) if len(p) < n else p

    return float(np.linalg.norm(pad(path_a) - pad(path_b), axis=1).mean())


def matched_layout_eval(sides: dict, layouts: list, max_steps: int = 200) -> tuple[dict, list]:
    """Run every side on every layout. Returns (metrics, per-layout records)."""
    records = []
    for pose, goal in layouts:
        rec = {"pose": pose, "goal": goal, "episodes": {}}
        for layout_idx, (name, side) in enumerate(sides.items()):
            rec["episodes"][name] = run_matched_episode(
                side, pose, goal, max_steps,
                noise_offset=500_000 + len(records) * 1000)
        records.append(rec)

    names = list(sides)
    div = [path_divergence(r["episodes"][names[0]]["path"], r["episodes"][names[1]]["path"])
           for r in records]
    metrics = {"behavior/path_divergence": float(np.mean(div))}
    for name in names:
        metrics[f"behavior/{name}_matched_success"] = float(
            np.mean([r["episodes"][name]["reached"] for r in records]))
    return metrics, records


def plot_matched_paths(records: list, arena_cfg: ArenaConfig, out_path: str | Path,
                       colors: dict | None = None) -> Path:
    """One panel per layout: arena square, goal star, spawn marker, one colored
    path per agent (solid = reached, dotted = timed out)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = colors or {"vision": "#1f6fb4", "audio": "#e07b26"}
    n = len(records)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows), squeeze=False)
    h = arena_cfg.half_extent

    for i, rec in enumerate(records):
        ax = axes[i // cols][i % cols]
        ax.add_patch(plt.Rectangle((-h, -h), 2 * h, 2 * h, fill=False, lw=1.5, color="0.3"))
        goal = np.asarray(rec["goal"])
        ax.plot(*goal, marker="*", ms=18, color="#2fa02f", mec="0.2", zorder=5)
        circle = plt.Circle(goal, arena_cfg.reach_threshold, fill=False,
                            ls=":", color="#2fa02f", lw=0.8)
        ax.add_patch(circle)
        start = np.asarray(rec["pose"])[:2]
        ax.plot(*start, marker="o", ms=8, color="0.2", zorder=5)
        titles = []
        for name, ep in rec["episodes"].items():
            path = ep["path"]
            style = "-" if ep["reached"] else ":"
            ax.plot(path[:, 0], path[:, 1], style, color=colors.get(name, None),
                    lw=1.8, alpha=0.9, label=name)
            # visible endpoint even when the whole path is a dot at spawn
            # (a not-yet-learning agent stops or scribbles in place)
            ax.plot(*path[-1], marker="x", ms=9, mew=2.5,
                    color=colors.get(name, None), zorder=6)
            net = float(np.linalg.norm(path[-1] - path[0]))
            titles.append(f"{name}: {'✓' if ep['reached'] else '✗'} "
                          f"{ep['steps']} ({net:.1f}m)")
        ax.set_title(" | ".join(titles), fontsize=8)
        ax.set_xlim(-h - 0.2, h + 0.2)
        ax.set_ylim(-h - 0.2, h + 0.2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=7, loc="lower left")
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("matched layouts: same spawn + goal, each agent's deterministic path",
                 fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

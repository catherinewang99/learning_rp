"""Cross-rendering: the same world state/transition, seen by each modality.

This is what makes Π comparable across agents whose BEHAVIOR is independent:
alignment experiences are the teacher's own window transitions, re-rendered
into the student's modality from the stored pose/distance histories. Also
builds the fixed probe bank (K's X) and the fixed eval-transition bank
(measurement Π), both deterministic given seed.

Audio rendering is a pure function of (distance history, clip phase), so a
teacher transition cross-renders from bookkeeping alone; vision cross-renders
via arena.render_at(pose, goal). Probe-bank audio uses a stationary history
(agent sat at the pose) — a documented statistics mismatch vs. moving
rollouts.
"""

from __future__ import annotations

import numpy as np
import torch

from ..data.experiences import Experience
from .arena import bearing, kinematic_step, reward_fn


def render_state(modality: str, side, pose, goal, cue_hist, phase) -> torch.Tensor:
    """One observation (C, H, W) of a world state in the given modality.
    ``side`` supplies the renderers (its own arena / sensor). cue_hist:
    (W, 2) [distance, bearing] rows (audio; vision ignores it)."""
    if modality == "vision":
        return torch.from_numpy(side.arena.render_at(np.asarray(pose), np.asarray(goal)))
    return side.sensor.observe_traj(np.asarray(cue_hist), int(phase), side.render_rng)


def cross_render_experiences(student, window, picks: list[tuple[int, int]],
                             gamma: float) -> list[Experience]:
    """Teacher window transitions -> the student's modality, with the
    student's OWN quantities in y (its logp of the teacher's action -> ratio 1,
    its critic's 1-step TD advantage). picks = [(t, b)] with no reset at t."""
    experiences = []
    with torch.no_grad():
        for t, b in picks:
            x = render_state(student.modality, student,
                             window["pose"][t, b], window["goal"][t, b],
                             window["cue_hist"][t, b], window["phase"][t, b]).unsqueeze(0)
            x_next = render_state(student.modality, student,
                                  window["pose"][t + 1, b], window["goal"][t, b],
                                  window["cue_hist"][t + 1, b],
                                  window["phase"][t + 1, b]).unsqueeze(0)
            x, x_next = x.to(student.device), x_next.to(student.device)
            action = window["action"][t, b : b + 1].to(student.device)
            reward = float(window["reward"][t, b])

            mean, log_std, value = student.probed.forward_output(
                student.detached_params(), x, student.buffers)
            _, _, value_next = student.probed.forward_output(
                student.detached_params(), x_next, student.buffers)
            from .policy import squashed_logp_entropy

            logp, _ = squashed_logp_entropy(mean, log_std, action)
            target = reward + gamma * value_next
            experiences.append(Experience(
                x=x,
                y={"action": action, "logp_old": logp.detach(),
                   "advantage": (target - value).detach(),
                   "value_target": target.detach()},
            ))
    return experiences


def teacher_experiences(window, picks: list[tuple[int, int]], device) -> list[Experience]:
    """The teacher's own transitions with its rollout quantities (GAE
    advantage, stored logp) — exactly what its real PPO update consumed."""
    out = []
    for t, b in picks:
        out.append(Experience(
            x=window["obs"][t, b : b + 1].to(device),
            y={"action": window["action"][t, b : b + 1].to(device),
               "logp_old": window["logp"][t, b : b + 1].to(device),
               "advantage": window["advantage"][t, b : b + 1].to(device),
               "value_target": window["returns"][t, b : b + 1].to(device)},
        ))
    return out


def pick_transitions(window, m: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """m random (t, b) with no reset at t (so t+1 is the true successor)."""
    t_len, b_len = window["reward"].shape
    valid = [(t, b) for t in range(t_len) for b in range(b_len)
             if not bool(window["reset"][t, b])]
    idx = rng.choice(len(valid), size=min(m, len(valid)), replace=False)
    return [valid[i] for i in idx]


def pick_transitions_multi(windows: list, m: int,
                           rng: np.random.Generator) -> list[tuple[int, int, int]]:
    """m random (window_idx, t, b) pooled UNIFORMLY over all valid transitions
    of the stored window history — the alignment event samples the teacher's
    whole inter-alignment period, not just the boundary window.

    NOTE on staleness: older windows' logp/advantages were computed under the
    teacher's then-current params; its V is evaluated at the boundary params,
    so those experiences enter with PPO ratio != 1 (importance-weighted
    exactly as the real PPO loss would weight them) and mildly stale advantage
    estimates — the accepted semantics of "the update it would take NOW on a
    remembered experience"."""
    valid: list[tuple[int, int, int]] = []
    for w_idx, window in enumerate(windows):
        t_len, b_len = window["reward"].shape
        valid.extend((w_idx, t, b) for t in range(t_len) for b in range(b_len)
                     if not bool(window["reset"][t, b]))
    idx = rng.choice(len(valid), size=min(m, len(valid)), replace=False)
    return [valid[i] for i in idx]


def group_picks(picks: list[tuple[int, int, int]]) -> dict[int, list[tuple[int, int]]]:
    """(w_idx, t, b) picks -> {w_idx: [(t, b)...]} preserving order within
    groups; iterate groups in sorted w_idx order on BOTH teacher and student
    so Π rows stay paired."""
    grouped: dict[int, list[tuple[int, int]]] = {}
    for w_idx, t, b in picks:
        grouped.setdefault(w_idx, []).append((t, b))
    return dict(sorted(grouped.items()))


# ---- fixed banks ------------------------------------------------------------


def build_probe_bank(sides: dict, n: int, seed: int) -> dict:
    """n world states rendered in every side's modality. Rows sorted by
    distance-to-goal so kernel heatmaps show task structure directly."""
    rng = np.random.default_rng(seed)
    any_side = next(iter(sides.values()))
    layouts = [any_side.arena._sample_layout() for _ in range(n)]
    dists = np.array([np.linalg.norm(p[:2] - g) for p, g in layouts])
    order = np.argsort(dists)
    layouts = [layouts[i] for i in order]
    dists = dists[order]

    probes = {}
    for name, side in sides.items():
        w = getattr(side, "window_steps", 1)
        obs = []
        for idx, ((pose, goal), d) in enumerate(zip(layouts, dists)):
            cues = np.column_stack([np.full(w, d), np.full(w, bearing(pose, goal)),
                                    10_000 + idx * w + np.arange(w, dtype=np.float64)])
            obs.append(render_state(side.modality, side, pose, goal, cues, 0))
        probes[name] = torch.stack(obs)
    return {"probes": probes, "dist": torch.from_numpy(dists), "layouts": layouts}


def build_eval_transitions(sides: dict, cfg, n: int, seed: int, gamma_unused=None) -> dict:
    """n fixed (s, a, r, s') transitions rendered in every modality — the
    shared measurement bank for tracked Π-CKA. Actions are random; next state
    and reward come from the pure kinematics, no simulator stepping."""
    rng = np.random.default_rng(seed)
    any_side = next(iter(sides.values()))
    rows = []
    for _ in range(n):
        pose, goal = any_side.arena._sample_layout()
        action = rng.uniform(-1, 1, 2)
        pose_next = kinematic_step(pose, action, cfg)
        d0 = float(np.linalg.norm(pose[:2] - goal))
        d1 = float(np.linalg.norm(pose_next[:2] - goal))
        reward = reward_fn(d0, d1, d1 < cfg.reach_threshold, cfg)
        rows.append((pose, pose_next, goal, action, reward, d0, d1))

    per_side = {}
    for name, side in sides.items():
        w = getattr(side, "window_steps", 1)
        obs, obs_next = [], []
        for idx, (pose, pose_next, goal, _, _, d0, d1) in enumerate(rows):
            ids = 100_000 + idx * (w + 1) + np.arange(w + 1, dtype=np.float64)
            hist = np.column_stack([np.full(w, d0), np.full(w, bearing(pose, goal)),
                                    ids[:w]])
            next_row = [[d1, bearing(pose_next, goal), ids[w]]]
            # x_next shares the overlap ids with x -> consistent shared past
            hist_next = np.concatenate([hist[1:], next_row]) if w > 1 else                 np.array(next_row)
            obs.append(render_state(side.modality, side, pose, goal, hist, 0))
            obs_next.append(render_state(side.modality, side, pose_next, goal,
                                         hist_next, 0))
        per_side[name] = {"obs": torch.stack(obs), "obs_next": torch.stack(obs_next)}
    return {
        "sides": per_side,
        "action": torch.tensor(np.array([r[3] for r in rows]), dtype=torch.float32),
        "reward": torch.tensor([r[4] for r in rows], dtype=torch.float32),
    }


def eval_experiences_for(side, bank: dict, gamma: float) -> list[Experience]:
    """Turn the shared eval bank into this side's experiences (own critic TD,
    own logp -> ratio 1) at its CURRENT detached params."""
    data = bank["sides"][side.name]
    experiences = []
    with torch.no_grad():
        obs = data["obs"].to(side.device)
        obs_next = data["obs_next"].to(side.device)
        actions = bank["action"].to(side.device)
        mean, log_std, value = side.probed.forward_output(side.detached_params(), obs, side.buffers)
        _, _, value_next = side.probed.forward_output(side.detached_params(), obs_next, side.buffers)
        from .policy import squashed_logp_entropy

        logp, _ = squashed_logp_entropy(mean, log_std, actions)
        target = bank["reward"].to(side.device) + gamma * value_next
        for i in range(len(actions)):
            experiences.append(Experience(
                x=obs[i : i + 1],
                y={"action": actions[i : i + 1], "logp_old": logp[i : i + 1],
                   "advantage": (target - value)[i : i + 1],
                   "value_target": target[i : i + 1]},
            ))
    return experiences

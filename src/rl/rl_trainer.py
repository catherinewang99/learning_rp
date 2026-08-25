"""The RL loop: independent navigation, plasticity-kernel coupling.

Every window (default 10 env steps x B envs) per side:
  1. collect the window with the current (detached) policy in the side's OWN
     envs — trajectories are fully independent across sides;
  2. GAE with a bootstrap at the window boundary;
  3. one AdamW update on the window's PPO loss.
Every ``align_every``-th window (default 10 -> alignment every ~100 env
steps), guided sides ADD the layerwise alignment loss to that same update:
the teacher's plasticity summary is computed on experiences sampled from its
LAST ``align_every`` WINDOWS (the whole inter-alignment period, kept in a
rolling history), cross-rendered into the student's modality. Between
alignment events, updates are pure PPO and no teacher machinery runs.
The rule that defines V is the same AdamW + per-transition PPO loss the real
update uses (honesty invariant, as in the other tracks).
"""

from __future__ import annotations

import numpy as np
import torch

from ..kernels.gram import linear_gram
from ..kernels.plasticity import plasticity_summary, representation_summary
from ..rules import AdamWRule
from ..training.metrics import cross_model_alignment
from .arena import VecArena, bearing
from .audio_sensor import AudioSensor
from .cross_render import (cross_render_experiences, eval_experiences_for,
                           group_picks, pick_transitions_multi,
                           teacher_experiences)
from .tasks import make_ppo_task


class RLSide:
    """One agent: its own envs, sensor, policy, functional params, optimizer."""

    def __init__(self, name: str, modality: str, arena: VecArena, probed,
                 optimizer_cfg: dict, ppo_cfg: dict, device: str = "cpu",
                 sensor: AudioSensor | None = None, seed: int = 0,
                 cue_steps: int | None = None, phase_step: int | None = None,
                 noise_id_base: int | None = None):
        self.name, self.modality, self.arena, self.sensor = name, modality, arena, sensor
        self.device = device
        self.probed = probed.to(device)
        self.params = {k: v.to(device).requires_grad_(True) for k, v in probed.params().items()}
        self.buffers = {k: v.to(device) for k, v in probed.buffers_dict().items()}
        self.task = make_ppo_task(ppo_cfg.get("clip_coef", 0.2),
                                  ppo_cfg.get("value_coef", 0.5),
                                  ppo_cfg.get("entropy_coef", 0.01),
                                  ppo_cfg.get("mean_reg", 1e-3))
        opt = optimizer_cfg
        self.rule = AdamWRule(lr=opt["lr"], betas=tuple(opt.get("betas", (0.9, 0.999))),
                              eps=opt.get("eps", 1e-8),
                              weight_decay=opt.get("weight_decay", 0.0),
                              clip_grad_norm=opt.get("clip_grad_norm"), task=self.task)
        self.optimizer = torch.optim.AdamW(
            list(self.params.values()), lr=opt["lr"], betas=tuple(opt.get("betas", (0.9, 0.999))),
            eps=opt.get("eps", 1e-8), weight_decay=opt.get("weight_decay", 0.0))
        self.clip_grad_norm = opt.get("clip_grad_norm")
        self.render_rng = np.random.default_rng(seed + 7)

        b = arena.cfg.num_envs
        # Cue bookkeeping serves CROSS-RENDERING into the audio modality, so
        # its geometry — rows kept per env (cue_steps) and clip samples per
        # step (phase_step) — must come from the shared AUDIO config on EVERY
        # side: a vision teacher has to hand the audio student real moving
        # histories and honest clip phases, not 1-row stationary stubs with
        # phase ticking by 1. Defaults (own sensor, else 1) only cover
        # audio-only or unguided setups; pass both explicitly when this side
        # can ever be a teacher.
        self.window_steps = cue_steps if cue_steps is not None else (
            sensor.cfg.window_steps if sensor is not None else 1)
        self.phase_step = phase_step if phase_step is not None else (
            sensor.step_samples if sensor is not None else 1)
        # per-side id namespace: without an offset, a cross-rendered teacher
        # step would reuse the exact noise chunk the student heard at the same
        # step index of its own rollout
        self.noise_id_base = (self.ROLLOUT_NOISE_BASE if noise_id_base is None
                              else noise_id_base)
        self.env_steps = 0                      # per-env step clock (all envs step together)
        # per-env history of [distance, bearing, noise_id] cue rows (bearing
        # measured with that step's heading) — what the audio sensor renders
        # from; the id freezes that step's sensor noise (must init AFTER
        # env_steps: _cue reads the clock)
        self.cue_hist = [[self._cue(i)] for i in range(b)]
        self.phase = np.zeros(b, dtype=np.int64)
        self.episode_return = np.zeros(b)
        self.episode_len = np.zeros(b, dtype=np.int64)
        # (completion_step, success, return, length) per finished episode.
        # NOTE the length bias in any completed-episode statistic: short
        # (successful) episodes recycle faster and are over-represented; the
        # unbiased uniform-over-layouts measure is behavior/*_matched_success.
        self.finished: list[tuple[int, bool, float, int]] = []

    ROLLOUT_NOISE_BASE = 1_000_000   # id ranges: rollout / probes / eval / paths

    def _cue(self, i: int) -> list[float]:
        """[distance, bearing, noise_id]: the id is unique per (env, step) so
        this step's sensor noise is frozen — re-observing or cross-rendering
        the step reproduces the identical waveform chunk."""
        noise_id = self.noise_id_base + self.env_steps * self.arena.cfg.num_envs + i
        return [float(self.arena.dists()[i]),
                bearing(self.arena.poses[i], self.arena.goals[i]),
                float(noise_id)]

    def detached_params(self):
        return {k: v.detach() for k, v in self.params.items()}

    def sync_rule_state(self):
        self.rule.sync_state(self.optimizer, self.params)

    # -- observation of the CURRENT env state ---------------------------------

    def observe(self) -> torch.Tensor:
        if self.modality == "vision":
            return self.arena.observe().to(self.device)
        obs = [self.sensor.observe_traj(np.asarray(h), int(p), self.render_rng)
               for h, p in zip(self.cue_hist, self.phase)]
        return torch.stack(obs).to(self.device)

    # -- window collection ----------------------------------------------------

    def collect_window(self, t_len: int) -> dict:
        cfg = self.arena.cfg
        b, w = cfg.num_envs, self.window_steps
        window = {k: [] for k in
                  ("obs", "action", "logp", "value", "reward", "reset", "done",
                   "pmean", "pose", "goal", "cue_hist", "phase")}
        for _ in range(t_len):
            window["pose"].append(self.poses_padded())
            window["goal"].append(self.arena.goals.copy())
            window["cue_hist"].append(self.hist_snapshot())
            window["phase"].append(self.phase.copy())
            obs = self.observe()
            with torch.no_grad():
                mean, log_std, value = self.probed.forward_output(
                    self.detached_params(), obs, self.buffers)
                from .policy import squashed_sample

                action, logp = squashed_sample(mean, log_std)
            act_np = action.cpu().numpy()
            out = self.arena.step(act_np)
            self.episode_return += out["reward"]
            self.episode_len += 1
            self.env_steps += 1
            for i in range(b):
                self.phase[i] += self.phase_step
                if out["reset"][i]:
                    self.finished.append((self.env_steps, bool(out["done"][i]),
                                          float(self.episode_return[i]),
                                          int(self.episode_len[i])))
                    self.episode_return[i] = 0.0
                    self.episode_len[i] = 0
                    self.cue_hist[i] = [self._cue(i)]
                else:
                    self.cue_hist[i].append(self._cue(i))
                    self.cue_hist[i] = self.cue_hist[i][-(w + 1):]
            window["obs"].append(obs.cpu())
            window["action"].append(action.cpu())
            window["pmean"].append(mean.cpu())
            window["logp"].append(logp.cpu())
            window["value"].append(value.cpu())
            window["reward"].append(torch.from_numpy(out["reward"]).float())
            window["reset"].append(torch.from_numpy(out["reset"]))
            window["done"].append(torch.from_numpy(out["done"]))
        # final state row (for next-obs cross-rendering and the bootstrap)
        window["pose"].append(self.poses_padded())
        window["cue_hist"].append(self.hist_snapshot())
        window["phase"].append(self.phase.copy())

        out = {}
        for k, v in window.items():
            out[k] = (torch.stack(v) if torch.is_tensor(v[0])
                      else torch.from_numpy(np.stack(v)))
        with torch.no_grad():
            _, _, bootstrap = self.probed.forward_output(
                self.detached_params(), self.observe(), self.buffers)
        out["bootstrap"] = bootstrap.cpu()
        return out

    def poses_padded(self) -> np.ndarray:
        return self.arena.poses.copy()

    def hist_snapshot(self) -> np.ndarray:
        """(B, W, 3) [distance, bearing, noise_id] rows, oldest-padded."""
        w = self.window_steps
        rows = []
        for h in self.cue_hist:
            c = np.asarray(h, dtype=np.float64).reshape(-1, 3)
            if len(c) < w:
                c = np.concatenate([np.repeat(c[:1], w - len(c), axis=0), c])
            rows.append(c[-w:])
        return np.stack(rows)

    def episode_stats(self, horizon_steps: int) -> dict[str, float]:
        """Stats over episodes completed within the last ``horizon_steps`` env
        steps — time-windowed, so slow failures aren't crowded out of a
        fixed-count buffer by fast successes (they still complete less often;
        see the bias note above; matched_success is the unbiased metric)."""
        self.finished = self.finished[-5000:]
        recent = [f for f in self.finished if f[0] > self.env_steps - horizon_steps]
        if not recent:
            return {}
        return {
            "success_rate": float(np.mean([f[1] for f in recent])),
            "episode_return": float(np.mean([f[2] for f in recent])),
            "episode_len": float(np.mean([f[3] for f in recent])),
            "episodes_recent": float(len(recent)),
        }

    # -- the real update's loss (vectorized == mean per-transition task) ------

    def window_ppo_loss(self, window: dict, params):
        """Returns (total loss WITH graph, detached component parts dict:
        policy_loss / value_loss / entropy / saturation_penalty / clip_frac —
        unweighted, so config weights can be reasoned about separately)."""
        t_len, b = window["reward"].shape
        obs = window["obs"].reshape(t_len * b, *window["obs"].shape[2:]).to(self.device)
        y = {"action": window["action"].reshape(t_len * b, -1).to(self.device),
             "logp_old": window["logp"].reshape(-1).to(self.device),
             "advantage": window["advantage"].reshape(-1).to(self.device),
             "value_target": window["returns"].reshape(-1).to(self.device)}
        from ..data.experiences import Experience

        return self.task.components(self.probed, params, Experience(x=obs, y=y), self.buffers)


class RLTrainer:
    def __init__(self, sides: dict[str, RLSide], guided: list[str], align_loss,
                 window_len: int = 10, m_per_window: int = 12,
                 align_every: int = 10,     # alignment rides every k-th window
                 #   update; the k-1 in between are pure PPO. Teacher
                 #   experiences pool over its last k windows.
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 probe_bank: dict | None = None, eval_bank: dict | None = None,
                 kernel_fn=linear_gram, use_checkpoint: bool = True,
                 device: str = "cpu", seed: int = 0, log_fn=None,
                 stats_horizon: int = 2000,     # env steps for episode stats
                 n_matched_layouts: int = 8):   # fixed layouts for path eval
        assert set(guided) <= set(sides)
        self.sides, self.guided, self.align_loss = sides, guided, align_loss
        self.window_len, self.m_per_window = window_len, m_per_window
        self.align_every = max(1, align_every)
        from collections import deque

        # rolling per-side history of the last align_every windows (CPU
        # tensors); the pool the alignment event samples from
        self.history: dict[str, deque] = {
            name: deque(maxlen=self.align_every) for name in sides}
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.probe_bank, self.eval_bank = probe_bank, eval_bank
        self.kernel_fn, self.use_checkpoint = kernel_fn, use_checkpoint
        self.device, self.log_fn = device, log_fn
        self.rng = np.random.default_rng(seed)
        self.needs = getattr(align_loss, "needs", {"K", "V", "Pi"}) if guided else set()
        self.window_count = 0
        self.stats_horizon = stats_horizon
        # fixed matched layouts (same spawn+goal for BOTH agents) for the
        # path-divergence eval and the 2D board plots — the unbiased
        # uniform-over-layouts success measure
        any_side = next(iter(sides.values()))
        self.matched_layouts = [any_side.arena._sample_layout()
                                for _ in range(n_matched_layouts)]

    def _teacher_of(self, name: str) -> str:
        others = [s for s in self.sides if s != name]
        assert len(others) == 1
        return others[0]

    def _summary(self, side: RLSide, params, experiences):
        probes = self.probe_bank["probes"][side.name].to(side.device)
        if self.needs <= {"K"}:
            return representation_summary(side.probed, params, probes, side.buffers,
                                          self.kernel_fn)
        return plasticity_summary(side.probed, params, side.rule, experiences, probes,
                                  side.buffers, self.kernel_fn,
                                  use_checkpoint=self.use_checkpoint)

    def window(self) -> dict:
        from .ppo import gae

        metrics: dict[str, float] = {}
        windows: dict[str, dict] = {}
        for name, side in self.sides.items():
            w = side.collect_window(self.window_len)
            adv, ret = gae(w["reward"], w["value"], w["reset"], w["bootstrap"],
                           self.gamma, self.gae_lambda)
            w["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)
            w["returns"] = ret
            windows[name] = w
            metrics[f"{name}/reward_mean"] = float(w["reward"].mean())
            metrics.update({f"{name}/{k}": v
                            for k, v in side.episode_stats(self.stats_horizon).items()})
            # policy-health diagnostics (saturation was invisible before):
            # |pre-tanh mean| large => near-deterministic railed actions
            metrics[f"{name}/policy_mean_abs"] = float(w["pmean"].abs().mean())
            metrics[f"{name}/action_sat_frac"] = float(
                (w["action"].abs() > 0.95).float().mean())
            metrics[f"{name}/log_std"] = float(side.params["log_std"].detach().mean())
            # entropy of the EXECUTED dist, estimated from collection samples
            # (valid as a metric; useless as a loss — E[score] = 0 on stored
            # actions). Plummeting while log_std holds = mean-driven collapse.
            metrics[f"{name}/exec_entropy_est"] = float(-w["logp"].mean())


        for name in self.sides:
            self.history[name].append(windows[name])

        align_now = (self.window_count + 1) % self.align_every == 0
        # teacher summaries (detached), pooled over the teacher's stored
        # window history — only computed on alignment windows
        summaries_detached: dict[str, tuple] = {}
        if align_now:
            for name in {self._teacher_of(g) for g in self.guided}:
                side = self.sides[name]
                side.sync_rule_state()
                hist = list(self.history[name])
                picks = group_picks(pick_transitions_multi(hist, self.m_per_window, self.rng))
                experiences = []
                for w_idx, sub in picks.items():
                    experiences.extend(teacher_experiences(hist[w_idx], sub, side.device))
                summaries_detached[name] = (
                    self._summary(side, side.detached_params(), experiences), picks)

        for name, side in self.sides.items():
            side.sync_rule_state()
            task_loss, parts = side.window_ppo_loss(windows[name], side.params)
            metrics[f"{name}/ppo_loss"] = float(task_loss)
            metrics.update({f"{name}/{k}": v for k, v in parts.items()})
            total = task_loss

            if name in self.guided and align_now:
                teacher_name = self._teacher_of(name)
                teacher_sum, picks = summaries_detached[teacher_name]
                teacher_hist = list(self.history[teacher_name])
                student_exps = []
                for w_idx, sub in picks.items():   # same grouped order as the
                    student_exps.extend(cross_render_experiences(   # teacher ->
                        side, teacher_hist[w_idx], sub, self.gamma))  # rows pair
                own = self._summary(side, side.params, student_exps)
                align_total, parts = self.align_loss(own, teacher_sum)
                metrics[f"{name}/align_loss"] = float(align_total)
                metrics.update({f"{name}/align/{k}": float(v) for k, v in parts.items()})
                total = total + align_total

            side.optimizer.zero_grad()
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(side.params.values()), side.clip_grad_norm or float("inf"))
            metrics[f"{name}/grad_norm"] = float(grad_norm)
            side.optimizer.step()
            metrics[f"{name}/total_loss"] = float(total)

        self.window_count += 1
        metrics["step"] = self.window_count * self.window_len
        if self.log_fn is not None:
            self.log_fn(metrics)
        return metrics

    # -- measurement (all arms identical) -------------------------------------

    def eval_summaries(self) -> dict:
        sums = {}
        for name, side in self.sides.items():
            side.sync_rule_state()
            experiences = eval_experiences_for(side, self.eval_bank, self.gamma)
            probes = self.probe_bank["probes"][name].to(side.device)
            sums[name] = plasticity_summary(side.probed, side.detached_params(),
                                            side.rule, experiences, probes,
                                            side.buffers, self.kernel_fn)
        return sums

    def tracked_eval(self) -> dict:
        names = list(self.sides)
        guide = self._teacher_of(self.guided[0]) if self.guided else names[0]
        target = next(n for n in names if n != guide)
        sums = self.eval_summaries()
        out = {f"eval/{k}": v
               for k, v in cross_model_alignment(sums[guide], sums[target]).items()}
        out.update({f"eval/{k}": v for k, v in self.behavioral_eval().items()})
        return out, sums

    def path_eval(self, max_steps: int = 200):
        """Matched-layout episodes for both agents: divergence metric + records
        for plot_matched_paths."""
        from .traj_viz import matched_layout_eval

        return matched_layout_eval(self.sides, self.matched_layouts, max_steps)

    def behavioral_eval(self) -> dict[str, float]:
        """Do the two policies act alike on the SAME world states? Mean action
        distance and symmetric Gaussian KL over the paired probe bank."""
        with torch.no_grad():
            stats = {}
            for name, side in self.sides.items():
                probes = self.probe_bank["probes"][name].to(side.device)
                mean, log_std, _ = side.probed.forward_output(
                    side.detached_params(), probes, side.buffers)
                stats[name] = (mean.cpu(), log_std.cpu())
            (m1, s1), (m2, s2) = stats.values()
            v1, v2 = (2 * s1).exp(), (2 * s2).exp()
            # KL of the squashed dists == KL of the base Gaussians (bijection)
            kl12 = (s2 - s1 + (v1 + (m1 - m2) ** 2) / (2 * v2) - 0.5).sum(-1)
            kl21 = (s1 - s2 + (v2 + (m2 - m1) ** 2) / (2 * v1) - 0.5).sum(-1)
            return {
                # distance between EXECUTED deterministic actions (tanh space)
                "behavior/action_dist": float(
                    (torch.tanh(m1) - torch.tanh(m2)).norm(dim=-1).mean()),
                "behavior/policy_kl_sym": float((kl12 + kl21).mean() / 2),
            }

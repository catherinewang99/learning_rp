"""RL track: kinematics, audio sensor, GAE, trainer wiring, kernel montage.

Uses a GL-free FakeArena (pure-numpy render_at) and a synthetic speech clip,
so nothing here needs MuJoCo rendering or the network.
"""

import numpy as np
import torch
from torch import nn

from src.losses import LayerwiseAlignmentLoss, build_loss
from src.models import ProbedModel
from src.rl.arena import ArenaConfig, kinematic_step, reward_fn
from src.rl.audio_sensor import AudioConfig, AudioSensor
from src.rl.cross_render import build_eval_transitions, build_probe_bank
from src.rl.policy import ActorCritic
from src.rl.ppo import gae
from src.rl.rl_trainer import RLSide, RLTrainer

CFG = ArenaConfig(num_envs=2, horizon=30, camera_hw=16, seed=0)
OPT = {"name": "adamw", "lr": 1e-3, "weight_decay": 0.01, "clip_grad_norm": 0.5}
PPO = {"clip_coef": 0.2, "value_coef": 0.5, "entropy_coef": 0.01}


class FakeArena:
    """Duck-typed VecArena: real kinematics/rewards, synthetic deterministic
    'camera' (pattern encodes goal direction + distance), no MuJoCo."""

    def __init__(self, cfg: ArenaConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        b = cfg.num_envs
        self.poses, self.goals = np.zeros((b, 3)), np.zeros((b, 2))
        self.t = np.zeros(b, dtype=np.int64)
        for i in range(b):
            self.reset_env(i)

    def _sample_layout(self):
        lim = self.cfg.half_extent - 3 * self.cfg.agent_radius
        while True:
            pose = np.array([*self.rng.uniform(-lim, lim, 2), self.rng.uniform(-np.pi, np.pi)])
            goal = self.rng.uniform(-lim, lim, 2)
            if np.linalg.norm(pose[:2] - goal) >= self.cfg.min_spawn_separation:
                return pose, goal

    def reset_env(self, i):
        self.poses[i], self.goals[i] = self._sample_layout()
        self.t[i] = 0

    def dists(self):
        return np.linalg.norm(self.poses[:, :2] - self.goals, axis=1)

    def step(self, actions):
        b = self.cfg.num_envs
        rewards, dones, resets = np.zeros(b), np.zeros(b, bool), np.zeros(b, bool)
        before = self.dists()
        for i in range(b):
            self.poses[i] = kinematic_step(self.poses[i], actions[i], self.cfg)
            self.t[i] += 1
        after = self.dists()
        for i in range(b):
            reached = after[i] < self.cfg.reach_threshold
            rewards[i] = reward_fn(before[i], after[i], reached, self.cfg)
            dones[i] = reached
            if reached or self.t[i] >= self.cfg.horizon:
                resets[i] = True
                self.reset_env(i)
        return {"reward": rewards, "done": dones, "reset": resets,
                "dist_before": before, "dist_after": after}

    def render_at(self, pose, goal, *_):
        from src.rl.arena import apply_state_mask

        hw = self.cfg.camera_hw
        rel = np.asarray(goal) - np.asarray(pose)[:2]
        img = np.zeros((3, hw, hw), dtype=np.float32)
        img[0] += np.tanh(rel[0])
        img[1] += np.tanh(rel[1])
        img[2] += 1.0 / (1.0 + np.linalg.norm(rel))
        img += 0.05 * np.sin(np.arange(hw))[None, None, :]  # spatial texture
        return apply_state_mask(img, pose, goal, self.cfg.state_mask)

    def observe(self):
        return torch.from_numpy(np.stack(
            [self.render_at(self.poses[i], self.goals[i]) for i in range(self.cfg.num_envs)]))


def make_sensor(window_steps=4, n_mels=16):
    # broadband synthetic "speech" (white noise): like real speech, energy is
    # spread across mel bins, so gain(d) moves the WHOLE spectrogram — a pure
    # tone would light one bin and let distance-noise dominate the mean.
    clip = np.random.default_rng(0).normal(size=32000)
    cfg = AudioConfig(window_steps=window_steps, n_mels=n_mels)
    return AudioSensor(cfg, seed=0, clip=clip)


def make_sides(guided_needs_audio=True):
    torch.manual_seed(0)
    sensor = make_sensor()
    sides = {}
    for idx, (name, in_ch, sens) in enumerate(
            (("vision", 3, None), ("audio", sensor.channels, sensor))):
        input_hw = (16, 16) if name == "vision" else (16, sensor.frames)
        net = ActorCritic(in_channels=in_ch, input_hw=input_hw,
                          widths=(8, 16), pool_after=(1, 2), trunk_dim=32)
        probed = ProbedModel(net, layer_types=[nn.Conv2d], drop_last=False)
        arena = FakeArena(ArenaConfig(num_envs=2, horizon=30, camera_hw=16,
                                      seed=0 if name == "vision" else 1))
        # mirror build_sides: shared audio cue geometry on both sides (either
        # side can teach), disjoint per-side noise-id namespaces
        sides[name] = RLSide(name, name, arena, probed, OPT, PPO, device="cpu",
                             sensor=sens, seed=0,
                             cue_steps=sensor.cfg.window_steps,
                             phase_step=sensor.step_samples,
                             noise_id_base=1_000_000 + idx * 100_000_000)
    return sides


def make_trainer(guided, loss_terms):
    sides = make_sides()
    probe_bank = build_probe_bank(sides, n=6, seed=3)
    eval_bank = build_eval_transitions(sides, CFG, n=5, seed=4)
    return RLTrainer(sides, guided=guided,
                     align_loss=LayerwiseAlignmentLoss(build_loss(loss_terms)),
                     window_len=5, m_per_window=4,
                     probe_bank=probe_bank, eval_bank=eval_bank, seed=0)


def test_kinematics_and_reward():
    cfg = ArenaConfig()
    pose = np.array([0.0, 0.0, 0.0])
    out = kinematic_step(pose, np.array([1.0, 0.0]), cfg)
    assert abs(out[0] - cfg.step_length) < 1e-9 and abs(out[1]) < 1e-9
    # stop action: no movement
    out = kinematic_step(pose, np.array([-1.0, 0.5]), cfg)
    assert np.allclose(out[:2], 0)
    # wall clamp
    edge = np.array([cfg.half_extent, 0.0, 0.0])
    out = kinematic_step(edge, np.array([1.0, 0.0]), cfg)
    assert out[0] <= cfg.half_extent - cfg.agent_radius + 1e-9
    assert reward_fn(1.0, 0.5, False, cfg) > reward_fn(0.5, 1.0, False, cfg)
    assert reward_fn(0.5, 0.2, True, cfg) > cfg.reward["reach_bonus"] - 1


def test_audio_louder_near_goal_and_binaural():
    sensor = make_sensor()
    near = sensor.observe_stationary(0.2, rng=np.random.default_rng(1))
    far = sensor.observe_stationary(4.0, rng=np.random.default_rng(1))
    assert near.shape == (2, 16, sensor.frames)  # binaural default
    assert float(near.mean()) > float(far.mean())  # louder = more log-mel energy
    # ILD: source on the left -> left channel louder (and mirrored)
    left_src = sensor.observe_stationary(1.0, bearing=np.pi / 2,
                                         rng=np.random.default_rng(2))
    right_src = sensor.observe_stationary(1.0, bearing=-np.pi / 2,
                                          rng=np.random.default_rng(2))
    assert float(left_src[0].mean()) > float(left_src[1].mean())
    assert float(right_src[1].mean()) > float(right_src[0].mean())
    # front bias breaks front/back ambiguity
    front = sensor.observe_stationary(1.0, bearing=0.0, rng=np.random.default_rng(3))
    back = sensor.observe_stationary(1.0, bearing=np.pi, rng=np.random.default_rng(3))
    assert float(front.mean()) > float(back.mean())
    # deterministic given the same rng seed; accepts bare distance histories
    a = sensor.observe_traj(np.linspace(3, 1, 4), 100, np.random.default_rng(7))
    b = sensor.observe_traj(np.linspace(3, 1, 4), 100, np.random.default_rng(7))
    assert torch.equal(a, b)
    # mono ablation keeps one channel
    from src.rl.audio_sensor import AudioConfig, AudioSensor as AS

    mono = AS(AudioConfig(window_steps=4, n_mels=16, binaural=False),
              clip=np.random.default_rng(0).normal(size=32000))
    assert mono.observe_stationary(1.0, rng=np.random.default_rng(1)).shape[0] == 1


def test_gae_matches_naive():
    torch.manual_seed(0)
    t_len, b = 6, 3
    rewards, values = torch.randn(t_len, b), torch.randn(t_len, b)
    resets = torch.zeros(t_len, b, dtype=torch.bool)
    resets[3, 1] = True
    bootstrap = torch.randn(b)
    adv, ret = gae(rewards, values, resets, bootstrap, gamma=0.9, lam=0.8)
    # naive per-env recursion
    for j in range(b):
        last, nv = 0.0, bootstrap[j]
        expect = torch.zeros(t_len)
        for t in reversed(range(t_len)):
            alive = 0.0 if resets[t, j] else 1.0
            delta = rewards[t, j] + 0.9 * nv * alive - values[t, j]
            last = delta + 0.9 * 0.8 * alive * last
            expect[t] = last
            nv = values[t, j]
        assert torch.allclose(adv[:, j], expect, atol=1e-5)
    assert torch.allclose(ret, adv + values)


def test_arm_a_independent_runs():
    trainer = make_trainer([], [])
    metrics = trainer.window()
    assert np.isfinite(metrics["vision/ppo_loss"]) and np.isfinite(metrics["audio/ppo_loss"])
    assert not any("align" in k for k in metrics)
    # loss components logged separately (unweighted) per side
    for name in ("vision", "audio"):
        for part in ("policy_loss", "value_loss", "entropy",
                     "saturation_penalty", "clip_frac"):
            assert np.isfinite(metrics[f"{name}/{part}"]), part
    assert 0.0 <= metrics["vision/clip_frac"] <= 1.0


def test_arm_c_pi_guidance_runs_and_moves_student():
    trainer = make_trainer(["audio"], [{"name": "cka_pi", "weight": 1.0}])
    vision_before = {k: v.detach().clone() for k, v in trainer.sides["vision"].params.items()}
    metrics = trainer.window()
    assert np.isfinite(metrics["audio/align_loss"])
    assert metrics["audio/total_loss"] != metrics["audio/ppo_loss"]
    assert "vision/align_loss" not in metrics
    # teacher trained by its own PPO only (params moved, but not via align graph)
    assert any(not torch.equal(vision_before[k], trainer.sides["vision"].params[k].detach())
               for k in vision_before)
    # second window exercises optimizer-state sync in the rule
    metrics = trainer.window()
    assert np.isfinite(metrics["audio/align_loss"])


def test_tracked_eval_and_kernel_montage(tmp_path):
    trainer = make_trainer(["audio"], [{"name": "cka_pi", "weight": 1.0}])
    trainer.window()
    out, sums = trainer.tracked_eval()
    assert 0.0 <= out["eval/k_cka/mean"] <= 1.0 + 1e-5
    assert 0.0 <= out["eval/pi_cka/mean"] <= 1.0 + 1e-5
    assert out["eval/behavior/policy_kl_sym"] >= 0.0
    assert "eval/behavior/action_dist" in out

    from src.training.kernel_viz import kernel_montage

    path = kernel_montage(sums, "conv2", tmp_path / "montage.png",
                          order_note="probes sorted by dist-to-goal")
    assert path.exists() and path.stat().st_size > 1000


def test_matched_paths_and_divergence(tmp_path):
    from src.rl.traj_viz import path_divergence, plot_matched_paths

    trainer = make_trainer([], [])
    trainer.window()
    metrics, records = trainer.path_eval(max_steps=15)
    assert np.isfinite(metrics["behavior/path_divergence"])
    assert 0.0 <= metrics["behavior/vision_matched_success"] <= 1.0
    assert len(records) == 8  # n_matched_layouts default
    for rec in records:
        for name in ("vision", "audio"):
            assert rec["episodes"][name]["path"].shape[1] == 2
    # identical paths -> zero divergence; padding handles unequal lengths
    p1 = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert path_divergence(p1, p1) == 0.0
    assert path_divergence(p1, p1[:2]) > 0.0

    board = plot_matched_paths(records, trainer.sides["vision"].arena.cfg,
                               tmp_path / "board.png")
    assert board.exists() and board.stat().st_size > 1000


def test_squashed_policy_consistency_and_diagnostics():
    from src.rl.policy import squashed_logp_entropy, squashed_sample

    torch.manual_seed(0)
    mean = torch.tensor([[0.3, -4.0]])          # second dim: far-out mean
    log_std = torch.full_like(mean, -0.5)
    action, logp = squashed_sample(mean, log_std)
    assert action.abs().max() < 1.0             # squashed: strictly inside (-1,1)
    # replay logp of the SAME action reproduces the collection logp (ratio=1)
    logp2, ent = squashed_logp_entropy(mean, log_std, action)
    assert torch.allclose(logp, logp2, atol=1e-5)
    assert torch.isfinite(ent).all()

    trainer = make_trainer([], [])
    metrics = trainer.window()
    for name in ("vision", "audio"):
        assert np.isfinite(metrics[f"{name}/policy_mean_abs"])
        assert 0.0 <= metrics[f"{name}/action_sat_frac"] <= 1.0
        assert np.isfinite(metrics[f"{name}/log_std"])


def test_mean_reg_penalizes_saturation():
    """With advantage 0 and a perfect value target, the ONLY active loss term
    difference between a centered and a far-out policy mean is the saturation
    penalty — the far-out mean must cost more, and its gradient must point
    back toward zero."""
    from src.data.experiences import Experience
    from src.rl.tasks import make_ppo_task
    from src.rl.policy import squashed_logp_entropy

    torch.manual_seed(0)
    net = ActorCritic(in_channels=3, input_hw=(16, 16), widths=(8, 16),
                      pool_after=(1, 2), trunk_dim=32)
    probed = ProbedModel(net, layer_types=[nn.Conv2d], drop_last=False)
    task = make_ppo_task(mean_reg=1e-2)
    x = torch.randn(1, 3, 16, 16)

    def loss_at_bias(bias):
        params = {k: v.detach().clone() for k, v in probed.params().items()}
        params["actor_mean.bias"] = torch.tensor([bias, bias]).requires_grad_(True)
        mean, log_std, value = probed.forward_output(params, x)
        action = torch.tanh(mean).detach()          # on-policy-ish stored action
        logp, _ = squashed_logp_entropy(mean, log_std, action)
        exp = Experience(x=x, y={"action": action, "logp_old": logp.detach(),
                                 "advantage": torch.zeros(1),
                                 "value_target": value.detach()})
        loss = task(probed, params, exp)
        grad = torch.autograd.grad(loss, params["actor_mean.bias"])[0]
        return float(loss), grad

    loss_far, grad_far = loss_at_bias(4.0)
    loss_ctr, _ = loss_at_bias(0.0)
    assert loss_far > loss_ctr
    assert (grad_far > 0).all()   # pushes the +4.0 bias back down

    trainer = make_trainer([], [])
    metrics = trainer.window()
    assert np.isfinite(metrics["vision/exec_entropy_est"])


def test_time_windowed_episode_stats():
    """Episode stats are time-windowed and carry length/count context; the
    fast-success crowding of a last-N-episodes buffer is what this replaces."""
    trainer = make_trainer([], [])
    side = trainer.sides["vision"]
    # synthetic history: at env-step clock 100, a slow failure (len 200,
    # finished at step 40) and fast successes (len 25) within the window
    side.env_steps = 300
    side.finished = [(40, False, -2.0, 200)] + [
        (260 + i, True, 8.0, 25) for i in range(4)]
    stats = side.episode_stats(horizon_steps=100)
    assert stats["episodes_recent"] == 4.0          # the old failure aged out
    assert stats["success_rate"] == 1.0
    stats_all = side.episode_stats(horizon_steps=1000)
    assert stats_all["episodes_recent"] == 5.0      # includes the slow failure
    assert abs(stats_all["success_rate"] - 0.8) < 1e-9
    assert abs(stats_all["episode_len"] - (200 + 4 * 25) / 5) < 1e-9

    # trainer produces the keys once real episodes finish
    for _ in range(8):                              # horizon 30, window 5
        metrics = trainer.window()
    assert "vision/episodes_recent" in metrics
    assert 0.0 <= metrics["vision/success_rate"] <= 1.0
    assert metrics["vision/episode_len"] > 0


def test_flatten_preserves_spatial_information():
    """The trunk input must distinguish mirrored scenes (goal left vs right);
    GAP's per-channel spatial mean is nearly blind to the flip."""
    torch.manual_seed(0)
    net = ActorCritic(in_channels=3, input_hw=(16, 16), widths=(8, 16),
                      pool_after=(1, 2), trunk_dim=32, stats_bypass=False)
    x = torch.zeros(1, 3, 16, 16)
    x[:, :, 6:10, 2:5] = 1.0                       # bright blob on the LEFT
    x_flip = torch.flip(x, dims=[3])               # same blob on the RIGHT
    with torch.no_grad():
        f1, f2 = net.encoder(x), net.encoder(x_flip)
        # GAP equivalent: per-channel spatial means of the pre-flatten map
        hw = 4 * 4  # 16x16 after two pools
        g1 = f1.view(1, -1, hw).mean(-1)
        g2 = f2.view(1, -1, hw).mean(-1)
    diff_flat = float((f1 - f2).norm())
    diff_gap = float((g1 - g2).norm()) * hw ** 0.5   # scale-matched
    assert diff_flat > 3 * diff_gap                  # position survives flatten
    # trunk dim follows the modality's spatial size
    assert net.trunk[0].in_features == 16 * 4 * 4


def test_audio_past_is_frozen_across_overlapping_windows():
    """The same physical step re-renders with the IDENTICAL waveform chunk:
    window t+1's first W-1 chunks == window t's last W-1 chunks."""
    sensor = make_sensor()          # W = 4
    S = sensor.step_samples
    dists = np.array([3.0, 2.5, 2.0, 1.5, 1.0])
    betas = np.array([0.3, 0.2, 0.1, 0.0, -0.1])
    ids = np.arange(5, dtype=np.float64) + 42
    cues_t = np.column_stack([dists[:4], betas[:4], ids[:4]])
    cues_t1 = np.column_stack([dists[1:], betas[1:], ids[1:]])
    w_t = sensor.received_window(cues_t, phase=0)
    w_t1 = sensor.received_window(cues_t1, phase=S)   # window slid by one step
    assert np.allclose(w_t[:, S:], w_t1[:, :3 * S])   # shared past identical
    # and re-rendering the same window is bit-identical (no rng state)
    assert np.allclose(w_t, sensor.received_window(cues_t, phase=0))


def test_stats_bypass_preserves_amplitude_cue():
    """GN-normalized conv features are nearly invariant to a common additive
    offset (log-domain loudness); the raw-input stats bypass is not."""
    torch.manual_seed(0)
    net = ActorCritic(in_channels=2, input_hw=(16, 20), widths=(8, 16),
                      pool_after=(1, 2), trunk_dim=32, stats_bypass=True)
    x = torch.randn(1, 2, 16, 20)
    x_loud = x + 0.7                                # uniform log-domain gain shift
    with torch.no_grad():
        f, f_loud = net.encoder(x), net.encoder(x_loud)
        s_, s_loud = net._input_stats(x), net._input_stats(x_loud)
    # conv+GN attenuates the offset...
    conv_rel = float((f - f_loud).norm() / (f.norm() + 1e-8))
    # ...while the bypass carries it exactly (per-channel means shift by 0.7)
    c = x.shape[1]
    assert torch.allclose(s_loud[:, :c] - s_[:, :c], torch.full((1, c), 0.7), atol=1e-5)
    stats_rel = float((s_ - s_loud).norm() / (s_.norm() + 1e-8))
    assert stats_rel > 2 * conv_rel
    # trunk input dim includes the stats block
    assert net.trunk[0].in_features == 16 * 4 * 5 + 2 * (2 + 20)


def test_state_mask_is_a_world_property():
    """Same state -> identical mask, everywhere; different states -> different
    masks; within-bucket moves keep the mask; fraction ~ requested."""
    from src.rl.arena import state_mask_for

    cfg = {"enabled": True, "fraction": 0.5, "block": 4, "fill": 0.5,
           "quant_pos": 0.25, "quant_yaw_deg": 15.0, "seed": 9}
    pose = np.array([1.0, -0.5, 0.3])
    goal = np.array([-1.2, 2.0])
    m1 = state_mask_for(pose, goal, 32, cfg)
    m2 = state_mask_for(pose.copy(), goal.copy(), 32, cfg)
    assert np.array_equal(m1, m2)                       # deterministic
    within = state_mask_for(pose + [0.05, 0.05, 0.01], goal, 32, cfg)
    assert np.array_equal(m1, within)                   # same bucket
    across = state_mask_for(pose + [0.5, 0.0, 0.0], goal, 32, cfg)
    assert not np.array_equal(m1, across)               # new bucket, new mask
    other_goal = state_mask_for(pose, goal + [0.5, 0.0], 32, cfg)
    assert not np.array_equal(m1, other_goal)           # goal is part of state
    assert abs(m1.mean() - 0.5) < 0.2                   # roughly the fraction


def test_state_mask_flows_through_probes_and_rollout():
    """The choke point covers every vision render: probe-bank and rollout
    observations carry the fill value at masked positions, consistently."""
    torch.manual_seed(0)
    mask_cfg = {"enabled": True, "fraction": 0.6, "block": 4, "fill": 0.5, "seed": 3}
    sensor = make_sensor()
    sides = {}
    for name, in_ch, sens in (("vision", 3, None), ("audio", sensor.channels, sensor)):
        input_hw = (16, 16) if name == "vision" else (16, sensor.frames)
        net = ActorCritic(in_channels=in_ch, input_hw=input_hw,
                          widths=(8, 16), pool_after=(1, 2), trunk_dim=32)
        probed = ProbedModel(net, layer_types=[nn.Conv2d], drop_last=False)
        arena = FakeArena(ArenaConfig(num_envs=2, horizon=30, camera_hw=16,
                                      seed=0 if name == "vision" else 1,
                                      state_mask=mask_cfg if name == "vision" else None))
        sides[name] = RLSide(name, name, arena, probed, OPT, PPO, device="cpu",
                             sensor=sens, seed=0)
    probe_bank = build_probe_bank(sides, n=6, seed=3)
    vision_probes = probe_bank["probes"]["vision"]
    frac_at_fill = float((vision_probes == 0.5).float().mean())
    assert frac_at_fill > 0.3                           # masked blocks present
    # audio probes untouched by the vision mask machinery
    assert probe_bank["probes"]["audio"].shape[0] == 6
    # rollout obs from the masked arena also carry the fill
    obs = sides["vision"].observe()
    assert float((obs == 0.5).float().mean()) > 0.3
    # same state renders the same masked image twice
    a1 = sides["vision"].arena.render_at(np.array([0.5, 0.5, 0.1]), np.array([2.0, 2.0]))
    a2 = sides["vision"].arena.render_at(np.array([0.5, 0.5, 0.1]), np.array([2.0, 2.0]))
    assert np.array_equal(a1, a2)


def test_noise_model_keeps_level_cue_monotone():
    """The distance cue: mean log-mel level must fall MONOTONICALLY with d
    (the old sigma = base + slope*d had a minimum at d~2.5 and rose again —
    ambiguous level, untrainable), and the binaural L-R difference must stay
    well clear of draw noise at long range."""
    sensor = make_sensor()          # default AudioConfig noise: constant sigma
    assert sensor.cfg.noise_slope == 0.0
    levels = [float(np.mean([sensor.observe_stationary(
                  d, 0.0, noise_offset=1000 * i).mean() for i in range(4)]))
              for d in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)]
    assert all(a > b for a, b in zip(levels, levels[1:])), levels
    # ILD at range: separation between left- and right-source observations
    # dominates the draw-to-draw spread
    left = [float((o := sensor.observe_stationary(6.0, np.pi / 2,
                   noise_offset=50_000 + 1000 * i))[0].mean() - o[1].mean())
            for i in range(6)]
    right = [float((o := sensor.observe_stationary(6.0, -np.pi / 2,
                    noise_offset=90_000 + 1000 * i))[0].mean() - o[1].mean())
             for i in range(6)]
    sep = abs(np.mean(left) - np.mean(right))
    assert sep > 10 * (np.std(left) + 1e-9)


def test_cross_render_uses_full_teacher_history():
    """Finding-3 regression: a VISION teacher must hand the audio student full
    W-row moving cue histories with honest clip phases — not 1-row stationary
    stubs with phase ticking by 1 — and cross-rendering must be deterministic
    and distinct from a stationary render of the endpoint."""
    from src.rl.cross_render import render_state

    sides = make_sides()
    vision, audio = sides["vision"], sides["audio"]
    w = audio.sensor.cfg.window_steps
    assert vision.window_steps == w                      # shared cue geometry
    window = vision.collect_window(t_len=w + 2)
    assert window["cue_hist"].shape[2:] == (w, 3)        # full W rows, 3 cols
    # phases advance by real audio samples on the sensor-less side too
    deltas = np.diff(window["phase"][:, 0].numpy())
    assert (deltas == audio.sensor.step_samples).all()
    # per-side noise-id namespaces are disjoint
    assert vision.noise_id_base != audio.noise_id_base
    # pick a step late enough that the history is genuinely moving
    t, b = w, 0
    hist = window["cue_hist"][t, b].numpy()
    obs = render_state("audio", audio, window["pose"][t, b], window["goal"][t, b],
                       hist, window["phase"][t, b])
    again = render_state("audio", audio, window["pose"][t, b], window["goal"][t, b],
                         hist, window["phase"][t, b])
    assert torch.equal(obs, again)                       # deterministic re-render
    if len(np.unique(hist[:, 0])) > 1:                   # moving history...
        frozen = np.column_stack([np.full(w, hist[-1, 0]), np.full(w, hist[-1, 1]),
                                  hist[:, 2]])
        stationary = render_state("audio", audio, window["pose"][t, b],
                                  window["goal"][t, b], frozen, window["phase"][t, b])
        assert not torch.equal(obs, stationary)          # ...is not stationary

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
        hw = self.cfg.camera_hw
        rel = np.asarray(goal) - np.asarray(pose)[:2]
        img = np.zeros((3, hw, hw), dtype=np.float32)
        img[0] += np.tanh(rel[0])
        img[1] += np.tanh(rel[1])
        img[2] += 1.0 / (1.0 + np.linalg.norm(rel))
        img += 0.05 * np.sin(np.arange(hw))[None, None, :]  # spatial texture
        return img

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
    for name, in_ch, sens in (("vision", 3, None), ("audio", sensor.channels, sensor)):
        net = ActorCritic(in_channels=in_ch, widths=(8, 16), pool_after=(1, 2), trunk_dim=32)
        probed = ProbedModel(net, layer_types=[nn.Conv2d], drop_last=False)
        arena = FakeArena(ArenaConfig(num_envs=2, horizon=30, camera_hw=16,
                                      seed=0 if name == "vision" else 1))
        sides[name] = RLSide(name, name, arena, probed, OPT, PPO, device="cpu",
                             sensor=sens, seed=0)
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
    assert len(records) == 4
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

"""Vectorized MuJoCo arena: one agent, one audible/visible goal, four walls.

One compiled MjModel + B independent MjData. Motion is KINEMATIC: the action
[forward, turn] in [-1,1]^2 integrates the pose directly (egocentric, like the
sensory-hierarchy repo's point agent), positions are clamped to the arena, and
MuJoCo is used for rendering only (mj_forward, never mj_step). Rewards:
step penalty + potential-based distance shaping + bonus on reaching the goal.

All pose math lives in pure functions (kinematic_step, reward_fn) so tests and
the eval-bank builder run without a GL context; only render paths need one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class ArenaConfig:
    num_envs: int = 8
    horizon: int = 200
    half_extent: float = 3.0          # arena is [-h, h]^2
    agent_radius: float = 0.15
    reach_threshold: float = 0.35     # distance to goal that counts as success
    step_length: float = 0.15         # max metres per step at full forward
    max_turn_deg: float = 25.0
    min_spawn_separation: float = 2.0 # agent never spawns on top of the goal
    camera_hw: int = 64
    reward: dict = field(default_factory=lambda: {
        "step_penalty": -0.01, "shaping": 1.0, "reach_bonus": 10.0})
    seed: int = 0


# ---- pure kinematics / reward (no mujoco needed) ---------------------------


def kinematic_step(pose: np.ndarray, action: np.ndarray, cfg: ArenaConfig) -> np.ndarray:
    """pose (3,) = [x, y, yaw]; action (2,) = [forward, turn] in [-1, 1]."""
    forward = (np.clip(action[0], -1, 1) + 1.0) / 2.0          # -1 -> stop, +1 -> full
    yaw = pose[2] + np.clip(action[1], -1, 1) * math.radians(cfg.max_turn_deg)
    step = forward * cfg.step_length
    lim = cfg.half_extent - cfg.agent_radius
    x = np.clip(pose[0] + step * math.cos(yaw), -lim, lim)
    y = np.clip(pose[1] + step * math.sin(yaw), -lim, lim)
    return np.array([x, y, math.atan2(math.sin(yaw), math.cos(yaw))])


def bearing(pose: np.ndarray, goal: np.ndarray) -> float:
    """Source bearing in the agent frame: 0 = ahead, +pi/2 = left."""
    to_goal = np.arctan2(goal[1] - pose[1], goal[0] - pose[0])
    b = to_goal - pose[2]
    return float(math.atan2(math.sin(b), math.cos(b)))


def reward_fn(dist_before: float, dist_after: float, reached: bool, cfg: ArenaConfig) -> float:
    r = cfg.reward["step_penalty"] + cfg.reward["shaping"] * (dist_before - dist_after)
    if reached:
        r += cfg.reward["reach_bonus"]
    return float(r)


# ---- MJCF ------------------------------------------------------------------


def build_mjcf(cfg: ArenaConfig) -> str:
    h, wall_h, t = cfg.half_extent, 0.4, 0.05
    return f"""
<mujoco>
  <visual><headlight ambient=".45 .45 .45" diffuse=".8 .8 .8"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.45 0.65 0.9" rgb2="0.08 0.12 0.3"
             width="64" height="64"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.28 0.28 0.30"
             rgb2="0.45 0.45 0.48" width="128" height="128"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="10 10"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="{h} {h} 0.1" material="floor_mat"/>
    <geom name="wall_n" type="box" pos="0 {h} {wall_h/2}" size="{h} {t} {wall_h/2}" rgba="0.55 0.45 0.35 1"/>
    <geom name="wall_s" type="box" pos="0 {-h} {wall_h/2}" size="{h} {t} {wall_h/2}" rgba="0.55 0.45 0.35 1"/>
    <geom name="wall_e" type="box" pos="{h} 0 {wall_h/2}" size="{t} {h} {wall_h/2}" rgba="0.45 0.55 0.35 1"/>
    <geom name="wall_w" type="box" pos="{-h} 0 {wall_h/2}" size="{t} {h} {wall_h/2}" rgba="0.45 0.35 0.55 1"/>
    <body name="agent" pos="0 0 0.12">
      <joint name="agent_x" type="slide" axis="1 0 0" limited="false"/>
      <joint name="agent_y" type="slide" axis="0 1 0" limited="false"/>
      <joint name="agent_yaw" type="hinge" axis="0 0 1" limited="false"/>
      <geom name="agent_geom" type="cylinder" size="{cfg.agent_radius} 0.08" rgba="0.75 0.2 0.2 1"/>
      <camera name="agent_cam" pos="{cfg.agent_radius} 0 0.06" xyaxes="0 -1 0 0 0 1" fovy="90"/>
    </body>
    <body name="goal" mocap="true" pos="1 1 0.18">
      <geom name="goal_geom" type="sphere" size="0.16" rgba="0.1 0.95 0.15 1"
            contype="0" conaffinity="0"/>
      <light name="goal_light" pos="0 0 0.6" diffuse="0.3 0.9 0.3" specular="0 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""


# ---- vectorized env --------------------------------------------------------


class VecArena:
    """B independent copies of the world; kinematic stepping; lazy renderer."""

    def __init__(self, cfg: ArenaConfig):
        import mujoco

        self._mj = mujoco
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(cfg))
        self.datas = [mujoco.MjData(self.model) for _ in range(cfg.num_envs)]
        self._scratch = mujoco.MjData(self.model)          # for render_at / probes
        self._renderer = None
        self.rng = np.random.default_rng(cfg.seed)

        b = cfg.num_envs
        self.poses = np.zeros((b, 3))
        self.goals = np.zeros((b, 2))
        self.t = np.zeros(b, dtype=np.int64)
        for i in range(b):
            self.reset_env(i)

    # -- state ----------------------------------------------------------------

    def _sample_layout(self) -> tuple[np.ndarray, np.ndarray]:
        lim = self.cfg.half_extent - 3 * self.cfg.agent_radius
        while True:
            pose = np.array([*self.rng.uniform(-lim, lim, 2), self.rng.uniform(-np.pi, np.pi)])
            goal = self.rng.uniform(-lim, lim, 2)
            if np.linalg.norm(pose[:2] - goal) >= self.cfg.min_spawn_separation:
                return pose, goal

    def reset_env(self, i: int):
        self.poses[i], self.goals[i] = self._sample_layout()
        self.t[i] = 0

    def dists(self) -> np.ndarray:
        return np.linalg.norm(self.poses[:, :2] - self.goals, axis=1)

    def step(self, actions: np.ndarray) -> dict:
        """actions (B, 2) in [-1,1]. Returns rewards/done/reset flags; callers
        read poses/goals/dists before AND after to build observations."""
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
            truncated = self.t[i] >= self.cfg.horizon
            if reached or truncated:
                resets[i] = True
                self.reset_env(i)
        return {"reward": rewards, "done": dones, "reset": resets,
                "dist_before": before, "dist_after": after}

    # -- rendering (needs GL) --------------------------------------------------

    def _get_renderer(self):
        if self._renderer is None:
            self._renderer = self._mj.Renderer(self.model, self.cfg.camera_hw, self.cfg.camera_hw)
        return self._renderer

    def render_at(self, pose: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """RGB (3, H, W) float in [0,1] from the agent camera at an arbitrary
        state — used for rollouts, cross-rendering, and probe banks alike."""
        data = self._scratch
        data.qpos[:3] = pose
        data.mocap_pos[0][:2] = goal
        self._mj.mj_forward(self.model, data)
        renderer = self._get_renderer()
        renderer.update_scene(data, camera="agent_cam")
        rgb = renderer.render().astype(np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))

    def observe(self) -> torch.Tensor:
        """(B, 3, H, W) current camera observations."""
        frames = [self.render_at(self.poses[i], self.goals[i]) for i in range(self.cfg.num_envs)]
        return torch.from_numpy(np.stack(frames))

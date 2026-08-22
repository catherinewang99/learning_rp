"""Track 2 placeholder: environment-based guide/target (ACI integration).

Planned modules (see ROADMAP "Track 2" and the planning discussion):
  adapter.py    wrap ACI's MjCambrianMazeEnv (single point agent, single
                ~64x64 camera eye, no evolution) behind a thin gym-style
                interface; optional import so lrp never hard-depends on the
                cambrian/hydra-dev stack (separate conda env on manitoulin)
  captioner.py  swappable state->text interface; v0 = templates over
                privileged maze state (agent/goal xy, walls, BFS direction);
                VLM captioner later. THE "sensor of Theseus" knob.
  rollouts.py   collect guide trajectories; bank transitions bimodally
                (image view + caption view of each state); probe states via
                qpos teleport + render
  Also needed in rules/: a functional policy-gradient rule (the guide's
  trainable rule; ACI's SB3 PPO is stateful/batched and cannot define V).
"""

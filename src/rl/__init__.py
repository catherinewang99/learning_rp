"""RL track: two single-agent navigators (vision / audio) in a MuJoCo arena,
coupled ONLY through plasticity-kernel alignment (see ROADMAP "RL track").

Design decisions (user-approved):
  * Separate independent envs per agent — behavioral convergence is an
    OUTCOME to measure, never built in. The alignment step cross-renders the
    teacher's window of transitions into the student's modality, so Π rows
    correspond (same world transitions) while trajectories stay independent.
  * Motion is kinematic (pose integration + wall clamping); MuJoCo provides
    rendering and the world description, not dynamics. Keeps the env
    deterministic and free of actuator tuning.
  * Audio = speech clip at the goal, gain falling and noise rising with
    distance; one observation per step = BINAURAL (2-ch) log-mel spectrogram
    whose time axis spans the last `window_steps` env steps. Head-shadow ILD
    + front bias give instantaneous direction (audio.binaural: false = the
    mono loudness-trend-only ablation).
  * PPO minimal: 1 epoch, the whole 10-step x B-env window as one batch,
    AdamW + clipping via the same honesty machinery as the other tracks —
    V(e) is the exact AdamW step on transition e's PPO loss.
"""

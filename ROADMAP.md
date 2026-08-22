# Roadmap

## Track 0 — audio-vision joint co-training (ACTIVE)
CIFAR-100 subset <-> UrbanSound8K subset (6 paired classes per
arXiv:2601.22041), VGG-11-GN twins from scratch, audio guided by vision,
all 8 convs aligned 1:1. Arms A (control) / B (K-CKA) / C (Π); D dropped.
- [x] vgg11 (canonical structure, norm=group|batch|none), paired_av data
      (fixed seeded class-level pairing, shuffled_pairs control, US8K folds),
      prepare_us8k.py (download + log-mel cache), configs/av/, AV smoke tests
- [ ] Run prepare_us8k.py on manitoulin; sanity-check per-class counts
- [ ] Shakeout run (arm A, few hundred steps): task curves move, memory OK
- [ ] Launch A/B/C (same seed & data order); watch eval/k_cka vs arm A
- [ ] Analysis: K-CKA(t) overlays, layer x time heatmaps, val acc A vs B vs C
- [ ] Decisions to revisit: lr 0.05, steps 20k, audio augmentation (none in
      v0; paper used audiomentations), ESC-50 classes for cleaner semantic
      pairs, shuffled_pairs control run

## Track 1 — imagenet-captions joint co-training (built, parked)
- [x] Multi-layer probing by module type (Conv2d / LayerNorm), execution order,
      crossmodal-prior pooling + even-spread layer mapping (+ upper_half)
- [x] Paired ImageNet-Captions data (caption_index verbatim; manitoulin paths)
- [x] JointTrainer: pure SGD both sides, direction config (guided sides),
      detached live teacher, m_per_step cost knob, minibatch experiences
- [x] Tracked metrics on fixed banks: cross-model K-CKA / Π-CKA per layer,
      vision top-1, LM ppl, V magnitudes
- [x] Four arms: control / K-only / Π-only / K+Π (same seed & data order)
- [ ] Launch on manitoulin (one GPU per arm), monitor eval/k_cka curves
      (parked in favor of Track 0's smaller/faster AV version)
- [ ] Val transform: use Resize/CenterCrop for val loader (v0 uses train tf)
- [ ] Analysis notebook: K-CKA(t) overlays, layer x time heatmaps
- [ ] Decisions to revisit after first runs: lr (0.05?), probe_size, m_per_step,
      whether magnitude/v_match terms enter any arm, upper_half

## Track 2 — offline prior distillation (was milestones 1–4)
- [x] Identity sanity + meta-gradient flow (now layerwise)
- [ ] ResNet guide -> MLP target at θ0 (real data)
- [ ] Trajectory matching via checkpoint bank; persistence/decay measurement
- [ ] Downstream eval: plasticity-aligned init vs K-aligned init vs random
- OPEN (pinned): where does the guide start — pretrained vs scratch?

## Track 3 — environment version (ACI / cambrian)
- [ ] envs/adapter.py: MjCambrianMazeEnv, single point agent, single ~64x64 eye,
      no evolution; optional import, separate conda env (hydra-dev stack)
- [ ] envs/captioner.py: template captioner over privileged maze state
      (the "sensor of Theseus" knob); VLM captioner later
- [ ] envs/rollouts.py: bimodal transition banking; probe states via qpos
      teleport + render (deterministic)
- [ ] rules/policy_grad.py: functional per-transition policy-gradient rule
      (guide trains with the SAME rule that defines its V — not SB3 PPO)
- [ ] Offline "following": bank guide transitions -> align LM -> LM navigates
      via captions; eval vs random/K-only/BC-only controls

## Phase A — learning-rule distillation (later)
- [ ] Rule ladder: L1 per-layer LRs (contains SGD => known-solution test),
      L2 gated/preconditioned, L3 blackbox local (bio-plausible rules slot in)
- [ ] trainable=rule in the offline trainer (machinery already in place)

## Deliberately punted
- vmap/chunked Π (loop is fine; hooks preclude vmap — chunk if needed)
- DiffKNN term on Π (metrics.py notes where it goes)
- Multi-GPU joint training (single GPU per arm is the design point)

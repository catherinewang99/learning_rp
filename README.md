# lrp — Learning-Rule Prior / Plasticity Kernel Alignment

Align not just *what* a network represents (match K), but *how its
representations change with experience* (match the plasticity kernel Π),
between arbitrary guide and target networks — including across modalities.

## Core objects (src/kernels/ — the "edit the math here" zone)

For a network with parameters θ, a fixed probe set X, and a learning rule R,
at every probed layer l:

- **Representation kernel**  `K_l = H_l H_lᵀ`, H_l = pooled layer activations on X.
- **Representational response**  `V_l(X|e) = (K_l(θ + Δ_R θ(e)) − K_l(θ)) / η` —
  how one hypothetical update on experience `e` changes the probe geometry.
- **Plasticity kernel**  `Π_l(eᵢ, eⱼ) = ⟨vec V_l(X|eᵢ), vec V_l(X|eⱼ)⟩` — a kernel
  over experiences: a representation of the learning dynamics themselves.
- **Alignment losses** (src/losses/): CKA on Π, CKA on K, per-experience V̂
  matching, ΔK-magnitude preservation — composable weighted terms, applied
  layerwise via an even-spread guide→target layer mapping.

Everything is functional (`torch.func`): losses differentiate THROUGH the
hypothetical update (second order). Nothing stateful, nothing in-place.

## Experiment tracks, one measurement stack

1. **Audio-vision joint co-training (ACTIVE).** VGG-11-GN twins (canonical
   8-conv structure, GroupNorm — see gotchas) classify a 6-class CIFAR-100
   subset (3ch, 32x32) and its class-paired UrbanSound8K subset (1ch log-mel
   spectrograms), both from scratch with AdamW (decoupled weight decay,
   global-norm grad clipping; the hypothetical step defining V is the exact
   AdamW update via rules/adamw.py); audio is guided by a live,
   detached vision teacher, all 8 conv layers aligned strictly 1:1. Pairing
   recipe follows arXiv:2601.22041 (class-level pairing; fixed seeded
   assignment; `shuffled_pairs` control). Arms: A control / B K-CKA
   (training-the-untrainable analog) / C Π (the test). Headline readout:
   does Π-guidance raise cross-modal K-CKA above Arm A's drift, vs Arm B?
   Setup: `python scripts/prepare_us8k.py --root data --download` once, then
   `python scripts/av_train.py --config configs/av/arm_c_pi_only.yaml`.
2. **ImageNet-captions joint co-training (built, parked).** ResNet-18 +
   GPT-2 from scratch on paired ImageNet-Captions (configs/joint/arm_*.yaml,
   scripts/joint_train.py; manitoulin paths are the defaults).
3. **Offline prior distillation.** Bank a frozen guide's per-checkpoint
   summaries; optimize a target init (trainable=weights) or later a rule
   (trainable=rule) against the bank. `scripts/align.py`.
4. **Environment track (planned — src/envs/).** ACI/cambrian maze navigation:
   vision guide navigates, template/VLM captioner describes states, LM target
   follows. See src/envs/__init__.py for the planned seams.

## Layout

```
configs/            YAML (plain; `_base_:` deep-merge only). joint/ = the 4 arms
src/
  models/probed.py    ProbedModel: any nn.Module + layer types/names -> {layer: H}
  models/zoo.py       registry (mlp, tiny_cnn, resnet18/34, gpt2) + probe defaults
  rules/              LearningRule (functional Δθ); SGDRule; tasks.py (CE, next-token)
  kernels/            gram/response/plasticity/metrics — pure functions
  losses/             terms.py (per-pair) + layerwise.py (mapping) + composite.py
  data/               paired_av.py (CIFAR100<->US8K, class pairing), paired.py
                      (imagenet-captions), caption_index.py (verbatim from
                      crossmodal-prior), experiences.py, probes.py (fixed banks)
  guide/              offline-track guide bank
  training/           joint_trainer.py (sides are named modalities with views),
                      trainer.py (offline), metrics.py (tracked eval)
  envs/               env-track placeholder
scripts/            av_train.py, prepare_us8k.py, joint_train.py, align.py
tests/              19 tests: kernel math, identity sanity, multilayer, joint + AV smoke
```

## Development

Local: `./.venv/bin/python -m pytest tests/ -q` (CPU torch; tiny models).
Real runs: manitoulin GPUs; ImageNet + captions paths follow crossmodal-prior
(`$IMAGENET_ROOT`, `$IMAGENET_CAPTIONS_JSON` to override). wandb: NO default
project (crossmodal convention) — every tracked run must pass
`--wandb-project <name>` explicitly.

Gotchas already encoded in the code: GPT-2 must use eager attention
(flash/SDPA has no double-backward — zoo.py forces it); probed models are
kept in eval mode (BN/dropout must not pollute kernels); teacher summaries
use detached params so no graph/gradient ever reaches the teacher; VGG twins
use GroupNorm, not BatchNorm — BN's batch statistics make kernels
probe-batch-dependent in train mode and freeze useless at init in eval mode,
breaking the "V describes the update the model takes" invariant either way;
AdamW is implemented as a functional rule that reads the live optimizer
moments, with sqrt floors so second-order backward through the update never
NaNs on zero-gradient params.

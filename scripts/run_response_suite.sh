#!/usr/bin/env bash
# Sequential on one GPU. Override GPU/SEEDS/WINDOWS/TAG/PROJECT as needed.
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-3}
SEEDS=${SEEDS:-0}
WINDOWS=${WINDOWS:-10000}
TAG=${TAG:-pilot}
PROJECT=${PROJECT:-rl_audiovis}
RUNS=${RUNS:-"r0_independent r4_sgd_trunk_pi r5_sgd_trunk_v"}
for seed in $SEEDS; do
  for run in $RUNS; do
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$GPU" python scripts/rl_train.py \
      --config "configs/rl/response/${run}.yaml" --seed "$seed" \
      --windows "$WINDOWS" --run-tag "$TAG" --wandb-project "$PROJECT"
  done
done

#!/usr/bin/env bash
set -euo pipefail

ENV_BATCH_SIZE="${ENV_BATCH_SIZE:-768}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000}"
SEED="${SEED:-42}"

python -m ppga.RL.train_mjlab_reference \
  --num_envs "$ENV_BATCH_SIZE" \
  --max_iterations "$MAX_ITERATIONS" \
  --seed "$SEED" \
  --fixed_goal \
  --disable_curriculum \
  --run_name stationary_reference \
  --log_root experiments/mjlab_reference_stationary

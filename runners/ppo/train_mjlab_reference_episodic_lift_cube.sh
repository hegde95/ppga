#!/usr/bin/env bash
set -euo pipefail

ENV_BATCH_SIZE="${ENV_BATCH_SIZE:-768}"
MAX_ITERATIONS="${MAX_ITERATIONS:-700}"
SEED="${SEED:-42}"

python -m ppga.RL.train_mjlab_reference \
  --num_envs "$ENV_BATCH_SIZE" \
  --max_iterations "$MAX_ITERATIONS" \
  --seed "$SEED" \
  --no-fixed_goal \
  --disable_curriculum \
  --command_resampling_time 40.0 \
  --run_name episodic_dynamic_reference \
  --log_root experiments/mjlab_reference_episodic

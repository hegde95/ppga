#!/usr/bin/env bash
set -euo pipefail

ENV_BATCH_SIZE="${ENV_BATCH_SIZE:-768}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000}"
SEED="${SEED:-42}"

python -m mjlab.scripts.train Mjlab-Lift-Cube-Yam \
  --env.scene.num-envs "$ENV_BATCH_SIZE" \
  --env.seed "$SEED" \
  --agent.seed "$SEED" \
  --agent.max-iterations "$MAX_ITERATIONS" \
  --agent.logger tensorboard \
  --agent.upload-model False \
  --agent.run-name "reference_seed_${SEED}" \
  --log-root experiments/mjlab_reference

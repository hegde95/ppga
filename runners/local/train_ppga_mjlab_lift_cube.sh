#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="lift_cube"
SEED=42
RUN_NAME="ppga_mjlab_${ENV_NAME}_seed_${SEED}"

python -m ppga.algorithm.train_ppga_isaac \
  --env_type=mjlab \
  --env_name="$ENV_NAME" \
  --num_dims=2 \
  --grid_size=50 \
  --seed="$SEED" \
  --rollout_length=32 \
  --env_batch_size=3072 \
  --popsize=256 \
  --anneal_lr=False \
  --num_minibatches=8 \
  --update_epochs=5 \
  --learning_rate=0.0005 \
  --vf_coef=1.0 \
  --entropy_coef=0.0 \
  --target_kl=0.01 \
  --max_grad_norm=1.0 \
  --action_transform=tanh \
  --value_bootstrap=True \
  --eval_deterministic=True \
  --normalize_obs=False \
  --normalize_returns=False \
  --total_iterations=2000 \
  --dqd_algorithm=cma_maega \
  --sigma0=0.5 \
  --restart_rule=no_improvement \
  --calc_gradient_iters=10 \
  --move_mean_iters=10 \
  --archive_lr=0.1 \
  --threshold_min=0 \
  --use_wandb=True \
  --wandb_project=ppga \
  --wandb_group=mjlab_lift_cube \
  --wandb_run_name="$RUN_NAME" \
  --expdir="./experiments/ppga_mjlab_${ENV_NAME}"

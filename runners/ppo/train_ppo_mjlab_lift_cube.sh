#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="lift_cube"
SEED="${SEED:-42}"
ENV_BATCH_SIZE="${ENV_BATCH_SIZE:-768}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-15000000}"
USE_WANDB="${USE_WANDB:-False}"
RUN_NAME="ppo_mjlab_episodic_${ENV_NAME}_seed_${SEED}"

python -m ppga.RL.train_ppo \
  --env_type=mjlab \
  --env_name="$ENV_NAME" \
  --mjlab_fixed_goal=False \
  --mjlab_disable_curriculum=True \
  --mjlab_command_resampling_time=40.0 \
  --mjlab_descriptor_mode=height_approach \
  --num_dims=2 \
  --seed="$SEED" \
  --rollout_length=24 \
  --env_batch_size="$ENV_BATCH_SIZE" \
  --anneal_lr=False \
  --num_minibatches=4 \
  --update_epochs=5 \
  --norm_adv_per_minibatch=False \
  --mixed_precision=False \
  --learning_rate=0.001 \
  --vf_coef=1.0 \
  --entropy_coef=0.005 \
  --target_kl=0.01 \
  --adaptive_kl=True \
  --max_grad_norm=1.0 \
  --action_transform=none \
  --action_std_parameterization=direct \
  --initial_action_std=0.5 \
  --actor_hidden_dims 512 256 128 \
  --value_bootstrap=True \
  --eval_deterministic=True \
  --normalize_obs=True \
  --normalize_returns=False \
  --checkpoint_interval_updates=100 \
  --max_checkpoints=3 \
  --total_timesteps="$TOTAL_TIMESTEPS" \
  --use_wandb="$USE_WANDB" \
  --wandb_project=ppga \
  --wandb_group=mjlab_lift_cube_ppo_baseline \
  --wandb_run_name="$RUN_NAME" \
  --expdir="./experiments/ppo_mjlab_episodic_${ENV_NAME}"

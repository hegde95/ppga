#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="lift_cube"
SEED="${SEED:-42}"
ENV_BATCH_SIZE="${ENV_BATCH_SIZE:-768}"
POPSIZE="${POPSIZE:-32}"
TOTAL_ITERATIONS="${TOTAL_ITERATIONS:-1000}"
SIGMA0="${SIGMA0:-0.05}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
CALC_GRADIENT_ITERS="${CALC_GRADIENT_ITERS:-10}"
MOVE_MEAN_ITERS="${MOVE_MEAN_ITERS:-1}"
USE_WANDB="${USE_WANDB:-False}"
INITIAL_ACTOR_CHECKPOINT="${INITIAL_ACTOR_CHECKPOINT:-}"
INITIAL_ACTOR_ARGS=()
if [[ -n "$INITIAL_ACTOR_CHECKPOINT" ]]; then
  INITIAL_ACTOR_ARGS+=(--initial_actor_checkpoint="$INITIAL_ACTOR_CHECKPOINT")
fi
RUN_NAME="ppga_mjlab_episodic_${ENV_NAME}_seed_${SEED}"

python -m ppga.algorithm.train_ppga_isaac \
  --env_type=mjlab \
  --env_name="$ENV_NAME" \
  "${INITIAL_ACTOR_ARGS[@]}" \
  --mjlab_fixed_goal=False \
  --mjlab_disable_curriculum=True \
  --mjlab_command_resampling_time=40.0 \
  --mjlab_terminate_on_success=True \
  --mjlab_success_bonus=50.0 \
  --mjlab_success_max_object_speed=0.15 \
  --mjlab_descriptor_mode=grip_orientation_arm_length \
  --mjlab_arm_length_min=0.20 \
  --mjlab_arm_length_max=0.50 \
  --mjlab_transport_start_distance=0.03 \
  --mjlab_approach_deviation_reference=0.05 \
  --mjlab_transport_deviation_reference=0.15 \
  --num_dims=2 \
  --grid_size=12 \
  --seed="$SEED" \
  --rollout_length=24 \
  --env_batch_size="$ENV_BATCH_SIZE" \
  --popsize="$POPSIZE" \
  --anneal_lr=False \
  --num_minibatches=4 \
  --update_epochs=5 \
  --norm_adv_per_minibatch=False \
  --mixed_precision=False \
  --learning_rate="$LEARNING_RATE" \
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
  --eval_common_random_numbers=True \
  --normalize_obs=True \
  --normalize_returns=False \
  --total_iterations="$TOTAL_ITERATIONS" \
  --dqd_algorithm=cma_maega \
  --sigma0="$SIGMA0" \
  --xnes_center_init=zero \
  --restart_rule=no_improvement \
  --calc_gradient_iters="$CALC_GRADIENT_ITERS" \
  --move_mean_iters="$MOVE_MEAN_ITERS" \
  --archive_lr=0.1 \
  --threshold_min=0 \
  --archive_min_success_rate=0.75 \
  --log_arch_freq="${LOG_ARCH_FREQ:-10}" \
  --save_scheduler=False \
  --save_heatmaps=True \
  --heatmap_freq=10 \
  --use_wandb="$USE_WANDB" \
  --wandb_project=ppga \
  --wandb_group=mjlab_lift_cube \
  --wandb_run_name="$RUN_NAME" \
  --expdir="./experiments/ppga_mjlab_episodic_${ENV_NAME}"

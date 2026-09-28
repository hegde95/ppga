#!/usr/bin/env bash
# Run PPO reference training as a detached overnight job.

set -euo pipefail

SEED="${SEED:-42}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100000}"
MAX_TRAINING_HOURS="${MAX_TRAINING_HOURS:-7.5}"
ENV_BATCH_SIZE="${ENV_BATCH_SIZE:-512}"
PYTHON_BIN="${PYTHON_BIN:-/home/eisuke/miniconda3/envs/mjlab_ppga/bin/python}"

# Recommended parallel envs based on VRAM:
# 4096 MB: 384, 8192 MB: 512, 16384 MB: 768, 24576 MB: 1024, 32768 MB: 1536

EXPDIR="${EXPDIR:-./experiments/mjlab_reference_stationary}"
RUN_DIR="$EXPDIR/$SEED"
mkdir -p "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
LOG_FILE="$RUN_DIR/training.log"
PID_FILE="$RUN_DIR/training.pid"
UNIT_NAME="ppga-mjlab-ppo-${SEED}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter is not executable: $PYTHON_BIN" >&2
  exit 1
fi

if systemctl --user is-active --quiet "$UNIT_NAME.service"; then
  echo "Training is already running in $UNIT_NAME.service" >&2
  exit 1
fi

"$PYTHON_BIN" -c \
  'import colorlog, mjlab, torch; assert torch.cuda.is_available(), "CUDA is unavailable"'

echo "Starting MJLab PPO reference training: ${MAX_ITERATIONS} iterations, ${ENV_BATCH_SIZE} envs"
echo "Training time limit: ${MAX_TRAINING_HOURS} hours (final evaluation runs afterward)"

# A transient user service survives this terminal and the Codex launch session.
# The high iteration ceiling lets the wall-clock budget control the run; fixed
# learning rate avoids annealing against a ceiling the run will not reach.
systemd-run --user --collect --unit="$UNIT_NAME" \
  --description="MJLab PPO lift_cube seed $SEED" \
  --working-directory="$(pwd)" \
  --property="StandardOutput=append:$LOG_FILE" \
  --property="StandardError=append:$LOG_FILE" \
  -- "$PYTHON_BIN" -u -m ppga.RL.train_ppo \
  --env_name=lift_cube \
  --env_type=mjlab \
  --seed="$SEED" \
  --env_batch_size="$ENV_BATCH_SIZE" \
  --rollout_length=24 \
  --total_timesteps="$((ENV_BATCH_SIZE * 24 * MAX_ITERATIONS))" \
  --max_training_hours="$MAX_TRAINING_HOURS" \
  --anneal_lr=False \
  --num_minibatches=4 \
  --update_epochs=5 \
  --learning_rate=0.0001 \
  --entropy_coef=0.005 \
  --target_kl=0.01 \
  --adaptive_kl=True \
  --norm_adv_per_minibatch=False \
  --mixed_precision=False \
  --normalize_obs=True \
  --action_transform=none \
  --action_std_parameterization=direct \
  --initial_action_std=0.5 \
  --actor_hidden_dims 512 256 128 \
  --num_dims=2 \
  --value_bootstrap=True \
  --mjlab_fixed_goal=False \
  --mjlab_disable_curriculum=True \
  --mjlab_command_resampling_time=40 \
  --mjlab_terminate_on_success=True \
  --mjlab_success_bonus=50 \
  --mjlab_success_max_object_speed=0.15 \
  --mjlab_descriptor_mode=grip_orientation_elbow_extension \
  --checkpoint_interval_updates=100 \
  --max_checkpoints=3 \
  --expdir="$(dirname "$RUN_DIR")"

TRAIN_PID="$(systemctl --user show "$UNIT_NAME.service" --property=MainPID --value)"
printf '%s\n' "$TRAIN_PID" > "$PID_FILE"

echo "Training PID: $TRAIN_PID"
echo "Service: $UNIT_NAME.service"
echo "Log: $LOG_FILE"
echo "Checkpoints: $RUN_DIR/checkpoints"

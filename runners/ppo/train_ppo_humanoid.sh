#!/usr/bin/env bash

ENV_NAME="humanoid"
GRID_SIZE=50  # number of cells per archive dimension
SEED=1111


RUN_NAME="ppo_"$ENV_NAME"_seed_"$SEED
echo $RUN_NAME
python -m ppga.RL.train_ppo \
  --env_name=$ENV_NAME \
  --env_type=jax \
  --rollout_length=32 \
  --use_wandb=True \
  --wandb_group=btjanaka \
  --wandb_project=ppga \
  --num_dims=2 \
  --seed=$SEED \
  --anneal_lr=False \
  --num_minibatches=4 \
  --update_epochs=5 \
  --normalize_obs=False \
  --normalize_returns=False \
  --wandb_run_name=$RUN_NAME\
  --env_batch_size=3000 \
  --learning_rate=0.0005 \
  --vf_coef=1.0 \
  --entropy_coef=0.0 \
  --target_kl=0.01 \
  --max_grad_norm=1 \
  --total_timesteps=100000000 \
  --clip_obs_rew=False
  # --expdir=./experiments/paper_ppga_"$ENV_NAME"
  # --popsize=300 \
  # --total_iterations=2000 \
  # --dqd_algorithm=cma_maega \
  # --sigma0=1.0 \
  # --restart_rule=no_improvement \
  # --calc_gradient_iters=10 \
  # --move_mean_iters=10 \
  # --archive_lr=0.5 \
  # --threshold_min=200 \
  # --grid_size=$GRID_SIZE \

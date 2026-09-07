#!/usr/bin/env bash
#SBATCH --account=biyik_1173
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gpus-per-task=a40:1
#SBATCH --output=tmp/ppga_humanoid_%j.log

ENV_NAME="humanoid"
GRID_SIZE=50  # number of cells per archive dimension
SEED=43

RUN_NAME="paper_ppga_"$ENV_NAME"_seed_"$SEED
echo $RUN_NAME

module purge
eval "$(conda shell.bash hook)"
conda activate /home1/eh_352/ppga/env
module load apptainer
apptainer exec --env "ACCEPT_EULA=Y" --nv /home1/eh_352/isaac-lab_2.2.0_nano5.sif bash

python -m ppga.algorithm.train_ppga_isaac \
  --env_name=$ENV_NAME \
  --rollout_length=32 \
  --use_wandb=True \
  --wandb_group=paper \
  --num_dims=2 \
  --seed=$SEED \
  --anneal_lr=False \
  --num_minibatches=4 \
  --update_epochs=5 \
  --normalize_obs=False \
  --normalize_returns=False \
  --adaptive_stddev=True \
  --clip_obs_rew=True \
  --action_transform=tanh \
  --eval_deterministic=True \
  --value_bootstrap=True \
  --wandb_run_name=$RUN_NAME \
  --popsize=300 \
  --env_batch_size=6000 \
  --learning_rate=0.0005 \
  --vf_coef=1.0 \
  --entropy_coef=0.0 \
  --target_kl=0.01 \
  --max_grad_norm=1 \
  --total_iterations=5 \
  --dqd_algorithm=cma_maega \
  --sigma0=0.5 \
  --restart_rule=no_improvement \
  --calc_gradient_iters=10 \
  --move_mean_iters=10 \
  --archive_lr=0.1 \
  --threshold_min=0 \
  --grid_size=$GRID_SIZE \
  --expdir=./experiments/paper_ppga_"$ENV_NAME"

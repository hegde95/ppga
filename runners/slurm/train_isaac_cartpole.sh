#!/usr/bin/env bash
#SBATCH --account=biyik_1173
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=tmp/isaac_cartpole_%j.log

module purge
eval "$(conda shell.bash hook)"
conda activate /home1/eh_352/ppga/env
module load apptainer
apptainer exec --env "ACCEPT_EULA=Y" --nv /home1/eh_352/isaac-lab_2.2.0_nano5.sif bash

cd /home1/eh_352/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-v0 --headless
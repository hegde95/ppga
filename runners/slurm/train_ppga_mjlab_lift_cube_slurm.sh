#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gpus-per-task=1
#SBATCH --output=tmp/ppga_mjlab_lift_cube_%j.log

set -euo pipefail
module purge
eval "$(conda shell.bash hook)"
conda activate ppga-mjlab

srun bash runners/local/train_ppga_mjlab_lift_cube.sh

#!/bin/bash
#SBATCH --job-name=isaacsim-test
#SBATCH --output=isaacsim-test.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1

module purge
module load apptainer
apptainer exec --env "ACCEPT_EULA=Y" --nv isaac-lab_2.2.0_nano5.sif bash  

python /scr/eisukehirota/ppga/ppga/envs/isaac_lab/test_import.py
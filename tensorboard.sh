#!/bin/bash
#SBATCH --job-name="tensorboard"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00

module purge
module load conda/22.11.1
conda activate patchnetvlad

echo "Starting ${logdir} on port ${port}."

tensorboard --logdir=$logdir --port=$port

#!/bin/bash
# Launch pytorch distributed in a software environment or container
#
# (c) 2022, Eric Stubbs
# University of Florida Research Computing

#SBATCH --wait-all-nodes=1
#SBATCH --job-name=
#SBATCH --mail-type=NONE
#SBATCH --mail-user=
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:8
#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:30:00
#SBATCH --partition=gpu
#SBATCH --constraint=
#SBATCH --output=PatchNetvlad_%j.out
#SBATCH --error=PatchNetvlad_%j.err


# PYTHON SCRIPT
#==============

#This is the python script to run in the pytorch environment
LOSS="triplet"
METHOD="official"

# LOAD PYTORCH SOFTWARE ENVIRONMENT
#==================================

## You can load a software environment or use a singularity container.
## CONTAINER="singularity exec --nv /path/to/container.sif" (--nv option is to enable gpu)
module purge
module load conda/22.11.1
conda activate patchnetvlad


# PRINTS
#=======
date; pwd; which python
export HOST=$(hostname -s)
NODES=$(scontrol show hostnames | grep -v $HOST | tr '\n' ' ')
echo "Host: $HOST" 
echo "Other nodes: $NODES"

#LR=0.001
DATE=$(date '+%d-%b') 
FILES="/home/m.maqboolbhutta/models/patchnetvlad/${METHOD}-${LOSS}-${DATE}"
echo ${FILES}

# LAUNCH
#=======

echo "Starting $SLURM_GPUS_PER_TASK process(es) on each node..."
bash train.sh ${FILES}

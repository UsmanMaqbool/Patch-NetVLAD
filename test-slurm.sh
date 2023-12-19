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
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --constraint=
#SBATCH --output=logs/PatchNetvlad_%j.out
#SBATCH --error=logs/PatchNetvlad_%j.err


# PYTHON SCRIPT
#==============

#This is the python script to run in the pytorch environment

DATASET=$1
FILES=$2
config_path=patchnetvlad/configs/train.ini
--config_path=patchnetvlad/configs/train.ini \
--resume_path=$RESUME \
--dataset_root_dir=/home/leo/usman_ws/datasets/2015netVLAD/Pittsburgh250k \
--dataset_choice=pitts

if [ $# -ne 2 ]
  then
    echo "Arguments error: <DATASET (mapillary|Pitts30k > <PATH of checkpoints>"
    exit 1
fi

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

# LAUNCH
#=======

echo "Starting $SLURM_GPUS_PER_TASK process(es) on each node..."
bash test_model.sh ${DATASET} ${FILES}


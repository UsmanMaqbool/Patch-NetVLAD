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
#SBATCH --time=48:30:00
#SBATCH --partition=gpu
#SBATCH --constraint=
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err


allowed_arguments_list1=("netvlad" "graphvlad")
allowed_arguments_list2=("triplet" "sare_ind" "sare_joint")

if [ "$#" -ne 2 ]; then
    echo "Arguments error: <METHOD (netvlad|graphvlad>"
    echo "Arguments error: <LOSS_TYPE (triplet|sare_ind|sare_joint)>"
    exit 1
fi

METHOD="$1"
LOSS="$2"


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


# PYTHON SCRIPT
#==============
echo "Starting $SLURM_GPUS_PER_TASK process(es) on each node..."
bash train.sh ${METHOD} ${LOSS} Slurm

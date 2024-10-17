#!/bin/bash
# Launch pytorch distributed in a software environment or container
#
# (c) 2022, Eric Stubbs
# University of Florida Research Computing

#SBATCH --wait-all-nodes=1
#SBATCH --job-name=
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=m.maqboolbhutta@ufl.edu
#SBATCH --time=100:00:00
#SBATCH --partition=gpu
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=a100:4  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --constraint=a100
#SBATCH --mem-per-cpu=8GB
#SBATCH --distribution=cyclic:cyclic

## To RUN
# sbatch --j 1015-s1-sare-ind-graphvlad train-slurm.sh graphvlad vgg16 sare_ind /home/m.maqboolbhutta/usman_ws/models/patchnetvlad/1004-s1/vgg16-graphvlad-sare_ind-04-Oct-lr-0_01/Oct04_10-36-12_mapillary_nopanos/checkpoints/checkpoint_epoch6.pth.tar

####################################################################################################



allowed_arguments_list1=("netvlad" "graphvlad")
allowed_arguments_list2=("triplet" "sare_ind" "sare_joint")

if [ "$#" -lt 2 ]; then
    echo "Arguments error: <METHOD (netvlad|graphvlad>"
    echo "Arguments error: <LOSS_TYPE (triplet|sare_ind|sare_joint)>"
    exit 1
fi

METHOD="$1"
ARCH="$2"
LOSS="$3"
RESUMEPATH="$4"

# LOAD PYTORCH SOFTWARE ENVIRONMENT
#==================================

## You can load a software environment or use a singularity container.
## CONTAINER="singularity exec --nv /path/to/container.sif" (--nv option is to enable gpu)
module purge
module load conda/24.3.0
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
bash train-s.sh ${METHOD} ${ARCH} ${LOSS} ${RESUMEPATH}

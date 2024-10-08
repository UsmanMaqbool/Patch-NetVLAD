#!/bin/bash
# Launch pytorch distributed in a software environment or container
#
# (c) 2022, Eric Stubbs
# University of Florida Research Computing

#SBATCH --wait-all-nodes=1
#SBATCH --job-name=
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=m.maqboolbhutta@ufl.edu
#SBATCH --time=52:00:00
#SBATCH --partition=gpu
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=a100:2   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --constraint=a100
#SBATCH --mem-per-cpu=4gb



# PYTHON SCRIPT
#==============

#This is the python script to run in the pytorch environment

METHOD=$1
DATASET=$2
FILES=$3

if [ $# -ne 3 ]
  then
    echo "Arguments error:<METHOD (netvlad/graphvlad)>  <DATASET (mapillary/Pitts30k > <PATH of checkpoints>"
    exit 1
fi

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

# LAUNCH
#=======

echo "Starting $SLURM_GPUS_PER_TASK process(es) on each node..."
bash test-s.sh ${METHOD} ${DATASET} ${FILES}

##TO RUN
#./test.sh mapillary /home/leo/usman_ws/models/patchnetvlad/hipergator-Jan26_21-46-27_mapillary_nopanos/Jan26_21-46-27_mapillary_nopanos/checkpoints/ | tee 26Jan2146.txt


# sbatch --j Feb06_14-07-48-test test-slurm.sh mapillary /home/m.maqboolbhutta/models/patchnetvlad/netvlad-triplet-06-Feb/Feb06_14-07-48_mapillary_nopanos/checkpoints
#!/bin/bash
#SBATCH --job-name="tensorboard"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

module purge
module load conda/22.11.1
conda activate patchnetvlad

echo "Starting ${logdir} on port ${port}."


tensorboard --logdir=/home/m.maqboolbhutta/models/patchnetvlad/

# SSH using port forward and view from local pc using
# http://localhost:7000/
# ssh -L 7000:localhost:6006 m.maqboolbhutta@hpg.rc.ufl.edu
module purge
module load conda/22.11.1
conda activate patchnetvlad
tensorboard --logdir=/home/m.maqboolbhutta/models/patchnetvlad/
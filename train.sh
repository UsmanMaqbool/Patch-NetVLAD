#!/bin/sh
PYTHON=${PYTHON:-"python3"}

#GPUS=1
DATE=$(date '+%d-%b') 



if [ $# -ne 1 ]; then
    # echo "Arguments error: <LOSS_TYPE (triplet|sare_ind|sare_joint)>"
    # exit 1
    FILES="/home/leo/usman_ws/models/patchnetvlad/official-triplet-${DATE}"
    DATASET_DIR="/home/leo/usman_ws/datasets/mapillary_sls/"
    CASHE_PATH="/home/leo/usman_ws/models/patchnetvlad/cache"
    CONFIG="patchnetvlad/configs/train.ini"

else
    FILES=$1
    DATASET_DIR="/home/m.maqboolbhutta/usman_ws/datasets/mapillary_sls/"
    CASHE_PATH=${SLURM_TMPDIR}
    CONFIG="patchnetvlad/configs/train-slurm.ini"
fi

echo "==========Starting Training============="
echo "Saving checkpoints at ${FILES}"
echo "========================================"

$PYTHON -u train.py --save_every_epoch \
  --config_path=patchnetvlad/configs/train.ini \
  --cache_path=${CASHE_PATH} \
  --save_path=$FILES \
  --dataset_root_dir=${DATASET_DIR} 


#  --cluster_path=/home/m.maqboolbhutta/usman_ws/datasets/netvlad-official/vgg16_pitts_64_desc_cen.hdf5 \
#  --threads=2 \
#  --loss-type ${LOSS} 

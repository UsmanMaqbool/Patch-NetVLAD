#!/bin/sh
PYTHON=${PYTHON:-"python3"}

#GPUS=1
# DATE=$(date '+%d-%b') 
LOSS=$1


if [ $# -ne 2 ]; then
    # echo "Arguments error: <LOSS_TYPE (triplet|sare_ind|sare_joint)>"
    # exit 1
    FILES="/home/leo/usman_ws/models/patchnetvlad/loss-openibl"
    DATASET_DIR="/home/leo/usman_ws/datasets/mapillary_sls/"
    CASHE_PATH="/home/leo/usman_ws/models/patchnetvlad/cache"
    CONFIG="patchnetvlad/configs/train.ini"
    CLUSTER_PATH="/home/leo/usman_ws/datasets/2015netVLAD/official/vgg16_pitts_64_desc_cen.hdf5"
    OFFTHESHELF_PATH="/home/leo/usman_ws/datasets/2015netVLAD/official/vd16_offtheshelf_conv5_3_max.pth"

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
  --dataset_root_dir=${DATASET_DIR} \
  --loss=${LOSS} \
  --cluster_path=${CLUSTER_PATH} \
  --vd16_offtheshelf_path=${OFFTHESHELF_PATH}


  # --cluster_path=${CLUSTER_PATH} \
#  =/home/m.maqboolbhutta/usman_ws/datasets/netvlad-official/vgg16_pitts_64_desc_cen.hdf5 \
#  --threads=2 \
#  --loss-type ${LOSS} 

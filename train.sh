#!/bin/sh
PYTHON=${PYTHON:-"python3"}

DATE=$(date '+%d-%b') 

if [ "$#" -ne 2 ]; then
    echo "Arguments error: <METHOD (netvlad|graphvlad>"
    echo "Arguments error: <LOSS_TYPE (triplet|sare_ind|sare_joint)>"
    echo "./train.sh netvlad triplet"    
    exit 1
fi

if [ "$#" -ne 3 ]; then
    METHOD="$1"
    LOSS="$2"
    FILES="/home/leo/usman_ws/models/patchnetvlad/${METHOD}-${LOSS}-${DATE}"
    DATASET_DIR="/home/leo/usman_ws/datasets/mapillary_sls/"
    CASHE_PATH="/home/leo/usman_ws/models/patchnetvlad/cache"
    CONFIG="patchnetvlad/configs/train.ini"
    CLUSTER_PATH="/home/leo/usman_ws/datasets/2015netVLAD/official/vgg16_pitts_64_desc_cen.hdf5"
    OFFTHESHELF_PATH="/home/leo/usman_ws/datasets/2015netVLAD/official/vd16_offtheshelf_conv5_3_max.pth"
else
    FILES="/home/m.maqboolbhutta/models/patchnetvlad/${METHOD}-${LOSS}-${DATE}"
    DATASET_DIR="/home/m.maqboolbhutta/usman_ws/datasets/mapillary_sls/"
    CASHE_PATH=${SLURM_TMPDIR}
    CONFIG="patchnetvlad/configs/train-slurm.ini"
    CLUSTER_PATH="/home/m.maqboolbhutta/usman_ws/datasets/netvlad-official/vgg16_pitts_64_desc_cen.hdf5"
    OFFTHESHELF_PATH="/home/m.maqboolbhutta/usman_ws/datasets/netvlad-official/vd16_offtheshelf_conv5_3_max.pth"
fi

echo "==========Starting Training============="
echo "Saving checkpoints at ${FILES}"
echo "========================================"

$PYTHON -u train.py --save_every_epoch \
  --config_path=${CONFIG} \
  --cache_path=${CASHE_PATH} \
  --save_path=$FILES \
  --dataset_root_dir=${DATASET_DIR} \
  --loss=${LOSS} \
  --cluster_path=${CLUSTER_PATH} \
  --vd16_offtheshelf_path=${OFFTHESHELF_PATH}

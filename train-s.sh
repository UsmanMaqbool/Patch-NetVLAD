#!/bin/sh
PYTHON=${PYTHON:-"python3"}

DATE=$(date '+%d-%b') 
METHOD="$1"
ARCH="$2"
LOSS="$3"
RESUMEPATH="$4"

if [ "$#" -lt 3 ]; then
    echo "Arguments error: <METHOD (netvlad|graphvlad)>"
    echo "Arguments error: <ARCH (vgg16 | resnet)>"
    echo "Arguments error: <LOSS_TYPE (triplet|sare_ind|sare_joint)>"
    echo "./train.sh graphvlad vgg16 triplet"    
    exit 1
fi

FILES="/home/m.maqboolbhutta/usman_ws/models/patchnetvlad/fastscnn-v3/${ARCH}-${METHOD}-${LOSS}-${DATE}"
DATASET_DIR="/home/m.maqboolbhutta/usman_ws/datasets/Mapillary_Street_Level_Sequences/"
CASHE_PATH="/home/m.maqboolbhutta/usman_ws/datasets/official/patchnetvlad/"
CONFIG="patchnetvlad/configs/train-slurm.ini"
FAST_SCNN="/home/m.maqboolbhutta/usman_ws/datasets/official/fast-scnn/fast_scnn_citys.pth"
CLUSTER_PATH="/home/m.maqboolbhutta/usman_ws/datasets/official/patchnetvlad/centroids/vgg16_mapillary_16_desc_cen.hdf5"
OFFTHESHELF_PATH="/home/m.maqboolbhutta/usman_ws/datasets/official/openibl-init/vd16_offtheshelf_conv5_3_max.pth"

# Check if the checkpoint path argument is provided
if [ -z "$RESUMEPATH" ]; then
    # If not provided, set a default value (empty string)
    CHECKPOINT_PATH=''
else
    # If provided, use the provided value
    CHECKPOINT_PATH=$RESUMEPATH
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
  --method=${METHOD} \
  --fast-scnn=${FAST_SCNN} \
  --cluster_path=${CLUSTER_PATH} \
  --threads=6 \
  --vd16_offtheshelf_path=${OFFTHESHELF_PATH} \
  --resume_path=${CHECKPOINT_PATH}

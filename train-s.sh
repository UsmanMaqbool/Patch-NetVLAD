#!/bin/sh
PYTHON=${PYTHON:-"python3"}

DATE=$(date '+%d-%b') 
METHOD="$1"
LOSS="$2"
RESUMEPATH="$3"

if [ "$#" -lt 2 ]; then
    echo "Arguments error: <METHOD (netvlad|graphvlad>"
    echo "Arguments error: <LOSS_TYPE (triplet|sare_ind|sare_joint)>"
    echo "./train.sh netvlad triplet"    
    exit 1
fi

FILES="/home/m.maqboolbhutta/usman_ws/models/patchnetvlad/${METHOD}-${LOSS}-${DATE}"
DATASET_DIR="/home/m.maqboolbhutta/usman_ws/datasets/mapillary_sls/"
CASHE_PATH="/home/m.maqboolbhutta/usman_ws/datasets/netvlad-official/"
CONFIG="patchnetvlad/configs/train-slurm.ini"
ESP_ENCODER="/home/m.maqboolbhutta/usman_ws/datasets/netvlad-official/espnet-encoder/espnet_p_2_q_8.pth"
CLUSTER_PATH="/home/m.maqboolbhutta/usman_ws/datasets/netvlad-official/centroids/vgg16_mapillary_16_desc_cen.hdf5"
OFFTHESHELF_PATH="/home/m.maqboolbhutta/usman_ws/datasets/netvlad-official/vd16_offtheshelf_conv5_3_max.pth"

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
  --esp_encoder=${ESP_ENCODER} \
  --cluster_path=${CLUSTER_PATH} \
  --threads=6 \
  --vd16_offtheshelf_path=${OFFTHESHELF_PATH} \
  --resume_path=${CHECKPOINT_PATH}

#!/bin/sh
PYTHON=${PYTHON:-"python3"}

DATE=$(date '+%d-%b') 
METHOD="$1"
LOSS="$2"

if [ "$#" -lt 2 ]; then
    echo "Arguments error: <METHOD (netvlad|graphvlad>"
    echo "Arguments error: <LOSS_TYPE (triplet|sare_ind|sare_joint)>"
    echo "./train.sh netvlad triplet"    
    exit 1
fi

if [ "$#" -ne 3 ]; then
    FILES="/home/leo/usman_ws/models/patchnetvlad/${METHOD}-${LOSS}-${DATE}-cen-map-16-lr0_001"
    DATASET_DIR="/home/leo/usman_ws/datasets/mapillary_sls/"
    CASHE_PATH="/home/leo/usman_ws/datasets/2015netVLAD/official/"
    CONFIG="patchnetvlad/configs/train.ini"
    CLUSTER_PATH="/home/leo/usman_ws/datasets/2015netVLAD/official/centroids/vgg16_mapillary_16_desc_cen.hdf5"
    ESP_ENCODER="/home/leo/usman_ws/datasets/espnet-encoder/espnet_p_2_q_8.pth"
    OFFTHESHELF_PATH="/home/leo/usman_ws/datasets/2015netVLAD/official/vd16_offtheshelf_conv5_3_max.pth"
else
    FILES="/home/m.maqboolbhutta/usman_ws/models/patchnetvlad/${METHOD}-${LOSS}-${DATE}-cen-map-16-lr0_001"
    DATASET_DIR="/home/m.maqboolbhutta/usman_ws/datasets/mapillary_sls/"
    CASHE_PATH="/home/m.maqboolbhutta/usman_ws/datasets/netvlad-official/"
    CONFIG="patchnetvlad/configs/train-slurm.ini"
    ESP_ENCODER="/home/m.maqboolbhutta/usman_ws/datasets/netvlad-official/espnet-encoder/espnet_p_2_q_8.pth"
    CLUSTER_PATH="/home/m.maqboolbhutta/usman_ws/datasets/netvlad-official/centroids/vgg16_mapillary_16_desc_cen.hdf5"
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
  --cluster_path=${CLUSTER_PATH} 


#   --resume_path=/home/leo/usman_ws/models/patchnetvlad/graphvlad-triplet-20-Feb-netvlad-triplet-cen-map-16-lr0_001/Feb20_09-18-06_mapillary_nopanos/checkpoints/checkpoint_epoch2.pth.tar
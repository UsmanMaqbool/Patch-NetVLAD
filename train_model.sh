#!/bin/sh
PYTHON=${PYTHON:-"python3"}

# python train.py \
# --config_path patchnetvlad/configs/train.ini \
# --cache_path=/media/leo/2C737A9872F69ECF/why-so-deepv2-data/mapillary-patch-netvlad/cache \
# --save_path=/media/leo/2C737A9872F69ECF/why-so-deepv2-data/mapillary-patch-netvlad/save_path \
# --dataset_root_dir=/media/leo/2C737A9872F69ECF/datasets/Mapillary_Street_Level_Sequences/

# ## Resume
# python train.py \
# --config_path patchnetvlad/configs/train.ini \
# --cache_path=/media/leo/2C737A9872F69ECF/why-so-deepv2-data/mapillary-patch-netvlad/cache \
# --save_path=/media/leo/2C737A9872F69ECF/why-so-deepv2-data/mapillary-patch-netvlad/save_path \
# --dataset_root_dir=/media/leo/2C737A9872F69ECF/datasets/Mapillary_Street_Level_Sequences/ \
# --resume_path=/media/leo/2C737A9872F69ECF/why-so-deepv2-data/mapillary-patch-netvlad/save_path/Sep02_09-39-50_mapillary_nopanos/


PYTHON=${PYTHON:-"python"}
GPUS=1
DATE=$(date '+%d-%b') 
METHOD=$1
ARCH=vgg16
LOSS=$2
LR=0.001

FILES="/home/leo/usman_ws/models/patchnetvlad/${METHOD}-${ARCH}/${LOSS}-${DATE}"

echo ${FILES}

if [ $# -ne 2 ]
  then
    echo "Arguments error: <LOSS_TYPE (triplet|sare_ind|sare_joint)>"
    exit 1
fi

echo "==========Starting Training============="
echo "========================================"

$PYTHON train.py \
--config_path=patchnetvlad/configs/train.ini \
--cache_path=/home/leo/usman_ws/models/patchnetvlad/cache \
--save_path=$FILES \
--dataset_root_dir=/home/leo/usman_ws/datasets/mapillary_sls/ \
--threads=1 \
--cluster_path=/home/leo/usman_ws/datasets/2015netVLAD/official/vgg16_pitts_64_desc_cen.hdf5 \
--loss-type ${LOSS} 


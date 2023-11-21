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
LR=0.0001

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
--threads=2 \
--cluster_path=/home/leo/usman_ws/datasets/2015netVLAD/official/vgg16_pitts_64_desc_cen.hdf5 \
--loss-type ${LOSS} 
# $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
# examples/netvlad_img.py --launcher pytorch --tcp-port ${PORT} \
#   -d ${DATASET} --scale ${SCALE} \
#   -a ${ARCH} --layers ${LAYERS} --vlad --syncbn --sync-gather \
#   --width 640 --height 480 --tuple-size 1 -j 1 --neg-num 10 --test-batch-size 32 \
#   --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} \
#   --eval-step 1 --epochs 5 --step-size 5 --cache-size 1000 \
#   --logs-dir ${FILES}





# BASEDir="/home/leo/usman_ws/models/graphvlad/pytorch-netvlad/pittsburgh-sare_joint-lr0.0001-14-Aug-1601/checkpoints/"
# FILES="${BASEDir}*.tar"
# echo "${FILES}"

# for RESUME in $FILES
# do
#   echo "Building PCA Model of $RESUME file..."
#   # take action on each file. $f store current file name
  
#   python add_pca.py \
#   --config_path patchnetvlad/configs/train-pitts.ini \
#   --resume_path=$RESUME \
#   --dataset_root_dir=/home/leo/usman_ws/datasets/2015netVLAD/ \
#   --dataset_choice=pitts
  
#   filename=$(basename $RESUME .pth.tar)
#   PCA_RESUME="${BASEDir}${filename}"
#   PCA_RESUME="${PCA_RESUME}_WPCA"
#   echo $PCA_RESUME
  
#   echo "==================================="
#   echo "============Pittsburgh Testing====="
#   echo "==================================="
  
  
#   echo "Extracting Features of Index Images"
#   # PCA_RESUME="${BASEDir}{$filename}_WPCA"

#   python feature_extract.py \
#   --config_path patchnetvlad/configs/performance.ini \
#   --dataset_file_path=pitts30k_imageNames_index.txt \
#   --dataset_root_dir=/home/leo/usman_ws/datasets/2015netVLAD/ \
#   --output_features_dir=/home/leo/usman_ws/models/patch-netvlad/pitts30k_index \
#   --resume_path=${PCA_RESUME}

  
#   echo "Extracting Features of Query Images"

#   python feature_extract.py \
#   --config_path patchnetvlad/configs/performance.ini \
#   --dataset_file_path=pitts30k_imageNames_query.txt \
#   --dataset_root_dir=/home/leo/usman_ws/datasets/2015netVLAD/ \
#   --output_features_dir=/home/leo/usman_ws/models/patch-netvlad/pitts30k_query \
#   --resume_path=${PCA_RESUME}
#   echo "Performing Features Matching and Recall Result"

#   python feature_match.py \
#   --config_path patchnetvlad/configs/performance.ini \
#   --dataset_root_dir==/media/leo/2C737A9872F69ECF/datasets/maqbool-datasets/datasets-place-recognition/Test_Pitts250k/ \
#   --query_file_path=pitts30k_imageNames_query.txt \
#   --index_file_path=pitts30k_imageNames_index.txt \
#   --query_input_features_dir /home/leo/usman_ws/models/patch-netvlad/pitts30k_query \
#   --index_input_features_dir /home/leo/usman_ws/models/patch-netvlad/pitts30k_index \
#   --result_save_folder=./patchnetvlad/results/pitts30k-pytorchnetvlad \
#   --ground_truth_path=./patchnetvlad/dataset_gt_files/pitts30k_test.npz

#   echo "==================================="
#   echo "============Toyko 247 Testing======"
#   echo "==================================="

#   echo "Extracting Features of Query Images"
#   python feature_extract.py \
#   --config_path patchnetvlad/configs/performance.ini \
#   --dataset_file_path=tokyo247_imageNames_query-V3.txt \
#   --dataset_root_dir /home/leo/usman_ws/datasets/2015netVLAD/ \
#   --output_features_dir=/home/leo/usman_ws/models/patch-netvlad/tokyo247_query \
#   --resume_path=${PCA_RESUME}

#   echo "Extracting Features of Index Images"

#   python feature_extract.py \
#   --config_path patchnetvlad/configs/performance.ini \
#   --dataset_file_path=tokyo247_imageNames_index.txt \
#   --dataset_root_dir /home/leo/usman_ws/datasets/2015netVLAD/ \
#   --output_features_dir=/home/leo/usman_ws/models/patch-netvlad/tokyo247_index \
#   --resume_path=${PCA_RESUME}

#   echo "Performing Features Matching and Recall Result"  

#   python feature_match.py \
#   --config_path patchnetvlad/configs/performance.ini \
#   --dataset_root_dir /home/leo/usman_ws/datasets/2015netVLAD/ \
#   --query_file_path=tokyo247_imageNames_query-V3.txt \
#   --index_file_path=tokyo247_imageNames_index.txt \
#   --query_input_features_dir /home/leo/usman_ws/models/patch-netvlad/tokyo247_query \
#   --index_input_features_dir /home/leo/usman_ws/models/patch-netvlad/tokyo247_index \
#   --ground_truth_path patchnetvlad/dataset_gt_files/tokyo247.npz \
#   --result_save_folder patchnetvlad/results/tokyo247


#done



#!/bin/sh
PYTHON=${PYTHON:-"python3"}
METHOD=$1
DATASET=$2
BASEDir=$3
FILES=$(find "${BASEDir}" -maxdepth 1 -type f -name "*.tar" ! -name '*WPCA4096*')

echo "${FILES}"

## Dataset path [Change according to yours]
if [ "$DATASET" = "mapillary" ]; then
  dataset_root_dir=/home/leo/usman_ws/datasets/Mapillary_Street_Level_Sequences/
elif [ "$DATASET" = "pitts" ]; then
  dataset_root_dir=/home/leo/usman_ws/datasets/2015netVLAD/Pittsburgh250k/
else
    echo "Invalid dataset choice"
    exit 1
fi
FAST_SCNN="/home/leo/usman_ws/datasets/official/fast-scnn/fast_scnn_citys.pth"


for RESUME in $FILES
do

  echo "==========################============="
  echo " Testing $RESUME file on Pitts30k ..."
  echo "======================================="
  
  # Extracting the directory and filename without extension
  dir=$(dirname "$RESUME")
  # take action on each file. $f store current file name
  filename=$(basename $RESUME .pth.tar)
  # Generating the new filename
  # echo "filename: $filename"
  PCA_RESUME="${dir}/${filename}"
  PCA_RESUME="${PCA_RESUME}_WPCA4096.pth.tar"
  # echo "PCA_Resume Path: $PCA_RESUME"
  filename=$(basename "$RESUME" .pth.tar)
  # Extracting the date and identifier from the path
  date_and_identifier=$(basename "$(dirname "$dir")")
  # echo "$filename"
  # Generating the new filename
  SAVEFILENAME="${date_and_identifier}-${filename}"

  if [ -f "$PCA_RESUME" ]; then
      echo ""Skipping the addition of the PCA layer as $PCA_RESUME already exists.""
  else
      echo "Building PCA Model of $RESUME file..."
      echo "Adding PCA Layer"
      python add_pca.py \
      --config_path=patchnetvlad/configs/train.ini \
      --resume_path=$RESUME \
      --dataset_root_dir=$dataset_root_dir \
      --dataset_choice=$DATASET \
      --method=${METHOD} \
      --fast-scnn=${FAST_SCNN}
  fi

  
  echo "Extracting Features of pitts30k Index Images"

  python feature_extract.py \
  --config_path patchnetvlad/configs/test-fast.ini \
  --dataset_file_path=pitts30k_imageNames_index.txt \
  --dataset_root_dir=/home/leo/usman_ws/datasets/2015netVLAD/Pittsburgh250k/ \
  --output_features_dir=/home/leo/usman_ws/models/patchnetvlad/features/pitts30k_index \
  --resume_path=${PCA_RESUME} \
  --method=${METHOD} \
  --fast-scnn=${FAST_SCNN} \

  
  echo "Extracting Features of pitts30k Query Images"

  python feature_extract.py \
  --config_path patchnetvlad/configs/test-fast.ini \
  --dataset_file_path=pitts30k_imageNames_query.txt \
  --dataset_root_dir=/home/leo/usman_ws/datasets/2015netVLAD/Pittsburgh250k/ \
  --output_features_dir=/home/leo/usman_ws/models/patchnetvlad/features/pitts30k_query \
  --resume_path=${PCA_RESUME} \
  --method=${METHOD} \
  --fast-scnn=${FAST_SCNN} \
  
  echo "Performing Features Matching and Recall Result of pitts30k"
  python feature_match_graphvlad.py \
  --config_path patchnetvlad/configs/test-fast.ini \
  --dataset_root_dir=/home/leo/usman_ws/datasets/2015netVLAD/Pittsburgh250k/ \
  --query_file_path=pitts30k_imageNames_query.txt \
  --index_file_path=pitts30k_imageNames_index.txt \
  --query_input_features_dir /home/leo/usman_ws/models/patchnetvlad/features/pitts30k_query \
  --index_input_features_dir /home/leo/usman_ws/models/patchnetvlad/features/pitts30k_index \
  --ground_truth_path patchnetvlad/dataset_gt_files/pitts30k_test.npz \
  --method=${METHOD} \
  --result_save_folder=./patchnetvlad/results/pitts30k

  echo "==========################============="
  echo " Testing $RESUME file on Tokyo24/7 ..."
  echo "======================================="
  echo "Extracting Features of tokyo247 Index Images"

  python feature_extract.py \
  --config_path patchnetvlad/configs/test-fast.ini \
  --dataset_file_path=tokyo247_imageNames_index.txt \
  --dataset_root_dir=/home/leo/usman_ws/datasets/2015netVLAD/tokyo247/ \
  --output_features_dir=/home/leo/usman_ws/models/patchnetvlad/features/tokyo247_index \
  --resume_path=${PCA_RESUME} \
  --method=${METHOD} \
  --fast-scnn=${FAST_SCNN} \

  
  echo "Extracting Features of tokyo247 Query Images"

  python feature_extract.py \
  --config_path patchnetvlad/configs/test-fast.ini \
  --dataset_file_path=tokyo247_imageNames_query.txt \
  --dataset_root_dir=/home/leo/usman_ws/datasets/2015netVLAD/tokyo247/ \
  --output_features_dir=/home/leo/usman_ws/models/patchnetvlad/features/tokyo247_query \
  --resume_path=${PCA_RESUME} \
  --method=${METHOD} \
  --fast-scnn=${FAST_SCNN} \
  
  echo "Performing Features Matching and Recall Result of tokyo247"
  python feature_match_graphvlad.py \
  --config_path patchnetvlad/configs/test-fast.ini \
  --dataset_root_dir=/home/leo/usman_ws/datasets/2015netVLAD/tokyo247/ \
  --query_file_path=tokyo247_imageNames_query.txt \
  --index_file_path=tokyo247_imageNames_index.txt \
  --query_input_features_dir /home/leo/usman_ws/models/patchnetvlad/features/tokyo247_query \
  --index_input_features_dir /home/leo/usman_ws/models/patchnetvlad/features/tokyo247_index \
  --ground_truth_path patchnetvlad/dataset_gt_files/tokyo247.npz \
  --method=${METHOD} \
  --result_save_folder=./patchnetvlad/results/tokyo247
 

done



#!/bin/sh
PYTHON=${PYTHON:-"python3"}
DATASET=$1
BASEDir=$2
FILES=$(find "${BASEDir}" -maxdepth 1 -type f -name "*.tar")
# echo "${FILES}"

## Dataset path [Change according to yours]
if [ "$dataset" = "mapillary" ]; then
  dataset_root_dir=/home/leo/usman_ws/datasets/mapillary_sls/
elif [ "$dataset" = "pitts" ]; then
  dataset_root_dir=/home/leo/usman_ws/datasets/2015netVLAD/Pittsburgh250k/
else
    echo "Invalid dataset choice"
    exit 1
fi


for RESUME in $FILES
do
  echo "Building PCA Model of $RESUME file..."
  # take action on each file. $f store current file name

  filename=$(basename $RESUME .pth.tar)
  PCA_RESUME="${BASEDir}${filename}"
  PCA_RESUME="${PCA_RESUME}_WPCA"
  
  if [ -f "$FILE" ]; then
      echo ""Skipping the addition of the PCA layer as $PCA_RESUME already exists.""
  else
      echo "Adding PCA Layer"
      python add_pca.py \
      --config_path=patchnetvlad/configs/train.ini \
      --resume_path=$RESUME \
      --dataset_root_dir=$dataset_root_dir \
      --dataset_choice=$dataset
  fi



  
  
  # echo "Extracting Features of Index Images"
  # # PCA_RESUME="${BASEDir}{$filename}_WPCA"

  # python feature_extract.py \
  # --config_path patchnetvlad/configs/performance.ini \
  # --dataset_file_path=mapillarysf_imageNames_index.txt \
  # --dataset_root_dir=--dataset_root_dir=/home/leo/usman_ws/datasets/mapillary_sls/ \
  # --output_features_dir=/home/leo/usman_ws/models/patch-netvlad/mapillarysf_index \
  # --resume_path=${PCA_RESUME}

  
  # echo "Extracting Features of Query Images"

  # python feature_extract.py \
  # --config_path patchnetvlad/configs/performance.ini \
  # --dataset_file_path=mapillarysf_imageNames_query.txt \
  # --dataset_root_dir=--dataset_root_dir=/home/leo/usman_ws/datasets/mapillary_sls/ \
  # --output_features_dir=/home/leo/usman_ws/models/patch-netvlad/mapillarysf_query \
  # --resume_path=${PCA_RESUME}
  # echo "Performing Features Matching and Recall Result"

  # python feature_match.py \
  # --config_path patchnetvlad/configs/performance.ini \
  # --dataset_root_dir==/media/leo/2C737A9872F69ECF/datasets/maqbool-datasets/datasets-place-recognition/Test_Pitts250k/ \
  # --query_file_path=mapillarysf_imageNames_query.txt \
  # --index_file_path=mapillarysf_imageNames_index.txt \
  # --query_input_features_dir /home/leo/usman_ws/models/patch-netvlad/mapillarysf_query \
  # --index_input_features_dir /home/leo/usman_ws/models/patch-netvlad/mapillarysf_index \
  # --result_save_folder=./patchnetvlad/results/mapillarysf-pytorchnetvlad \
  # --ground_truth_path=./patchnetvlad/dataset_gt_files/mapillarysf_test.npz



  # echo "==================================="
  # echo "============Pittsburgh Testing====="
  # echo "==================================="
  
  
  # echo "Extracting Features of Index Images"
  # # PCA_RESUME="${BASEDir}{$filename}_WPCA"

  # python feature_extract.py \
  # --config_path patchnetvlad/configs/performance.ini \
  # --dataset_file_path=pitts30k_imageNames_index.txt \
  # --dataset_root_dir=/home/leo/usman_ws/datasets/2015netVLAD/Pittsburgh250k \
  # --output_features_dir=/home/leo/usman_ws/models/patch-netvlad/pitts30k_index \
  # --resume_path=${PCA_RESUME}

  
  # echo "Extracting Features of Query Images"

  # python feature_extract.py \
  # --config_path patchnetvlad/configs/performance.ini \
  # --dataset_file_path=pitts30k_imageNames_query.txt \
  # --dataset_root_dir=/home/leo/usman_ws/datasets/2015netVLAD/Pittsburgh250k \
  # --output_features_dir=/home/leo/usman_ws/models/patch-netvlad/pitts30k_query \
  # --resume_path=${PCA_RESUME}
  # echo "Performing Features Matching and Recall Result"

  # python feature_match.py \
  # --config_path patchnetvlad/configs/performance.ini \
  # --dataset_root_dir==/media/leo/2C737A9872F69ECF/datasets/maqbool-datasets/datasets-place-recognition/Test_Pitts250k/ \
  # --query_file_path=pitts30k_imageNames_query.txt \
  # --index_file_path=pitts30k_imageNames_index.txt \
  # --query_input_features_dir /home/leo/usman_ws/models/patch-netvlad/pitts30k_query \
  # --index_input_features_dir /home/leo/usman_ws/models/patch-netvlad/pitts30k_index \
  # --result_save_folder=./patchnetvlad/results/pitts30k-pytorchnetvlad \
  # --ground_truth_path=./patchnetvlad/dataset_gt_files/pitts30k_test.npz

  # echo "==================================="
  # echo "============Toyko 247 Testing======"
  # echo "==================================="

  # echo "Extracting Features of Query Images"
  # python feature_extract.py \
  # --config_path patchnetvlad/configs/performance.ini \
  # --dataset_file_path=tokyo247_imageNames_query-V3.txt \
  # --dataset_root_dir /home/leo/usman_ws/datasets/2015netVLAD/ \
  # --output_features_dir=/home/leo/usman_ws/models/patch-netvlad/tokyo247_query \
  # --resume_path=${PCA_RESUME}

  # echo "Extracting Features of Index Images"

  # python feature_extract.py \
  # --config_path patchnetvlad/configs/performance.ini \
  # --dataset_file_path=tokyo247_imageNames_index.txt \
  # --dataset_root_dir /home/leo/usman_ws/datasets/2015netVLAD/ \
  # --output_features_dir=/home/leo/usman_ws/models/patch-netvlad/tokyo247_index \
  # --resume_path=${PCA_RESUME}

  # echo "Performing Features Matching and Recall Result"  

  # python feature_match.py \
  # --config_path patchnetvlad/configs/performance.ini \
  # --dataset_root_dir /home/leo/usman_ws/datasets/2015netVLAD/ \
  # --query_file_path=tokyo247_imageNames_query-V3.txt \
  # --index_file_path=tokyo247_imageNames_index.txt \
  # --query_input_features_dir /home/leo/usman_ws/models/patch-netvlad/tokyo247_query \
  # --index_input_features_dir /home/leo/usman_ws/models/patch-netvlad/tokyo247_index \
  # --ground_truth_path patchnetvlad/dataset_gt_files/tokyo247.npz \
  # --result_save_folder patchnetvlad/results/tokyo247


done



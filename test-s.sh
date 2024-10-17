#!/bin/sh
PYTHON=${PYTHON:-"python3"}
METHOD=$1
DATASET=$2
BASEDir=$3
FILES=$(find "${BASEDir}" -maxdepth 1 -type f -name "*.tar" ! -name '*WPCA4096*' | sort -r)
# FILES=$(find "${BASEDir}" -type f -name "*.tar" ! -name '*WPCA4096*')

echo "${FILES}"

## Dataset path [Change according to yours]
if [ "$DATASET" = "mapillary" ]; then
  dataset_root_dir=/home/m.maqboolbhutta/usman_ws/datasets/Mapillary_Street_Level_Sequences/

elif [ "$DATASET" = "pitts" ]; then
  dataset_root_dir=/home/leo/usman_ws/datasets/2015netVLAD/Pittsburgh250k/
else
    echo "Invalid dataset choice"
    exit 1
fi

FAST_SCNN="/home/m.maqboolbhutta/usman_ws/datasets/official/fast_scnn/fast_scnn_citys.pth"

for RESUME in $FILES
do
  echo "Building PCA Model of $RESUME file..."
 # Extracting the directory and filename without extension
  dir=$(dirname "$RESUME")


  # take action on each file. $f store current file name
  filename=$(basename $RESUME .pth.tar)
  # Generating the new filename
  echo "filename: $filename"
  PCA_RESUME="${dir}/${filename}"
  PCA_RESUME="${PCA_RESUME}_WPCA4096.pth.tar"
  echo "PCA_Resume Path: $PCA_RESUME"


  filename=$(basename "$RESUME" .pth.tar)

  # Extracting the date and identifier from the path
  date_and_identifier=$(basename "$(dirname "$dir")")
  echo "$filename"


  # Generating the new filename
  SAVEFILENAME="${date_and_identifier}-${filename}"

  if [ -f "$PCA_RESUME" ]; then
      echo ""Skipping the addition of the PCA layer as $PCA_RESUME already exists.""
  else
      echo "Adding PCA Layer"
      python add_pca.py \
      --config_path=patchnetvlad/configs/train.ini \
      --resume_path=$RESUME \
      --dataset_root_dir=$dataset_root_dir \
      --dataset_choice=$DATASET \
      --method=${METHOD} \
      --fast_scnn=${FAST_SCNN}

  fi

  
  echo "Extracting Features of mapillarysf Index Images"

  python feature_extract.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_file_path=mapillarysf_imageNames_index.txt \
  --dataset_root_dir=/home/m.maqboolbhutta/usman_ws/datasets/ \
  --output_features_dir=/home/m.maqboolbhutta/usman_ws/models/features1/mapillarysf_index \
  --resume_path=${PCA_RESUME} \
  --method=${METHOD} \
  --fast_scnn=${FAST_SCNN}

  
  echo "Extracting Features of mapillarysf Query Images"

  python feature_extract.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_file_path=mapillarysf_imageNames_query.txt \
  --dataset_root_dir=/home/m.maqboolbhutta/usman_ws/datasets/ \
  --output_features_dir=/home/m.maqboolbhutta/usman_ws/models/features1/mapillarysf_query \
  --resume_path=${PCA_RESUME} \
  --method=${METHOD} \
  --fast_scnn=${FAST_SCNN}
  
  echo "Performing Features Matching and Recall Result of mapillarysf"
  python feature_match.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_root_dir=/home/m.maqboolbhutta/usman_ws/datasets/ \
  --query_file_path=mapillarysf_imageNames_query.txt \
  --index_file_path=mapillarysf_imageNames_index.txt \
  --query_input_features_dir /home/m.maqboolbhutta/usman_ws/models/features1/mapillarysf_query \
  --index_input_features_dir /home/m.maqboolbhutta/usman_ws/models/features1/mapillarysf_index \
  --result_save_folder=./patchnetvlad/results/mapillarysf

  echo "Extracting Features of mapillarycph Index Images"

  python feature_extract.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_file_path=mapillarycph_imageNames_index.txt \
  --dataset_root_dir=/home/m.maqboolbhutta/usman_ws/datasets/ \
  --output_features_dir=/home/m.maqboolbhutta/usman_ws/models/features1/mapillarycph_index \
  --resume_path=${PCA_RESUME} \
  --method=${METHOD} \
  --fast_scnn=${FAST_SCNN}

  
  echo "Extracting Features of mapillarycph Query Images"

  python feature_extract.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_file_path=mapillarycph_imageNames_query.txt \
  --dataset_root_dir=/home/m.maqboolbhutta/usman_ws/datasets/ \
  --output_features_dir=/home/m.maqboolbhutta/usman_ws/models/features1/mapillarycph_query \
  --resume_path=${PCA_RESUME} \
  --method=${METHOD} \
  --fast_scnn=${FAST_SCNN}

  echo "Performing Features Matching and Recall Result of mapillarycph"
  python feature_match.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_root_dir=/home/m.maqboolbhutta/usman_ws/datasets/ \
  --query_file_path=mapillarycph_imageNames_query.txt \
  --index_file_path=mapillarycph_imageNames_index.txt \
  --query_input_features_dir /home/m.maqboolbhutta/usman_ws/models/features1/mapillarycph_query \
  --index_input_features_dir /home/m.maqboolbhutta/usman_ws/models/features1/mapillarycph_index \
  --result_save_folder=./patchnetvlad/results/mapillarycph

cat patchnetvlad/results/mapillarycph/NetVLAD_predictions.txt patchnetvlad/results/mapillarysf/NetVLAD_predictions.txt > patchnetvlad/results/PatchNetVLAD_predictions_combined_mapval.txt

python ./patchnetvlad/training_tools/convert_kapture_to_msls.py patchnetvlad/results/PatchNetVLAD_predictions_combined_mapval.txt patchnetvlad/results/PatchNetVLAD_predictions_combined_msls.txt

echo "Recall Results of $PCA_RESUME file..."
python /home/m.maqboolbhutta/usman_ws/codes/mapillary_sls/evaluate.py --msls-root=/home/m.maqboolbhutta/usman_ws/datasets/Mapillary_Street_Level_Sequences/  --cities=cph,sf --prediction=patchnetvlad/results/PatchNetVLAD_predictions_combined_msls.txt

done

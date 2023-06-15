#!/bin/sh
PYTHON=${PYTHON:-"python3"}

# /media/leo/2C737A9872F69ECF/models/graphnetvlad/pytorch-netvlad/pittsburgh-sare_ind-lr0.001-05-Jun-2239/checkpoints/model_best.pth.tar

DATASET=tokyo

BASEDir="/media/leo/2C737A9872F69ECF/models/openibl/official-sare/conv5-sare_ind-lr0.001-tuple4/"
FILES="${BASEDir}*.tar"
echo "${FILES}"

for RESUME in $FILES
do
  echo "Building PCA Model of $RESUME file..."
  # take action on each file. $f store current file name
  
  python add_pca.py \
  --config_path patchnetvlad/configs/train.ini \
  --resume_path=$RESUME \
  --dataset_root_dir=/media/leo/2C737A9872F69ECF/datasets/maqbool-datasets/datasets-place-recognition/Test_Pitts250k \
  --dataset_choice=pitts
  
  filename=$(basename $RESUME .pth.tar)
  PCA_RESUME="${BASEDir}${filename}"
  PCA_RESUME="${PCA_RESUME}_WPCA"
  echo $PCA_RESUME
  
  echo "Extracting Features of Index Images"
  # PCA_RESUME="${BASEDir}{$filename}_WPCA"

  python feature_extract.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_file_path=pitts30k_imageNames_index.txt \
  --dataset_root_dir=/media/leo/2C737A9872F69ECF/datasets/maqbool-datasets/datasets-place-recognition/Test_Pitts250k/ \
  --output_features_dir=/media/leo/2C737A9872F69ECF/why-so-deepv2-data/pittsburgh/patch-netvlad-features/pitts30k_index \
  --resume_path=${PCA_RESUME}

  
  echo "Extracting Features of Query Images"

  python feature_extract.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_file_path=pitts30k_imageNames_query.txt \
  --dataset_root_dir=/media/leo/2C737A9872F69ECF/datasets/maqbool-datasets/datasets-place-recognition/Test_Pitts250k/ \
  --output_features_dir=/media/leo/2C737A9872F69ECF/why-so-deepv2-data/pittsburgh/patch-netvlad-features/pitts30k_query \
  --resume_path=${PCA_RESUME}
  echo "Performing Features Matching and Recall Result"

  python feature_match.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_root_dir==/media/leo/2C737A9872F69ECF/datasets/maqbool-datasets/datasets-place-recognition/Test_Pitts250k/ \
  --query_file_path=pitts30k_imageNames_query.txt \
  --index_file_path=pitts30k_imageNames_index.txt \
  --query_input_features_dir /media/leo/2C737A9872F69ECF/why-so-deepv2-data/pittsburgh/patch-netvlad-features/pitts30k_query \
  --index_input_features_dir /media/leo/2C737A9872F69ECF/why-so-deepv2-data/pittsburgh/patch-netvlad-features/pitts30k_index \
  --result_save_folder=./patchnetvlad/results/pitts30k-pytorchnetvlad \
  --ground_truth_path=./patchnetvlad/dataset_gt_files/pitts30k_test.npz



  


done



## PatchNetVLAD

## TODO
- [x] change the normlaization
- [x] create new cluster

### Clustering
use netvlad for creating the cluster trained in mapilary dataset
`./train.sh netvlad triplet`

### Training


Open the train.sh file and change the paths

```sh
#PC
./train.sh netvlad triplet
./train.sh netvlad sare_ind
./train.sh netvlad sare_joint
# graphvlad training
./train.sh graphvlad triplet
./train.sh graphvlad sare_ind
./train.sh graphvlad sare_joint

#SLURM
sbatch --j netvlad-triplet-8Jab train-slurm.sh netvlad triplet
### Aug 5
sbatch --j graphvlad-sare-ind-b48c80-lr001 train-slurm.sh graphvlad vgg16 sare_ind
## if has checkpoints
sbatch --j graphvlad-triplet-b50c80-lr001 train-slurm.sh graphvlad triplet /home/m.maqboolbhutta/usman_ws/models/patchnetvlad/graphvlad-triplet-lr0.01-08-May/May08_21-38-20_mapillary_nopanos/checkpoints/checkpoint_epoch5.pth.tar


```



### Testing
change the dataset path `dataset_root_dir` in `test.sh` file.
#### Testing on Local PC
```sh
# checkpoints
test.sh DATASET CHECKPOINTS_PATH
## example

test.sh graphvlad mapillary /home/leo/usman_ws/models/patchnetvlad/netvlad-sare_ind-05-Jan/Jan05_15-53-48_mapillary_nopanos/checkpoints/


```

#### Testing on Slurm
```sh
## Moving to the Server
cd /home/leo/usman_ws/models/patchnetvlad
rsync -ah --progress -e 'ssh -p 2222' graphvlad-sare_ind-lr0.01-30-Apr m.maqboolbhutta@hpg.rc.ufl.edu:/home/m.maqboolbhutta/usman_ws/models/patchnetvlad/pc/
# checkpoints
test.sh DATASET CHECKPOINTS_PATH
# RUN LIKE THIS
sbatch --j 30Apr-pc test-slurm.sh graphvlad mapillary /home/m.maqboolbhutta/usman_ws/models/patchnetvlad/pc/graphvlad-sare_ind-lr0.01-30-Apr/Apr30_20-08-22_mapillary_nopanos/checkpoints/


sbatch --j 08May test-slurm.sh graphvlad mapillary /home/m.maqboolbhutta/usman_ws/models/patchnetvlad/graphvlad-triplet-lr0.01-07-May/May07_11-02-35_mapillary_nopanos/checkpoints/checkpoint_epoch3.pth.tar

```


### Install
```sh
#PatchNetVLAD
module purge /
module load conda/22.11.1
conda create -n patchnetvlad python=3.8 -c conda-forge
conda activate patchnetvlad
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html h5py tensorboardx pandas faiss-gpu scikit-learn tqdm numpy==1.21.0 tensorboard


### Tried  using PYthon 3.10
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 h5py tensorboardx pandas faiss-gpu scikit-learn tqdm tensorboard numpy==1.22.4

## copy datafiles
rsync -ah --progress /home/leo/usman_ws/datasets/2015netVLAD  leo@IF-ABE-5JJTBW3.ad.ufl.edu:/home/leo/usman_ws/datasets/
rsync -ah --progress /home/leo/usman_ws/datasets/espnet-encoder  leo@IF-ABE-5JJTBW3.ad.ufl.edu:/home/leo/usman_ws/datasets/

```


ln -s mapillary_sls Mapillary_Street_Level_Sequences
git clone https://github.com/FrederikWarburg/mapillary_sls

### Official Trained Model

```sh
cd /blue/hmedeiros/m.maqboolbhutta/models/official-models
wget -O mapillary_WPCA4096.pth.tar https://cloudstor.aarnet.edu.au/plus/s/ZgW7DMEpeS47ELI/download
## or copy using trained local
rsync -ah --progress -e 'ssh -p 2222' /home/leo/usman_ws/models/patchnetvlad/official-vgg16/triplet-13-Dec/  m.maqboolbhutta@hpg.rc.ufl.edu:/blue/hmedeiros/m.maqboolbhutta/models/patch-netvlad/trained
```

Set the dataset root path in `test_model.sh`

```sh
## sbatch launch-slurm-test.sh DATASET CHECKPOINTS_PATH
sbatch launch-slurm-test.sh mappillary /blue/hmedeiros/m.maqboolbhutta/models/patch-netvlad/trained/Dec13_14-38-27_mapillary_nopanos/checkpoints

```




### Tensorboard

```sh
## Hipergator
ssh -L 7000:localhost:6007 m.maqboolbhutta@hpg.rc.ufl.edu
module load conda/24.1.2 && conda activate patchnetvlad && tensorboard --port=6007 --logdir=home/m.maqboolbhutta/usman_ws/models/patchnetvlad/fastscnn-v2/
## PC
tensorboard --logdir=patchnetvlad



/home/m.maqboolbhutta/usman_ws/models/patch-netvlad/trained

sbatch --export=logdir="/home/m.maqboolbhutta/usman_ws/models/patch-netvlad/trained",port=16006 tensorboard.sh

```


### symbolic Links
```sh
##dataset location | has test, train and val folder
/home/m.maqboolbhutta/usman_ws/datasets/mapillary_sls
## create symbolic links
cd /home/m.maqboolbhutta/usman_ws/codes/OpenIBL/examples/data/mapillary
ln -s /home/m.maqboolbhutta/usman_ws/datasets/msls images

rsync -ah --progress -e 'ssh -p 2222' /home/leo/usman_ws/datasets/VPR-datasets-downloader/msls/val  m.maqboolbhutta@hpg.rc.ufl.edu:/blue/hmedeiros/m.maqboolbhutta/datasets/msls
/home/leo/usman_ws/datasets/VPR-datasets-downloader/msls/val
```


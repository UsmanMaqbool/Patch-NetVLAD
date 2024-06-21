## Project Name
![GitHub Icon](https://img.icons8.com/fluent/24/000000/github.png): GitHub
![YouTube Icon](https://img.icons8.com/fluent/24/000000/pdf.png): Paper
![YouTube Icon](https://img.icons8.com/fluent/24/000000/youtube-play.png): YouTube

## Todo 📝

- [ ] Bugs
    - [ ] Todo
- [ ] Install
    - [ ] Todo
- [ ] Training
    - [ ] Todo
- [ ] Testing
    - [ ] Todo



## Training and Testing 🚀

Open the train.sh file and change the paths






### PC

```sh
#PC
./train.sh netvlad triplet
./train.sh netvlad sare_ind
./train.sh netvlad sare_joint
```
change the dataset path `dataset_root_dir` in `test.sh` file.
#### Testing on Local PC
```sh
# checkpoints
test.sh DATASET CHECKPOINTS_PATH
## example

test.sh graphvlad mapillary /home/leo/usman_ws/models/patchnetvlad/netvlad-sare_ind-05-Jan/Jan05_15-53-48_mapillary_nopanos/checkpoints/
```
#### Tensorboard


## PC
```sh
tensorboard --logdir=patchnetvlad
```

### Slurm
```sh
#SLURM
sbatch --j netvlad-triplet-8Jab train-slurm.sh netvlad triplet
sbatch --j graphvlad-triplet-b24c60-lr001 train-slurm.sh graphvlad triplet
## if has checkpoints
sbatch --j graphvlad-triplet-b50c80-lr001 train-slurm.sh graphvlad triplet /home/m.maqboolbhutta/usman_ws/models/patchnetvlad/graphvlad-triplet-lr0.01-08-May/May08_21-38-20_mapillary_nopanos/checkpoints/checkpoint_epoch5.pth.tar
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



#### Tensorboard

```sh
## Hipergator
ssh -L 7000:localhost:6007 m.maqboolbhutta@hpg.rc.ufl.edu
module load conda/24.1.2 && conda activate patchnetvlad && tensorboard --logdir=/home/m.maqboolbhutta/usman_ws/models/patchnetvlad/ --port=6007

/home/m.maqboolbhutta/usman_ws/models/patch-netvlad/trained

sbatch --export=logdir="/home/m.maqboolbhutta/usman_ws/models/patch-netvlad/trained",port=16006 tensorboard.sh
```

## Install ⚙️
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

### Configure 🪛

### Dataset 🖼️
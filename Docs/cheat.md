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

## SFRS

### Openibl
The given code snippet is part of a machine learning training loop, which involves a multi-generational training process with model checkpointing, evaluation, and optimization. Here’s a step-by-step explanation of the key components and their roles:

#### Initialization
1. **Model and Cache Initialization:**
    ```python
    model = get_model(args)
    model_cache = get_model(args)
    ```
    - Two instances of the model are created. `model` will be used for training, and `model_cache` will be used for maintaining a stable reference to the model's parameters for certain computations.

2. **Checkpoint Loading:**
    ```python
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']+1
        start_gen = checkpoint['generation']
        best_recall5 = checkpoint['best_recall5']
        if (args.rank==0):
            print("=> Start epoch {}  best recall5 {:.1%}"
                  .format(start_epoch, best_recall5))
    ```
    - If the training is being resumed from a checkpoint, the model's state is restored from the checkpoint, and training variables are set to the values stored in the checkpoint.

3. **Evaluator Initialization:**
    ```python
    evaluator = Evaluator(model)
    ```
    - An `Evaluator` object is created to evaluate the model’s performance on a validation dataset.

4. **Initial Model Evaluation:**
    ```python
    if (args.rank==0):
        print("Test the initial model:")
    recalls = evaluator.evaluate(val_loader, ...)
    ```
    - The initial model (untrained or loaded from checkpoint) is evaluated to get baseline performance metrics.

#### Trainer Initialization
5. **Trainer Initialization:**
    ```python
    trainer = SFRSTrainer(model, model_cache, margin=args.margin**0.5, ...)
    if ((args.cache_size<args.tuple_size) or (args.cache_size>len(dataset.q_train))):
        args.cache_size = len(dataset.q_train)
    ```
    - An `SFRSTrainer` object is created, which will handle the training process. The cache size for training is adjusted based on the dataset size and tuple size.

#### Training Loop
6. **Generational Loop:**
    ```python
    for gen in range(start_gen, args.generations):
        model_cache.load_state_dict(model.state_dict())
        model.module._init_params()
    ```
    - For each generation, the model’s state is copied to the `model_cache`, and the model parameters are initialized.

7. **Optimizer and Scheduler Initialization:**
    ```python
    optimizer = torch.optim.SGD(...)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(...)
    ```
    - The optimizer (SGD) and learning rate scheduler are initialized.

8. **Epoch Loop:**
    ```python
    if (gen==0):
        start_epoch = args.epochs-1

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(args.seed+epoch)
        if (epoch%args.step_size==0):
            args.cache_size = args.cache_size * (2 ** (epoch // args.step_size))
    ```
    - For each epoch, the training loop is configured. The cache size is adjusted at specific intervals (`step_size`).

9. **Subset Generation:**
    ```python
    g = torch.Generator()
    g.manual_seed(args.seed+epoch)
    subset_indices = torch.randperm(len(dataset.q_train), generator=g).long().split(args.cache_size)
    ```
    - A random subset of the training data is generated for the current epoch.

10. **Sub-Epoch Training:**
    ```python
    for subid, subset in enumerate(subset_indices):
        update_sampler(sampler, model, train_extract_loader, ...)
        synchronize()

        trainer.train(gen, epoch, subid, train_loader, optimizer, ...)
        synchronize()
    ```
    - The training data is further divided into smaller subsets, and the model is trained on each subset. Synchronization ensures that updates are applied consistently across different processes (useful in distributed training).

11. **Evaluation and Checkpointing:**
    ```python
    if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
        recalls = evaluator.evaluate(val_loader, ...)
        is_best = recalls[1] > best_recall5
        best_recall5 = max(recalls[1], best_recall5)

        if (args.rank==0):
            save_checkpoint({...}, is_best, fpath=...)
            print(...)
    ```
    - After certain intervals or at the end of each epoch, the model is evaluated, and performance metrics are recorded. If the model achieves a new best recall@5 score, it is saved as a checkpoint.

12. **Learning Rate Scheduling:**
    ```python
    lr_scheduler.step()
    synchronize()
    ```
    - The learning rate is adjusted according to the scheduler after each epoch, and synchronization ensures consistency across processes.

13. **Reset for Next Generation:**
    ```python
    start_epoch = 0
    ```
    - The starting epoch is reset for the next generation.

#### Summary
The code follows a structured process for training a machine learning model over multiple generations, with periodic evaluation, checkpointing, and adaptive adjustments to training parameters. This process helps in progressively improving the model's performance while maintaining checkpoints for recovery and further fine-tuning.


### TODO
To create a flowchart in Markdown, I'll use a combination of bullet points and indentation to represent the flow of the program. Markdown doesn't support actual flowchart diagrams, so this textual representation is the best way to outline the program flow.

- Start
  - For each generation `gen` in `range(start_gen, args.generations)`.
    - Update model cache and initialize model parameters
    - Initialize optimizer and learning rate scheduler
    - If first generation, set [`start_epoch` to `args.epochs - 1`
    - For each epoch in `range(start_epoch, args.epochs)`
      - Set sampler for the current epoch
      - If epoch is a step size, double the cache size
      - Generate subset indices for training data
        - For each subset
          - Update sampler with current subset
          - Synchronize processes
          - Train model with current subset
          - Synchronize processes again
      - If it's time to evaluate or last epoch
        - Evaluate model
        - Update best recall if current recall is better
        - If best recall, save checkpoint
        - Print evaluation results
      - Step learning rate scheduler
      - Synchronize processes
    - Reset `start_epoch`to 0 for next generation
- End

This textual representation outlines the main steps and conditional logic within the provided Python code, mimicking a flowchart's structure using Markdown formatting.

### PatchNetVLAD


```py
# Model initialization
model = model.to(device)

if opt.resume_path:
    checkpoint = load_checkpoint(opt.resume_path)
    optimizer.load_state_dict(checkpoint['optimizer'])

print('===> Loading dataset(s)')
exclude_panos_training = not config['train'].getboolean('includepanos')
train_dataset = MSLS(opt.dataset_root_dir, mode='train', nNeg=int(config['train']['nNeg']), transform=input_transform(),
                     bs=int(config['train']['cachebatchsize']), threads=opt.threads, margin=float(config['train']['margin']),
                     exclude_panos=exclude_panos_training)

validation_dataset = MSLS(opt.dataset_root_dir, mode='val', transform=input_transform(),
                          bs=int(config['train']['batchsize']), threads=opt.threads,
                          margin=float(config['train']['margin']), posDistThr=25)

print('===> Training query set:', len(train_dataset.qIdx))
print('===> Evaluating on val set, query count:', len(validation_dataset.qIdx))

# SummaryWriter for TensorBoard
lr_str = str(float(config['train']['lr'])).replace('.', '_')
writer = SummaryWriter(log_dir=join(opt.save_path + '-lr-' + lr_str, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + opt.identifier))

# Checkpoint directory
logdir = writer.file_writer.get_logdir()
opt.save_file_path = join(logdir, 'checkpoints')
makedirs(opt.save_file_path)

# Initialize variables for tracking best score and patience
not_improved = 0
best_score = 0
start_epoch = opt.start_epoch + 1

if opt.resume_path:
    not_improved = checkpoint['not_improved']
    best_score = checkpoint['best_score']
    start_epoch = checkpoint['epoch'] + 1

# Multi-generational training process
for gen in range(start_gen, opt.generations):
    # Clone the model state to the model_cache
    model_cache = copy.deepcopy(model.state_dict())
    model.apply(init_params)

    # Optimizer and scheduler
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=float(config['train']['lr']), momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config['train']['stepsize']), gamma=0.5)

    for epoch in trange(start_epoch, opt.nEpochs + 1, desc='Epoch number'.rjust(15), position=0):
        train_epoch(train_dataset, model, optimizer, criterion, encoder_dim, device, epoch, opt, config, writer)

        if scheduler is not None:
            scheduler.step()

        if (epoch % int(config['train']['evalevery'])) == 0:
            recalls = val(validation_dataset, model, encoder_dim, device, opt, config, writer, epoch,
                          write_tboard=True, pbar_position=1)
            is_best = recalls[5] > best_score
            if is_best:
                not_improved = 0
                best_score = recalls[5]
            else:
                not_improved += 1

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'recalls': recalls,
                'best_score': best_score,
                'not_improved': not_improved,
                'optimizer': optimizer.state_dict(),
                'parallel': isParallel,
            }, opt, is_best)

            if int(config['train']['patience']) > 0 and not_improved > (int(config['train']['patience']) / int(config['train']['evalevery'])):
                print('Performance did not improve for', config['train']['patience'], 'epochs. Stopping.')
                break

    # Reset the starting epoch for the next generation
    start_epoch = 1
```
To convert the given code into the multi-generational style from the previous code, we will include generational looping, checkpointing, evaluation, and optimization steps, integrating the logic into a structured process. Here is the converted code:

```python
# Model initialization
model = model.to(device)

if opt.resume_path:
    checkpoint = load_checkpoint(opt.resume_path)
    optimizer.load_state_dict(checkpoint['optimizer'])

print('===> Loading dataset(s)')
exclude_panos_training = not config['train'].getboolean('includepanos')
train_dataset = MSLS(opt.dataset_root_dir, mode='train', nNeg=int(config['train']['nNeg']), transform=input_transform(),
                     bs=int(config['train']['cachebatchsize']), threads=opt.threads, margin=float(config['train']['margin']),
                     exclude_panos=exclude_panos_training)

validation_dataset = MSLS(opt.dataset_root_dir, mode='val', transform=input_transform(),
                          bs=int(config['train']['batchsize']), threads=opt.threads,
                          margin=float(config['train']['margin']), posDistThr=25)

print('===> Training query set:', len(train_dataset.qIdx))
print('===> Evaluating on val set, query count:', len(validation_dataset.qIdx))

# SummaryWriter for TensorBoard
lr_str = str(float(config['train']['lr'])).replace('.', '_')
writer = SummaryWriter(log_dir=join(opt.save_path + '-lr-' + lr_str, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + opt.identifier))

# Checkpoint directory
logdir = writer.file_writer.get_logdir()
opt.save_file_path = join(logdir, 'checkpoints')
makedirs(opt.save_file_path)

# Initialize variables for tracking best score and patience
not_improved = 0
best_score = 0
start_epoch = opt.start_epoch + 1

if opt.resume_path:
    not_improved = checkpoint['not_improved']
    best_score = checkpoint['best_score']
    start_epoch = checkpoint['epoch'] + 1

# Multi-generational training process
for gen in range(start_gen, opt.generations):
    # Clone the model state to the model_cache
    model_cache = copy.deepcopy(model.state_dict())
    model.apply(init_params)

    # Optimizer and scheduler
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=float(config['train']['lr']), momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config['train']['stepsize']), gamma=0.5)

    for epoch in trange(start_epoch, opt.nEpochs + 1, desc='Epoch number'.rjust(15), position=0):
        train_epoch(train_dataset, model, optimizer, criterion, encoder_dim, device, epoch, opt, config, writer)

        if scheduler is not None:
            scheduler.step()

        if (epoch % int(config['train']['evalevery'])) == 0:
            recalls = val(validation_dataset, model, encoder_dim, device, opt, config, writer, epoch,
                          write_tboard=True, pbar_position=1)
            is_best = recalls[5] > best_score
            if is_best:
                not_improved = 0
                best_score = recalls[5]
            else:
                not_improved += 1

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'recalls': recalls,
                'best_score': best_score,
                'not_improved': not_improved,
                'optimizer': optimizer.state_dict(),
                'parallel': isParallel,
            }, opt, is_best)

            if int(config['train']['patience']) > 0 and not_improved > (int(config['train']['patience']) / int(config['train']['evalevery'])):
                print('Performance did not improve for', config['train']['patience'], 'epochs. Stopping.')
                break

    # Reset the starting epoch for the next generation
    start_epoch = 1
```

### Explanation:

1. **Model Initialization:**
   - The model is moved to the specified device.
   - If a checkpoint path is provided, the optimizer state is loaded from the checkpoint.

2. **Dataset Loading:**
   - Training and validation datasets are loaded with specified configurations.

3. **SummaryWriter Initialization:**
   - A `SummaryWriter` is created for logging with TensorBoard.

4. **Checkpoint Directory:**
   - A directory for saving checkpoints is created.

5. **Tracking Variables Initialization:**
   - Variables for tracking the best score and the number of epochs without improvement are initialized.
   - The starting epoch is determined based on whether a checkpoint is being resumed.

6. **Multi-generational Training Loop:**
   - For each generation:
     - The model state is cloned to `model_cache`.
     - Optimizer and learning rate scheduler are initialized.

7. **Epoch Loop:**
   - For each epoch within the generation:
     - The model is trained for one epoch.
     - The scheduler updates the learning rate if defined.
     - Evaluation is performed at specified intervals.
     - If the model achieves a new best recall@5 score, it is saved as a checkpoint.
     - If the model does not improve for a specified number of evaluations, training stops early.

8. **Reset for Next Generation:**
   - The starting epoch is reset for the next generation.

## Debug
### VScode Training

```json
{
    "version": "0.2.0",
    "configurations": [
      {
        "name": "Python: Train Script",
        "type": "python",
        "request": "launch",
        "program": "${workspaceFolder}/train.py",
        "args": [
          "--save_every_epoch",
          "--config_path=${workspaceFolder}/patchnetvlad/configs/train.ini",
          "--cache_path=/home/leo/usman_ws/datasets/2015netVLAD/official/",
          "--save_path=/home/leo/usman_ws/models/patchnetvlad/netvlad-triplet",
          "--dataset_root_dir=/home/leo/usman_ws/datasets/mapillary_sls/",
          "--loss=triplet",
          "--method=netvlad",
          "--esp_encoder=/home/leo/usman_ws/datasets/espnet-encoder/espnet_p_2_q_8.pth",
          "--cluster_path=/home/leo/usman_ws/datasets/2015netVLAD/official/centroids/vgg16_mapillary_16_desc_cen.hdf5",
          "--threads=6",
          "--vd16_offtheshelf_path=/home/leo/usman_ws/datasets/2015netVLAD/official/vd16_offtheshelf_conv5_3_max.pth",
          "--resume_path=/home/leo/usman_ws/models/patchnetvlad/debug/"
        ],
        "console": "integratedTerminal",
        "env": {
          "PYTHONUNBUFFERED": "1"
        }
      }
    ]
}
  
```

## Training and Testing 🚀

Open the train.sh file and change the paths






### PC

#### Training 
```sh
#PC
./train.sh netvlad triplet
./train.sh netvlad sare_ind
./train.sh netvlad sare_joint
```
change the dataset path `dataset_root_dir` in `test.sh` file.

#### Testing
```sh
# checkpoints
test.sh DATASET CHECKPOINTS_PATH
## example

test.sh graphvlad mapillary /home/leo/usman_ws/models/patchnetvlad/netvlad-sare_ind-05-Jan/Jan05_15-53-48_mapillary_nopanos/checkpoints/
```
#### Tensorboard
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
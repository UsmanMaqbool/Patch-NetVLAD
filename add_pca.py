#!/usr/bin/env python

'''
MIT License

Copyright (c) 2021 Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford and Tobias Fischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Trains a PCA model and adds a WPCA layer to an existing checkpoint.
'''


from __future__ import print_function

import argparse
import configparser
import os
import random
from os.path import join, isfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

import numpy as np

from patchnetvlad.training_tools.tools import pca
from patchnetvlad.tools.datasets import input_transform
from patchnetvlad.models.models_generic import get_backend, get_model, create_model, create_model_pca, Flatten, L2Norm
from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR

from tqdm.auto import tqdm

from patchnetvlad.training_tools.msls import MSLS, ImagesFromList
from patchnetvlad.tools.datasets import PlaceDataset

# taken from 
# https://github.com/UsmanMaqbool/OpenIBL
from pca import PCA
import code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Patch-NetVLAD-add-pca')

    parser.add_argument('--config_path', type=str, default=join(PATCHNETVLAD_ROOT_DIR, 'configs/train.ini'),
                        help='File name (with extension) to an ini file that stores most of the configuration data for patch-netvlad')
    parser.add_argument('--resume_path', type=str, default='',
                        help='Full path and name (with extension) to load checkpoint from, for resuming training.')
    parser.add_argument('--dataset_root_dir', type=str, default='/work/qvpr/data/raw/Mapillary_Street_Level_Sequences',
                        help='Root directory of dataset')
    parser.add_argument('--dataset_choice', type=str, default='mapillary', help='choice of mapillary or pitts, for PCA',
                        choices=['mapillary', 'pitts'])
    parser.add_argument('--threads', type=int, default=6, help='Number of threads for each data loader to use')
    parser.add_argument('--nocuda', action='store_true', help='If true, use CPU only. Else use GPU.')
    parser.add_argument('--vd16_offtheshelf_path', type=str, default=None,
                        help='NetVLAD Off the Shelf VGG Weights.')

    opt = parser.parse_args()
    print(opt)

    configfile = opt.config_path
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    random.seed(int(config['train']['seed']))
    np.random.seed(int(config['train']['seed']))
    torch.manual_seed(int(config['train']['seed']))
    if cuda:
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed(int(config['train']['seed']))

    print('===> Building model')

    encoder_dim, encoder = get_backend(None)
      

    if opt.resume_path: # must resume for PCA
        if isfile(opt.resume_path):
            print("=> loading checkpoint '{}'".format(opt.resume_path))
            checkpoint = torch.load(opt.resume_path, map_location=lambda storage, loc: storage)            
             # for i in checkpoint['state_dict']['module.net_vlad.centroids']:
            #     print(i)
            
            # Create a new state_dict without the 'module.' prefix
            new_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('module.'):
                    new_key = key[7:]  # Remove the 'module.' prefix
                else:
                    new_key = key
                new_state_dict[new_key] = value

            config['global_params']['num_clusters'] = str(new_state_dict['net_vlad.centroids'].shape[0])
            pool_layer = get_model(encoder, encoder_dim, config['global_params'], append_pca_layer=False)       

            # model = get_model(encoder, encoder_dim, config['global_params'], append_pca_layer=False)
            model = create_model('embednetpca',encoder, pool_layer)
            
            # Load the new state_dict into your model
            model.load_state_dict(new_state_dict)
            
            opt.start_epoch = checkpoint['epoch']

            print("=> loaded checkpoint '{}'".format(opt.resume_path, ))
        else:
            raise FileNotFoundError("=> no checkpoint found at '{}'".format(opt.resume_path))
    else:
        raise ValueError("Need an existing checkpoint in order to run PCA")

    isParallel = False
    if int(config['global_params']['nGPU']) > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        model.pool = nn.DataParallel(model.pool)
        isParallel = True

    model = model.to(device)

    pool_size = encoder_dim
    if config['global_params']['pooling'].lower() == 'netvlad':
        pool_size *= int(config['global_params']['num_clusters'])

    print('===> Loading PCA dataset(s)')

    nFeatures = 10000
    if opt.dataset_choice == 'mapillary':
        exlude_panos_training = not config['train'].getboolean('includepanos')

        pca_train_set = MSLS(opt.dataset_root_dir, mode='test', cities='train',
                             transform=input_transform(),
                             bs=int(config['train']['cachebatchsize']), threads=opt.threads,
                             margin=float(config['train']['margin']),
                             exclude_panos=exlude_panos_training)

        pca_train_images = pca_train_set.dbImages
    elif opt.dataset_choice == 'pitts':
        dataset_file_path = join(PATCHNETVLAD_ROOT_DIR, 'dataset_imagenames', 'pitts30k_imageNames_index.txt')
        pca_train_set = PlaceDataset(None, dataset_file_path, opt.dataset_root_dir, None, config['train'])
        pca_train_images = pca_train_set.images
    else:
        raise ValueError('Unknown dataset choice: ' + opt.dataset_choice)

    if nFeatures > len(pca_train_images):
        nFeatures = len(pca_train_images)

    sampler = SubsetRandomSampler(np.random.choice(len(pca_train_images), nFeatures, replace=False))

    data_loader = DataLoader(
        dataset=ImagesFromList(pca_train_images, transform=input_transform()),
        num_workers=opt.threads, batch_size=int(config['train']['cachebatchsize']), shuffle=False,
        pin_memory=cuda,
        sampler=sampler)

    print('===> Do inference to extract features and save them.')

    model.eval()
    with torch.no_grad():
        tqdm.write('====> Extracting Features')

        dbFeat = np.empty((len(data_loader.sampler), pool_size))
        print('Compute', len(dbFeat), 'features')

        for iteration, (input_data, indices) in enumerate(tqdm(data_loader)):
            input_data = input_data.to(device)
            vlad_encoding = model(input_data.to(device))
            out_vectors = vlad_encoding.detach().cpu().numpy()
            # this allows for randomly shuffled inputs
            for idx, out_vector in enumerate(out_vectors):
                dbFeat[iteration * data_loader.batch_size + idx, :] = out_vector

            del input_data, vlad_encoding

    print('===> Compute PCA, takes a while')
    num_pcs = int(config['global_params']['num_pcs'])

    # print("till here")
    num_pcs = int(config['global_params']['num_pcs'])
    u, lams, mu = pca(dbFeat, num_pcs)

    u = u[:, :num_pcs]
    lams = lams[:num_pcs]

    print('===> Add PCA Whiten')
    u = np.matmul(u, np.diag(np.divide(1., np.sqrt(lams + 1e-9))))
   
    utmu = np.matmul(u.T, mu)



    ## Mine
    # dbFeat = torch.from_numpy(dbFeat)
    # dbFeat = list(dbFeat)
    # if (len(dbFeat)>10000):
    #     dbFeat = random.sample(dbFeat, 10000)
    # pca = PCA(pca_n_components= num_pcs, pca_whitening = True)
    # dbFeat = torch.stack(dbFeat)
    # u, lams, mu, utmu = pca.train(dbFeat)
    
        
    pca_conv = nn.Conv2d(pool_size, num_pcs, kernel_size=(1, 1), stride=1, padding=0)
    # noinspection PyArgumentList
    
    pca_conv.weight = nn.Parameter(torch.from_numpy(np.expand_dims(np.expand_dims(u.T, -1), -1)))
    # noinspection PyArgumentList
    pca_conv.bias = nn.Parameter(torch.from_numpy(-utmu).squeeze())
    # aa = torch.from_numpy(np.expand_dims(np.expand_dims(u.T, -1), -1)[0])
    # bb = torch.from_numpy(-utmu)
    # code.interact(local=locals())
    pca_str = 'pca_layer'
    model.add_module(pca_str, pca_conv)  # Naming the convolutional layer directly
    model.add_module(pca_str + '_flatten', Flatten())
    model.add_module(pca_str + '_l2norm', L2Norm(dim=-1))

    save_path = opt.resume_path.replace(".pth.tar", "_WPCA" + str(num_pcs) + ".pth.tar")

    torch.save({'num_pcs': num_pcs, 'state_dict': model.state_dict()}, save_path)

    torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
    # memory after runs

    print('Done')
"""
Anatomical evaluation of generated tumor.
The goals is to evalute the alligment between the condition and the generated tumor.
"""
import argparse
import glob
import os
import pickle
import logging

import cv2
import pandas as pd
import seaborn as sns
import torch
import torchvision
import yaml
import numpy as np
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

from intraoperative_us.diffusion.evaluation.investigate_vae import get_config_value, get_best_model
from intraoperative_us.diffusion.dataset.dataset import IntraoperativeUS, GenerateDataset
from intraoperative_us.diffusion.utils.utils import get_best_model, load_autoencoder, get_number_parameter

from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
from scipy.spatial.distance import cdist
from PIL import Image
from torchvision import transforms
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## set reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def tumor_not_tumor_tissue(ius, mask, show_plot=False):
    """
    Compute the tumor and not tumor bool mask
    """

    tumor_mask = mask > 0
    background_mask = ius < -0.96
    not_tumor =  ~(tumor_mask | background_mask)

    tumor_tissue = (ius[tumor_mask] + 1) / 2 
    not_tumor_tissue = (ius[not_tumor] + 1) / 2

    if show_plot:
        plt.figure(figsize=(18,12), num='tumor_not_tumor', tight_layout=True)
        plt.subplot(1,3,1)
        plt.title('Tumor', fontsize=30) 
        plt.imshow(tumor_mask, cmap='Oranges')
        plt.axis('off') 
        plt.subplot(1,3,2)
        plt.title('Not tumor', fontsize=30)
        plt.imshow(not_tumor, cmap='Greens')
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.title('iUS', fontsize=30)
        plt.imshow(ius, cmap='gray')
        plt.axis('off')
        plt.show()

    return tumor_tissue, not_tumor_tissue

    

def infer(par_dir, conf, trial, split, experiment, epoch, guide_w, scheduler, show_gen_mask):
    ######## Read the config file #######
    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    dataset_config = config['dataset_params']
    autoencoder_model_config = config['autoencoder_params']
    condition_config = get_config_value(autoencoder_model_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']



    # Train dataset
    data_img = IntraoperativeUS(size= [dataset_config['im_size_h'], dataset_config['im_size_w']],
                               dataset_path= dataset_config['dataset_path'],
                               im_channels= dataset_config['im_channels'],
                               splitting_json=dataset_config['splitting_json'], 
                               split='train',
                               splitting_seed=dataset_config['splitting_seed'],
                               train_percentage=dataset_config['train_percentage'],
                               val_percentage=dataset_config['val_percentage'],
                               test_percentage=dataset_config['test_percentage'],
                               condition_config=config['ldm_params']['condition_config'],
                               data_augmentation=False)
    logging.info(f'len train data {len(data_img)}')
    data_loader = DataLoader(data_img, batch_size=1, shuffle=False, num_workers=8)


    data_gen = GenerateDataset(par_dir, trial, split, experiment, guide_w, scheduler, epoch,
                               size=[dataset_config['im_size_h'], dataset_config['im_size_w']], input_channels=dataset_config['im_channels'],
                               mask=True)
    data_loader_gen = DataLoader(data_gen, batch_size=1, shuffle=False, num_workers=8)
    logging.info(f'len gen data {len(data_gen)}')

    ## compute the istograms of tumor and not tumor for real image
    real_tumor_list, real_not_tumor_list = [], []
    for i, data in enumerate(data_loader):
        img = data[0]
        mask = data[1]['image']
        tumor, not_tumor = tumor_not_tumor_tissue(img[0,0,:,:].cpu().numpy(), mask[0,0,:,:].cpu().numpy(), show_plot=False) 
        real_tumor_list.append(tumor)
        real_not_tumor_list.append(not_tumor)

    ## compute the istograms of tumor and not tumor for generated image
    gen_tumor_list, gen_not_tumor_list = [], []
    for i, data in enumerate(data_loader_gen):
        # plt the generated image and the mask
        if i < len(data_loader):
            img = data[0]
            mask = data[1]
            tumor, not_tumor = tumor_not_tumor_tissue(img[0,0,:,:].cpu().numpy(), mask[0,0,:,:].cpu().numpy(), show_plot=show_gen_mask)
            gen_tumor_list.append(tumor)
            gen_not_tumor_list.append(not_tumor)
        else:
            break
            # tumor, not_tumor = tumor_not_tumor_tissue(gen_img[0,:,:].cpu().numpy(), mask[0,:,:].cpu().numpy())


    # from a list of list to a single list
    real_tumor_list = [item for sublist in real_tumor_list for item in sublist]
    real_not_tumor_list = [item for sublist in real_not_tumor_list for item in sublist]
    gen_tumor_list = [item for sublist in gen_tumor_list for item in sublist]
    gen_not_tumor_list = [item for sublist in gen_not_tumor_list for item in sublist]

    fig, ax = plt.subplots(1, 2, figsize=(18,10), num='Analysis_Tumor_not_Tumor', tight_layout=True)
    bins = np.linspace(0, 1, 50)
    stat, p = stats.ranksums(real_tumor_list, gen_tumor_list)
    ax[0].set_title(f'Tumor - p-value={p:.4f}', fontsize=30)
    ax[0].hist(real_tumor_list, bins=bins, label=f'Real data', color='C3', alpha=0.5, density=True)
    ax[0].hist(gen_tumor_list,  bins=bins, label='Gen data', color='C1', alpha=0.5, density=True)
    ax[0].set_xlabel('Pixel Intensity', fontsize=30)
    ax[0].set_ylabel('Frequency (%)', fontsize=30)
    ax[0].tick_params(axis='both', which='major', labelsize=30)
    ax[0].set_ylim([0, 10.5])
    ax[0].legend(fontsize=30)
    ax[0].grid(linestyle=':')

    stat, p = stats.ranksums(real_not_tumor_list, gen_not_tumor_list)
    ax[1].set_title(f'Not tumor - p-value={p:.4f}', fontsize=30)
    ax[1].hist(real_not_tumor_list, bins=bins, label=f'Real data', color='blue', alpha=0.5, density=True)
    ax[1].hist(gen_not_tumor_list,  bins=bins, label='Gen data', color='green',alpha=0.5, density=True)
    ax[1].set_xlabel('Not tumor size', fontsize=30)
    ax[1].set_ylim([0, 10.5])
    ax[1].tick_params(axis='both', which='major', labelsize=30)
    ax[1].legend(fontsize=30)
    ax[1].grid(linestyle=':')

    
    fig, ax = plt.subplots(1, 2, figsize=(18,10), num='Hist_Tumor_not_Tumor', tight_layout=True)
    ax[0].set_title('Real image', fontsize=30)
    ax[0].hist(real_tumor_list, alpha=0.5, bins=bins, color='C3', label='tumor', density=True)
    ax[0].hist(real_not_tumor_list, alpha=0.5, bins=bins, color='blue', label='not tumor', density=True)
    ax[0].legend(fontsize=26)
    ax[0].tick_params(axis='both', which='major', labelsize=26)
    ax[0].set_xlabel('Pixel Intensity', fontsize=28)
    ax[0].set_ylabel('Frequency (%)', fontsize=28)
    ax[0].set_ylim([0, 10.5])
    ax[0].grid(linestyle=':')
    # Generated image
    ax[1].set_title('Generated image', fontsize=30)
    ax[1].hist(gen_tumor_list, alpha=0.5, bins=bins, color='C1', label='tumor', density=True)
    ax[1].hist(gen_not_tumor_list, alpha=0.5, bins=bins,  color='green', label='not tumor', density=True)
    ax[1].legend(fontsize=26)
    ax[1].tick_params(axis='both', which='major', labelsize=26)
    ax[1].set_xlabel('Pixels Intensity', fontsize=28)
    ax[1].set_ylim([0, 10.5])
    ax[1].set_ylabel('Frequency (%)', fontsize=28)
    ax[1].grid(linestyle=':')
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anatomical evaluation of echogenicity')
    parser.add_argument('--save_folder', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model",
                                                   help='folder to save the model')
    parser.add_argument('--type_image', type=str, default='ius', help='type of image to evaluate, ius or mask')
    parser.add_argument('--split', type=str, default='split_1', help='splitting name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the 
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--guide_w', type=float, default=0.0, help='guide_w for the conditional model, w=-1 [unconditional], w=0 [vanilla conditioning], w>0 [guided conditional]')
    parser.add_argument('--scheduler', type=str, default='ddpm', help='sheduler used for sampling, i.e. ddpm, pndm')
    parser.add_argument('--epoch', type=int, default=99, help='epoch to sample, this is the epoch of cond ldm model') 
    parser.add_argument('--show_gen_mask', action='store_true', help="show the generative and mask images, default=False")
    parser.add_argument('--log', type=str, default='debug', help='Logging level')
    args = parser.parse_args()

    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    experiment_dir = os.path.join(args.save_folder, args.type_image, args.trial, args.split)
    if 'vae' in os.listdir(experiment_dir): config = os.path.join(experiment_dir, 'vae', 'config.yaml')

    infer(par_dir = os.path.join(args.save_folder, args.type_image), conf=config, trial=args.trial, split = args.split,
         experiment=args.experiment, epoch=args.epoch, guide_w=args.guide_w, scheduler=args.scheduler, show_gen_mask=args.show_gen_mask)
    plt.show()
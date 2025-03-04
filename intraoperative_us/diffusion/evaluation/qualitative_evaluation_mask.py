"""
Qualitative comparison between real image (training) and generated image 
"""
import argparse
import glob
import os
import pickle
import logging

import torch
import torchvision
import yaml
import numpy as np
import cv2
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt

from intraoperative_us.diffusion.evaluation.investigate_vae import get_config_value, get_best_model
from intraoperative_us.diffusion.dataset.dataset import IntraoperativeUS, GenerateDataset, IntraoperativeUS_mask, GeneratedMaskDataset
from intraoperative_us.diffusion.models.vae import VAE

from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
from scipy.spatial.distance import cdist
from PIL import Image
from torchvision import transforms
from scipy.ndimage import center_of_mass

from scipy.fft import fft, fftfreq
import numpy as np
import numpy as np
from scipy import signal
from sklearn.preprocessing import MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fft_descriptor(mask, n_points=100, show_flot=False):
    """
    Compute the shape context descriptors of the mask
    """
    ## find the contours of the mask
    # Ensure binary mask is in the correct format (H, W) and uint8
    binary_mask = (mask > 0.5).astype(np.uint8)  # Convert to binary (0 or 1)
    binary_mask *= 255  # Scale to 0-255 (needed for OpenCV)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:  # If contours list is empty
        print("Warning: No contours found in the mask.")
        return None, None  # Or return default values

    contour = max(contours, key=cv2.contourArea)  # Take largest contour
    contour_points = contour.squeeze()

    # Sample 100 points uniformly
    if len(contour_points) < n_points:
        sampled_points = contour_points
    else:
        indices = np.linspace(0, len(contour_points)-1, n_points).astype(int)
        sampled_points = contour_points[indices]

    # CENTROIDS
    mean_x = np.mean(sampled_points[:,0])
    mean_y = np.mean(sampled_points[:,1])

    ## compute the distance between the centroid and the sampled points
    distances = np.linalg.norm(sampled_points - np.array([mean_x, mean_y]), axis=1)

    ## compute the fft of the distances
    N = len(distances)
    yf = fft(distances - np.mean(distances))
    xf = fftfreq(N, 1/N)[:N//2]

    # Compute Power Spectral Density (PSD)
    psd = (np.abs(yf) ** 2) / N  # Normalized power
    total_power = np.sum(psd)
 
    if show_flot:
        plt.figure(figsize=(25, 8), tight_layout=True)
        plt.subplot(1, 3, 1)
        plt.imshow(mask)
        for i in range(len(sampled_points)):
            plt.scatter(sampled_points[i,0], sampled_points[i,1], c='r', s=100)
        plt.scatter(mean_x, mean_y, c='b', s=100)
        
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.title('Distance from centroid', fontsize=26)
        plt.plot(distances , 'r', lw=6)
        plt.xlabel('Points', fontsize=20)
        plt.ylabel('Distance from centroid (D)', fontsize=20)
        plt.ylim(0,100)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()

        plt.subplot(1, 3, 3)
        plt.title('Power Spectral Density', fontsize=26)
        plt.plot(xf, psd[:N//2], 'b', lw=6)
        plt.xlabel('Frequency', fontsize=20)
        plt.ylabel('|FFT(D-D_mean)|^2', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()
        plt.show()
    
    return distances.mean(), total_power

def mask_metrics(mask):
    """
    Compute the base metrics of mask

    Parameters
    ----------
    mask : tensor
        mask tensor , (B, C, H, W)

    Returns
    -------
    to do 
    """
    
    mask_i = np.array(mask)
    
    ## SIZE - Value of 1s in the image, normalize to the resolution
    tumor = mask_i[mask_i>0.5]
    tumor_size = len(tumor) / (mask_i.shape[0]*mask_i.shape[1])

    ## SHAPE CONTEXT DESCRIPTORS
    mean, power = fft_descriptor(mask_i, n_points=100, show_flot=True)

    ## CENTROID - Compute the centroid of the tumor
    tumor_indices = np.argwhere(mask_i > 0.5)  # Get indices of tumor voxels
    if len(tumor_indices) > 0:
        centroid = np.mean(tumor_indices, axis=0)  # Compute mean along each axis
    else:
        print("No tumor detected in this mask.")
        return tumor_size, None, None, mean, power

    return tumor_size, centroid[0], centroid[1], mean, power
      
def analyze_tumor_from_dataloader(dataloader):
    """
    Compute the size of the tumor from the dataloader
    """
    tumpr_size_list, centroid_x_list, centroid_y_list = [], [], []
    mean_distances_list, power_list = [], []
    for mask_idx in tqdm(dataloader):
        for i in range(mask_idx.shape[0]): 
            tumor_size, centroid_x, centroid_y, mean_distances, power = mask_metrics(mask_idx[i,0,:,:])
            tumpr_size_list.append(tumor_size)
            centroid_x_list.append(centroid_x)
            centroid_y_list.append(centroid_y)
            mean_distances_list.append(mean_distances)
            power_list.append(power)
    
    return tumpr_size_list, centroid_x_list, centroid_y_list, mean_distances_list, power_list

def infer(par_dir, conf, trial, experiment, epoch, guide_w, compute_real, compute_gen, show_gen_mask):
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


    # Train dataset - mask
    data_img = IntraoperativeUS_mask(size= [dataset_config['im_size_h'], dataset_config['im_size_w']],
                            dataset_path= dataset_config['dataset_path'],
                            im_channels= dataset_config['im_channels'], 
                            splitting_json=dataset_config['splitting_json'],
                            split='train',
                            splitting_seed=dataset_config['splitting_seed'],
                            train_percentage=dataset_config['train_percentage'],
                            val_percentage=dataset_config['val_percentage'],
                            test_percentage=dataset_config['test_percentage'],
                            data_augmentation=False)
    logging.info(f'len train data {len(data_img)}')
    data_loader = DataLoader(data_img, batch_size=1, shuffle=False, num_workers=8)

    # To Do: create the dataset and the dataloader of the generated images
    generated_mask_dir = os.path.join(par_dir, trial, experiment, f'w_{guide_w}', f"samples_ep_{epoch}")
    data_gen = GeneratedMaskDataset(par_dir = generated_mask_dir, size=[dataset_config['im_size_h'], dataset_config['im_size_w']], input_channels=dataset_config['im_channels'])
    data_loader_gen = DataLoader(data_gen, batch_size=train_config['ldm_batch_size_sample'], shuffle=False, num_workers=8)
    logging.info(f'len of the dataset: {len(data_gen)}')

    tumor_size_real, cent_x_real, centr_y_real, mean_dist_real, power_real = analyze_tumor_from_dataloader(data_loader)
    tumor_size_gen, cent_x_gen, centr_y_gen, mean_dist_gen, power_gen = analyze_tumor_from_dataloader(data_loader_gen)

    real_masks, gen_masks = [], []
    for i, j, z in zip(tumor_size_real, mean_dist_real, power_real):  
        if j is not None and z is not None and i is not None:
            real_masks.append([i, j, z])

    for i, j, z in zip(tumor_size_gen, mean_dist_gen, power_gen):
        if j is not None and z is not None and i is not None:
            gen_masks.append([i, j, z])

    real_masks = np.array(real_masks)
    gen_masks = np.array(gen_masks)

    # real_masks_scaled = scaler.fit_transform(real_masks)
    # gen_masks_scaled = scaler.fit_transform(gen_masks)

    ######################## PLOT #################################################################

    ## plt the 2d scatter plot of tsne
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10), num=f'MASK plos', tight_layout=True)   
    ax[0].scatter(real_masks[:,0], real_masks[:,1], c='blue', label='Train data', s=100)
    ax[0].scatter(gen_masks[:,0], gen_masks[:,1], c='lightgreen', label='Gen data', s=100)
    ax[0].set_xlabel('Tumor size', fontsize=30)
    ax[0].set_ylabel('Mean D from centroid', fontsize=30)
    ## set the font of ticks
    ax[0].tick_params(axis='both', which='major', labelsize=30)
    ax[0].grid(linestyle=':')

    ax[1].scatter(real_masks[:,0], real_masks[:,2], c='blue', label='Train data', s=100)
    ax[1].scatter(gen_masks[:,0], gen_masks[:,2], c='lightgreen', label='Gen data', s=100)
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Tumor size', fontsize=30)
    ax[1].set_ylabel('E(PSD)', fontsize=30)
    ## set the font of ticks
    ax[1].tick_params(axis='both', which='major', labelsize=30)
    ax[1].grid(linestyle=':')
    # plt.legend(fontsize=34)
    plt.show()


    #################################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Invastigate the latent space')
    parser.add_argument('--save_folder', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model",
                                                   help='folder to save the model')
    parser.add_argument('--type_image', type=str, default='mask', help='type of image to evaluate, ius or mask')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the 
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--guide_w', type=float, default=0.0, help='guide_w for the conditional model, w=-1 [unconditional], w=0 [vanilla conditioning], w>0 [guided conditional]')
    parser.add_argument('--epoch', type=int, default=99, help='epoch to sample, this is the epoch of cond ldm model') 
    parser.add_argument('--compute_encodings_real', action='store_true', help="compite the embeddings, default=False")
    parser.add_argument('--compute_encodings_gen', action='store_true', help="compite the embeddings, default=False")
    parser.add_argument('--show_gen_mask', action='store_true', help="show the generative and mask images, default=False")
    parser.add_argument('--log', type=str, default='debug', help='Logging level')
    args = parser.parse_args()

    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    experiment_dir = os.path.join(args.save_folder, args.type_image, args.trial)
    if 'vae' in os.listdir(experiment_dir): config = os.path.join(experiment_dir, 'vae', 'config.yaml')
   
    infer(par_dir = os.path.join(args.save_folder, args.type_image), conf=config, trial=args.trial, 
         experiment=args.experiment, epoch=args.epoch, guide_w=args.guide_w, compute_real=args.compute_encodings_real, compute_gen=args.compute_encodings_gen,
         show_gen_mask=args.show_gen_mask)
    plt.show()
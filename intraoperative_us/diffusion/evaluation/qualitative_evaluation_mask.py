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
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.ticker as mticker

from intraoperative_us.diffusion.evaluation.investigate_vae import get_config_value
from intraoperative_us.diffusion.dataset.dataset import IntraoperativeUS_mask, GeneratedMaskDataset
from intraoperative_us.diffusion.models.vae import VAE

from scipy.fft import fft, fftfreq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def linear_fit_with_ci(x, y, label):
    """
    Compute the linear fit with confidence interval
    """
    # Compute linear fit
    coefficients, residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
    slope, intercept = coefficients

    # Compute R-squared
    pred_y = slope * x + intercept
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - pred_y) ** 2)
    r_squared = 1 - (ss_residual / ss_total)

    # Compute standard error and confidence interval
    n = len(x)
    pred_y = slope * x + intercept
    residual_std_error = np.sqrt(residuals[0] / (n - 2)) if residuals.size > 0 else 0
    x_mean = np.mean(x)
    se_slope = residual_std_error / np.sqrt(np.sum((x - x_mean) ** 2))
    se_intercept = residual_std_error * np.sqrt(1/n + x_mean**2 / np.sum((x - x_mean) ** 2))

    t_value = stats.t.ppf(0.975, df=n-2)  # 95% confidence interval
    t_ci_slope = t_value * se_slope
    t_ci_intercept = t_value * se_intercept

    # Statistical test for slope (H0: slope = 0)
    t_stat_slope = slope / se_slope
    p_value_slope = 2 * (1 - stats.t.cdf(abs(t_stat_slope), df=n-2))

    # Print equation and confidence intervals
    logging.info(f'{label} Linear Equation: y = {slope:.4f}x + {intercept:.4f} -- R^2: {r_squared:.4f}')
    logging.info(f'{label} 95% Confidence Interval for Slope: ({slope - t_ci_slope:.4f}, {slope + t_ci_slope:.4f})')
    logging.info(f'{label} 95% Confidence Interval for Intercept: ({intercept - t_ci_intercept:.4f}, {intercept + t_ci_intercept:.4f})')
    logging.info(f'{label} Slope p-value: {p_value_slope:.4f}')

    return slope, t_ci_slope, intercept, t_ci_intercept, r_squared, se_slope, se_intercept

def fft_descriptor(mask, n_points=100, show_plot=False):
    """
    Compute the shape context descriptors of the mask
    """
    # def create_circle_mask(shape=(256, 256), center=None, radius=50):
    #     """
    #     Create a binary mask with a white filled circle on a black background.

    #     Parameters:
    #         shape (tuple): Size of the mask (height, width).
    #         center (tuple): Center of the circle (x, y). Defaults to image center.
    #         radius (int): Radius of the circle.

    #     Returns:
    #         mask (ndarray): Binary mask with a white circle.
    #     """
    #     mask = np.zeros(shape, dtype=np.uint8)

    #     if center is None:
    #         center = (shape[1] // 2, shape[0] // 2)  # (x, y)

    #     cv2.circle(mask, center, radius, color=255, thickness=-1)  # filled circle

    #     return mask

    # # Example usage:
    # mask = create_circle_mask(shape=(256, 256), radius=30)
    # mask = np.array(mask, dtype=np.float32)  # Convert to float32 for consistency

    ## find the contours of the mask
    # Ensure binary mask is in the correct format (H, W) and uint8
    binary_mask = (mask > 0.5).astype(np.uint8)  # Convert to binary (0 or 1)
    binary_mask *= 255                           # Scale to 0-255 (needed for OpenCV)
    mask_size = np.sum(mask > 0.5)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:  # If contours list is empty
        print("Warning: No contours found in the mask.")
        return None, None  # Or return default values

    contour_points = np.vstack([c.squeeze() for c in contours if c.squeeze().ndim == 2])
    
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
    yf = fft((distances - np.mean(distances)))
    xf = fftfreq(N, 1/N)[:N//2]

    # Compute Power Spectral Density (PSD)
    psd = (np.abs(yf) ** 2) / (N/2)  # Normalized power
    total_power = np.sum(psd)

    if show_plot:

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 8), tight_layout=True)
        ax[0].imshow(mask, cmap='gray', alpha=0.8)
        for i in range(len(sampled_points)):
            if i == 0:
                ax[0].scatter(sampled_points[i,0], sampled_points[i,1], c='C3', s=50, label='contour points')
            else:
                ax[0].scatter(sampled_points[i,0], sampled_points[i,1], c='C3', s=50)
        ax[0].scatter(mean_x, mean_y, c='royalblue', s=800, marker='*', label='centroid')
        ax[0].set_title('Mask with contour points and centroid', fontsize=28)
        ax[0].legend(fontsize=28)
        ax[0].axis('off')
        
        
        ax[1].set_title('Distance from centroid', fontsize=26)
        ax[1].plot(distances , 'C3', lw=6)
        ax[1].set_xlabel('contourn points', fontsize=28)
        ax[1].set_ylabel('Distance from centroid '+ r'$(d)$', fontsize=28)
        ax[1].axhline(y=np.mean(distances), color='C3', linestyle='--', lw=4, label='mean '+r'$(\bar{d})$')
        ax[1].set_ylim(0,60)
        ax[1].tick_params(axis='x', labelsize=24)
        ax[1].tick_params(axis='y', labelsize=24)
        ax[1].legend(fontsize=26)
        ax[1].grid(linestyle=':')

        ax[2].set_title('Power Spectral Density', fontsize=28)
        ax[2].plot(xf, psd[:N//2], 'royalblue', lw=6)
        ax[2].set_xlabel('Frequency (f)', fontsize=26)
        ax[2].set_ylabel(r'$\mathcal{F}_{f}$'+' '+r'$(|d - \bar{d}|)$', fontsize=28)
        # fil the area under the curve with a gradient
        ax[2].fill_between(xf, psd[:N//2], color='royalblue', alpha=0.3, label='ESD')
        ax[2].tick_params(axis='x', labelsize=24)
        ax[2].tick_params(axis='y', labelsize=24)
        ax[2].legend(fontsize=26)
        ax[2].grid(linestyle=':')
        plt.show()

    return distances.mean(), total_power

def mask_metrics(mask, n_points, show_plot):
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
    mean, power = fft_descriptor(mask_i, n_points=n_points, show_plot=show_plot)

    ## CENTROID - Compute the centroid of the tumor
    tumor_indices = np.argwhere(mask_i > 0.5)  # Get indices of tumor voxels
    if len(tumor_indices) > 0:
        centroid = np.mean(tumor_indices, axis=0)  # Compute mean along each axis
    else:
        print("No tumor detected in this mask.")
        return tumor_size, None, None, mean, power

    return tumor_size, centroid[0], centroid[1], mean, power

def analyze_tumor_from_dataloader(dataloader, n_points, show_plot):
    """
    Compute the size of the tumor from the dataloader
    """
    tumpr_size_list, centroid_x_list, centroid_y_list = [], [], []
    mean_distances_list, power_list = [], []
    for mask_idx in tqdm(dataloader):
        for i in range(mask_idx.shape[0]):
            tumor_size, centroid_x, centroid_y, mean_distances, power = mask_metrics(mask_idx[i,0,:,:], n_points, show_plot=show_plot)
            tumpr_size_list.append(tumor_size)
            centroid_x_list.append(centroid_x)
            centroid_y_list.append(centroid_y)
            mean_distances_list.append(mean_distances)
            power_list.append(power)

    return tumpr_size_list, centroid_x_list, centroid_y_list, mean_distances_list, power_list

def infer(par_dir, conf, trial, experiment, epoch, guide_w, scheduler_type, n_points, show_gen_mask):
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


    # Train dataset - REAL TUMOR MASK
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

    tumor_size_real, cent_x_real, centr_y_real, mean_dist_real, power_real = analyze_tumor_from_dataloader(data_loader, n_points, show_plot=show_gen_mask)
    real_masks = []
    for i, j, z in zip(tumor_size_real, mean_dist_real, power_real):
        if j is not None and z is not None and i is not None:
            real_masks.append([i, j, z])
    real_masks = np.array(real_masks)
    slope_real, t_ci_slope_real, intercept_real, t_ci_intercept_real,  r_squared_real, se_slope_real, se_intercept_real = linear_fit_with_ci(np.log(real_masks[:,0]), np.log(real_masks[:,2]), 'Train data')
    logging.info(f'len real data {len(real_masks)}')
    logging.info(f'Linear fit for REAL DATA: slope={slope_real:.4f}, intercept={intercept_real:.4f}, R^2={r_squared_real:.4f}')
    logging.info(f'95% CI for slope: ({slope_real - t_ci_slope_real:.4f}, {slope_real + t_ci_slope_real:.4f})')
    logging.info(f'95% CI for intercept: ({intercept_real - t_ci_intercept_real:.4f}, {intercept_real + t_ci_intercept_real:.4f})')


    ## ANALYSIS OVER EPOCHS - GENERATED TUMOR MASK
    epochs_dict = {}
    for ep in [500, 1000, 1500, 2000, 2500, 3000]:
        ep_dict = {}
        ## load data
        generated_mask_dir = os.path.join(par_dir, trial, experiment, f'w_{guide_w}', scheduler_type, f"samples_ep_{ep}")
        data_gen = GeneratedMaskDataset(par_dir = generated_mask_dir, size=[dataset_config['im_size_h'], dataset_config['im_size_w']], input_channels=dataset_config['im_channels'])
        data_loader_gen = DataLoader(data_gen, batch_size=train_config['ldm_batch_size_sample'], shuffle=False, num_workers=8)

        tumor_size_gen, cent_x_gen, centr_y_gen, mean_dist_gen, power_gen = analyze_tumor_from_dataloader(data_loader_gen, n_points, show_plot=show_gen_mask)

        gen_masks = []
        for i, j, z in zip(tumor_size_gen, mean_dist_gen, power_gen):
            if j is not None and z is not None and i is not None:
                gen_masks.append([i, j, z])
        gen_masks = gen_masks[:len(data_img)]
        logging.info(f'len gen data {len(gen_masks)}')
        ep_dict['gen_mask'] = gen_masks

        gen_masks = np.array(gen_masks)
        slope_gen, t_ci_slope_gen, intercept_gen, t_ci_intercept_gen, r_squared_gen, se_slope_gen, se_intercept_gen = linear_fit_with_ci(np.log(gen_masks[:,0]), np.log(gen_masks[:,2]), 'Gen data')
        ep_dict['linear_fit'] = [slope_gen, t_ci_slope_gen, intercept_gen, t_ci_intercept_gen, r_squared_gen]
        epochs_dict[ep] = ep_dict

        logging.info('Statistical t test: H0: slope_gen = slope_real')
        t_stat_slope = (slope_gen - slope_real) / np.sqrt(se_slope_gen**2 + se_slope_real**2)
        p_value_slope = 2 * (1 - stats.t.cdf(abs(t_stat_slope), df=len(gen_masks)+len(real_masks)-4))
        logging.info(f'Slope t-statistic {trial}-{ep}: {t_stat_slope:.4f}, p-value: {p_value_slope:.4f}')

    ## Quantitative analisys
    keys_list = list(epochs_dict.keys())

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10), num=f'Quantitative analysis', tight_layout=True)
    ax[0].axhline(y=slope_real, color='orchid', linestyle='-', lw=3, label='Train data')
    ax[0].fill_between([keys_list[0], keys_list[-1]], slope_real - t_ci_slope_real, slope_real + t_ci_slope_real, color='orchid', alpha=0.1)
    ax[0].set_xlabel('Epochs', fontsize=30)
    ax[0].set_ylabel('Slope', fontsize=30)
    ax[0].tick_params(axis='both', which='major', labelsize=30)
    ax[0].grid(linestyle=':')

    ax[1].axhline(y=intercept_real, color='orchid', linestyle='-', lw=3, label='Train data')
    ax[1].fill_between([keys_list[0], keys_list[-1]], intercept_real - t_ci_intercept_real, intercept_real + t_ci_intercept_real, color='orchid', alpha=0.1)
    ax[1].set_xlabel('Epochs', fontsize=30)
    ax[1].set_ylabel('Intercept', fontsize=30)
    ax[1].tick_params(axis='both', which='major', labelsize=30)
    ax[1].legend(fontsize=30)
    ax[1].grid(linestyle=':')

    slope_list = [epochs_dict[ep]['linear_fit'][0] for ep in epochs_dict.keys()]
    intercept_list = [epochs_dict[ep]['linear_fit'][2] for ep in epochs_dict.keys()]
    slope_ci_list = [epochs_dict[ep]['linear_fit'][1] for ep in epochs_dict.keys()]
    intercept_ci_list = [epochs_dict[ep]['linear_fit'][3] for ep in epochs_dict.keys()]
    ax[0].plot(keys_list, slope_list, c='forestgreen', lw=3, ls='--', marker='o', ms=30, label='Gen data')
    ax[0].fill_between(keys_list, np.array(slope_list) - np.array(slope_ci_list), np.array(slope_list) + np.array(slope_ci_list), color='forestgreen', alpha=0.1)
    ax[0].legend(fontsize=30)
    ax[1].plot(keys_list, intercept_list, c='forestgreen', lw=3, ls='--',  marker='o', ms=30, label='Gen data')
    ax[1].fill_between(keys_list, np.array(intercept_list) - np.array(intercept_ci_list), np.array(intercept_list) + np.array(intercept_ci_list), color='forestgreen', alpha=0.1)
    ax[1].legend(fontsize=30)
    ######################## PLOT #################################################################
    gen_masks = np.array(epochs_dict[epoch]['gen_mask'])
    
    ## plt the 2d scatter plot of tsne
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(24,8), num=f'MASK plots {epoch} - {trial}', tight_layout=True)
    bins = np.histogram_bin_edges(np.concatenate([real_masks[:,0], gen_masks[:,0]]), bins='auto')
    stat_m, p_m = stats.ranksums(real_masks[:,0], gen_masks[:,0])
    ## print the logging in the legend
    ax[0].hist(real_masks[:,0], bins=bins, color='orchid', label=f'Real data', alpha=0.5)
    ax[0].hist(gen_masks[:,0],  bins=bins, color='forestgreen', label='Gen data', alpha=0.5)
    ax[0].set_xlabel('Tumour size', fontsize=30)
    ax[0].set_ylabel('Count', fontsize=30)
    ax[0].tick_params(axis='both', which='major', labelsize=30)
    ax[0].legend(fontsize=24)
    ax[0].grid(linestyle=':')

    bins = np.histogram_bin_edges(np.concatenate([real_masks[:,2], gen_masks[:,2]]), bins='auto')
    stat_e, p_e = stats.ranksums(real_masks[:,2], gen_masks[:,2])
    ## print the logging in the legend
    ax[1].hist(real_masks[:,2], bins=bins, color='orchid', label='Real', alpha=0.5)
    ax[1].hist(gen_masks[:,2],  bins=bins, color='forestgreen', label='Gen data', alpha=0.5)
    ax[1].set_xlabel('ESD', fontsize=30)
    ax[1].set_ylabel('Count', fontsize=30)
    ax[1].tick_params(axis='both', which='major', labelsize=30)
    ax[1].legend(fontsize=24)
    ax[1].grid(linestyle=':')

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0)) 
    ax[1].xaxis.set_major_formatter(formatter)
    ax[1].xaxis.get_offset_text().set_fontsize(30)
    ax[1].tick_params(axis='x', which='major', labelsize=30)

    ax[2].scatter(real_masks[:,0], real_masks[:,2], c='orchid', label='Train data', s=100, alpha=0.4)
    ax[2].scatter(gen_masks[:,0], gen_masks[:,2], c='forestgreen', label='Gen data', s=100, alpha=0.4)
    ax[2].plot(real_masks[:,0], np.exp(slope_real * np.log(real_masks[:,0]) + intercept_real), c='orchid', ls='-',  lw=3, label='Fit train data')
    ax[2].plot(gen_masks[:,0], np.exp(slope_gen * np.log(gen_masks[:,0]) + intercept_gen), c='forestgreen', ls='-', lw=3, label='Fit gen data')
    ax[2].set_yscale('log')
    ax[2].set_xscale('log')
    ax[2].set_xlabel('Tumour size', fontsize=30)
    ax[2].set_ylabel('ESD', fontsize=30)
    ax[2].tick_params(axis='both', which='major', labelsize=30)
    ax[2].legend(fontsize=24)
    ax[2].grid(linestyle=':')

    ## Print statistical distribution
    print('=========================================================================')
    print(trial)
    print('TUMOUR SIZE')
    print(f'REAL DATA: mean={np.mean(real_masks[:,0]):.4f}, std={np.std(real_masks[:,0]):.4f}, min={np.min(real_masks[:,0]):.4f}, max={np.max(real_masks[:,0]):.4f}')
    print(f'         : median={np.median(real_masks[:,0]):.4f}, 1-quantile={np.quantile(real_masks[:,0], 0.25):.4f}, 3-quantile={np.quantile(real_masks[:,0], 0.75):.4f}')
    print(f'GEN DATA: mean={np.mean(gen_masks[:,0]):.4f}, std={np.std(gen_masks[:,0]):.4f}, min={np.min(gen_masks[:,0]):.4f}, max={np.max(gen_masks[:,0]):.4f}')
    print(f'         : median={np.median(gen_masks[:,0]):.4f}, 1-quantile={np.quantile(gen_masks[:,0], 0.25):.4f}, 3-quantile={np.quantile(gen_masks[:,0], 0.75):.4f}')
    print(f'Wilcoxon test: statistic={stat_m:.4f}, p-value={p_m:.4f}')
    print()
    print('ESD')
    print(f'REAL DATA: mean={np.mean(real_masks[:,2]):.4f}, std={np.std(real_masks[:,2]):.4f}, min={np.min(real_masks[:,2]):.4f}, max={np.max(real_masks[:,2]):.4f}')
    print(f'         : median={np.median(real_masks[:,2]):.4f}, 1-quantile={np.quantile(real_masks[:,2], 0.25):.4f}, 3-quantile={np.quantile(real_masks[:,2], 0.75):.4f}')
    print(f'GEN DATA: mean={np.mean(gen_masks[:,2]):.4f}, std={np.std(gen_masks[:,2]):.4f}, min={np.min(gen_masks[:,2]):.4f}, max={np.max(gen_masks[:,2]):.4f}')
    print(f'         : median={np.median(gen_masks[:,2]):.4f}, 1-quantile={np.quantile(gen_masks[:,2], 0.25):.4f}, 3-quantile={np.quantile(gen_masks[:,2], 0.75):.4f}') 
    print(f'Wilcoxon test: statistic={stat_e:.4f}, p-value={p_e:.4f}')
    print('=========================================================================')
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
    parser.add_argument('--scheduler_type', type=str, default='ddpm', help='scheduler for the diffusion model')
    parser.add_argument('--epoch', type=int, default=99, help='epoch to sample, this is the epoch of cond ldm model')
    parser.add_argument('--n_points', type=int, default=100, help='number of points to sample the mask')
    parser.add_argument('--show_gen_mask', action='store_true', help="show the generative and mask images, default=False")
    parser.add_argument('--log', type=str, default='debug', help='Logging level')
    args = parser.parse_args()

    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    experiment_dir = os.path.join(args.save_folder, args.type_image, args.trial)
    if 'vae' in os.listdir(experiment_dir): config = os.path.join(experiment_dir, 'vae', 'config.yaml')

    infer(par_dir = os.path.join(args.save_folder, args.type_image), conf=config, trial=args.trial,
         experiment=args.experiment, epoch=args.epoch, guide_w=args.guide_w, scheduler_type=args.scheduler_type, n_points=args.n_points,
         show_gen_mask=args.show_gen_mask)
    plt.show()
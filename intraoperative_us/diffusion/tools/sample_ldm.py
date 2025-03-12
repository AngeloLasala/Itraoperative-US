"""
Sample new data from the trained LDM-VAE model trained with hugginface
"""
import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from PIL import Image
import cv2
import random
import logging
import numpy as np
from tqdm import tqdm

from diffusers import DDIMScheduler, PNDMScheduler,  DDPMScheduler
from diffusers import UNet2DModel
from accelerate import Accelerator
import torch.nn.functional as F
from intraoperative_us.diffusion.models.unet_cond_base import get_config_value, UNet2DConditionModelCostum

from intraoperative_us.diffusion.models.unet_base import Unet
from intraoperative_us.diffusion.scheduler.scheduler import LinearNoiseScheduler
from intraoperative_us.diffusion.dataset.dataset import IntraoperativeUS, IntraoperativeUS_mask
from intraoperative_us.diffusion.models.vae import VAE
from intraoperative_us.diffusion.utils.utils import get_best_model, load_autoencoder, get_number_parameter
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, diffusion_model_config, sampling_config,
           autoencoder_model_config, diffusion_config, dataset_config, vae, save_folder):
    """
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    im_size_h = dataset_config['im_size_h'] // 2**sum(autoencoder_model_config['down_sample'])
    im_size_w = dataset_config['im_size_w'] // 2**sum(autoencoder_model_config['down_sample'])
    logging.info(f'Resolution of latent space [{im_size_h},{im_size_w}]')

    ##set random seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    N_loop = int(sampling_config['N_gen'] / sampling_config['sampling_batch'])
    for btc, times_ii in enumerate(range(N_loop)):
        xt = torch.randn((sampling_config['sampling_batch'],
                        autoencoder_model_config['z_channels'],
                        im_size_h,
                        im_size_w)).to(device)

        # scheduler.set_timesteps(50) ## this for the pndm scheduler
        for t in tqdm(scheduler.timesteps):
            xt = scheduler.scale_model_input(xt, timestep=t)

            with torch.no_grad():
                # Get prediction of noise
                noise_pred = model(xt, t).sample
            # Use scheduler to get x0 and xt-1
            # xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
            xt = scheduler.step(noise_pred, t, xt).prev_sample #return_dict=False)[0]
            
        xt = xt * (1 / vae.config.scaling_factor)
        with torch.no_grad():
            ims = vae.decode(xt).sample
            
        ims = torch.clamp(ims, 0., 1.).detach().cpu()
        
        for i in range(xt.shape[0]):
            cv2.imwrite(os.path.join(save_folder, f"x0_{btc * sampling_config['sampling_batch'] + i}.png"), ims[i].numpy()[0]*255)

def infer(par_dir, conf, trial, experiment, epoch, type_image):
    # Read the config file #
    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    train_config = config['train_params']
    sampling_config = config['sampling_params']
    
    ########## Create the noise scheduler #############
    if diffusion_config['scheduler'] == 'linear':
        logging.info('Linear scheduler')
        scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                         beta_start=diffusion_config['beta_start'],
                                         beta_end=diffusion_config['beta_end'])
    elif diffusion_config['scheduler'] == 'ddim':
        logging.info(f"{diffusion_config['scheduler']} scheduler")
        scheduler = DDIMScheduler.from_pretrained(os.path.join(diffusion_config['scheduler_path'], diffusion_config['scheduler']),
                                                  prediction_type=diffusion_config['prediction_type'])

    elif diffusion_config['scheduler'] == 'pndm':
        logging.info(f"{diffusion_config['scheduler']} scheduler")
        scheduler = PNDMScheduler.from_pretrained(os.path.join(diffusion_config['scheduler_path'], diffusion_config['scheduler']),
                                                  prediction_type=diffusion_config['prediction_type'])

    elif diffusion_config['scheduler'] == 'ddpm':
        logging.info(f"{diffusion_config['scheduler']} scheduler")
        scheduler = DDPMScheduler(num_train_timesteps=diffusion_config['num_train_timesteps'])

    else:
        raise ValueError(f"Scheduler {diffusion_config['scheduler']} not implemented")
    ####################################################

    # Load the trained models    
    trial_folder = os.path.join(par_dir, type_image, trial)
    assert os.listdir(trial_folder), f'No trained model found in trial folder {trial_folder}'
    logging.info(f'Load trained model from {trial_folder}')
    model_dir = os.path.join(par_dir, type_image, trial, experiment)
    
    if 'vae' in os.listdir(trial_folder):
        logging.info(f'Load trained {os.listdir(trial_folder)[0]} model')
        with open(os.path.join(trial_folder, 'vae', 'config.yaml'), 'r') as f:
            autoencoder_model_config = yaml.safe_load(f)['autoencoder_params']

        best_model = get_best_model(os.path.join(trial_folder,'vae'))
        logging.info(f'best model  epoch {best_model}')
        vae = load_autoencoder(autoencoder_model_config, dataset_config, device)
        vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'vae', f'vae_best_{best_model}.pth'), map_location=device))

        ## unconditional model
        model = UNet2DModel(sample_size=diffusion_model_config['sample_size'],
                            in_channels=diffusion_model_config['z_channels'],
                            out_channels=diffusion_model_config['z_channels'],
                            block_out_channels=diffusion_model_config['down_channels']).to(device)
        model.eval()
        model.load_state_dict(torch.load(os.path.join(model_dir, f'ldm_{epoch}.pth'),map_location=device), strict=False)


    # Create output directories
    save_folder = os.path.join(model_dir, f'w_-1.0', f'samples_ep_{epoch}')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        overwrite = input("The save folder already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != 'y':
            print("Training aborted.")
            exit()


    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config, sampling_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae, save_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train unconditional LDM with VQVAE')
    parser.add_argument('--save_folder', type=str, default='trained_model', help='folder to save the model, default = trained_model')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the 
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--type_image', type=str, default='mask', help='type of image to sample, it can be ius or celebhq')
    parser.add_argument('--epoch', type=int, default=100, help='epoch to sample, this is the epoch of cond ldm model')
    parser.add_argument('--log', type=str, default='info', help='Logging level')
    args = parser.parse_args()

    print('Am I using GPU: ', torch.cuda.is_available())

    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    experiment_dir = os.path.join(args.save_folder, args.type_image, args.trial, args.experiment)
    config = os.path.join(experiment_dir, 'config.yaml')

    # save_folder = os.path.join(par_dir, 'trained_model', args.trial)
    infer(par_dir = args.save_folder, conf=config, trial=args.trial, experiment=args.experiment, epoch=args.epoch, type_image=args.type_image)
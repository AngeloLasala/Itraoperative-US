"""
Sample from trained conditional latent diffusion of trained model with hugginface.
Fhe sampling follow the classifier-free guidance, and self perceptual loss (to do!!)

"""
import numpy as np
import torch
import random
import torchvision
import argparse
import yaml
import os
import logging

from diffusers import DDIMScheduler, PNDMScheduler, UniPCMultistepScheduler, DDPMScheduler, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel
from accelerate import Accelerator
import torch.nn.functional as F

from intraoperative_us.diffusion.models.unet_cond_base import get_config_value, UNet2DConditionModelCostum
from intraoperative_us.diffusion.models.vqvae import VQVAE
from intraoperative_us.diffusion.models.vae import VAE 
import intraoperative_us.diffusion.models.unet_cond_base as unet_cond_base
import intraoperative_us.diffusion.models.unet_base as unet_base
from intraoperative_us.diffusion.scheduler.scheduler import LinearNoiseScheduler
from intraoperative_us.diffusion.dataset.dataset import IntraoperativeUS, GeneratedMaskDataset
from intraoperative_us.diffusion.utils.utils import get_best_model, load_autoencoder, get_number_parameter
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import cv2

import tqdm as tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def sample(model, scheduler, train_config, diffusion_model_config, condition_config, generated_mask_dir, tokenizer, text_encoder, 
           autoencoder_model_config, diffusion_config, dataset_config, type_model, vae, save_folder, mask_folder, ius_folder,
           guide_w):
    """
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    im_size_h = dataset_config['im_size_h'] // 2**sum(autoencoder_model_config['down_sample'])
    im_size_w = dataset_config['im_size_w'] // 2**sum(autoencoder_model_config['down_sample'])
    logging.info(f'RESOLUTION OF THE LATENT SPACE [{im_size_h},{im_size_w}]')

    # Get the spatial conditional mask, i.e. the heatmaps
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']
    logging.info(f"DIMENSION OF THE LATENT SPACE: {autoencoder_model_config['z_channels']}")

    prompt = diffusion_model_config['prompt']
    logging.info(f'Prompt: {prompt}')
    def tokenize_captions(prompt, current_batch_size):
        captions = [prompt] * current_batch_size
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    data_img = GeneratedMaskDataset(par_dir = generated_mask_dir, size=[dataset_config['im_size_h'], dataset_config['im_size_w']], input_channels=dataset_config['im_channels'])
    data_loader = DataLoader(data_img, batch_size=train_config['ldm_batch_size_sample'], shuffle=False, num_workers=8)
    logging.info(f'len of the dataset: {len(data_img)}')

    for btc, data in enumerate(data_loader):
        cond_input = {}
        uncond_input = {}
        if condition_config is not None:
            # im is the image (batch_size=8), cond_input is the conditional input ['image for the mask']
            for key in condition_config['condition_types']: ## for all the type of condition, we move the  tensor on the device
                cond_input[key] = data.to(device)
                uncond_input[key] = torch.zeros_like(cond_input[key])
        else:
            pass
    
        xt = torch.randn((cond_input[key].shape[0],
                      autoencoder_model_config['z_channels'],
                      im_size_h,
                      im_size_w)).to(device)
        test_tokenized_captions = tokenize_captions(prompt, xt.shape[0]).to(device)
        with torch.no_grad():
            text_embeddings = text_encoder(test_tokenized_captions)[0]
        
        ################# Sampling Loop ########################
        scheduler.set_timesteps(diffusion_config['num_sample_timesteps'])
        for t in tqdm.tqdm(scheduler.timesteps):
            xt = scheduler.scale_model_input(xt, timestep=t)
            
            with torch.no_grad():
                noise_pred_cond = model(xt, t, text_embeddings, cond_input)
                noise_pred_uncond = model(xt, t, text_embeddings, uncond_input)            # ## get the noise prediction for the conditional and unconditional model

            ## sampling the noise for the conditional and unconditional model
            noise_pred = (1 + guide_w) * noise_pred_cond - guide_w * noise_pred_uncond
     
            # Use scheduler to get x0 and xt-1
            # xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
            xt = scheduler.step(noise_pred, t, xt).prev_sample
                    
        xt = xt * (1 / vae.config.scaling_factor)
        with torch.no_grad():
            if type_model == 'vae':
                ims = vae.decode(xt).sample
        
        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        mask = cond_input['image'].detach().cpu()

        for i in range(ims.shape[0]):
            cv2.imwrite(os.path.join(ius_folder, f'x0_{btc * train_config["ldm_batch_size_sample"] + i}.png'), ims[i].numpy()[0]*255)
            cv2.imwrite(os.path.join(mask_folder, f'mask_{btc * train_config["ldm_batch_size_sample"] + i}.png'), mask[i].numpy()[0]*255)

def infer(par_dir, conf, trial, split, experiment, epoch, guide_w, generated_mask_dir, 
          scheduler='ddpm', num_sample_timesteps=1000):
    """
    Sample function for conditional latent diffusion model. i.e. staccked condition

    Parameters
    ----------
    par_dir : str
        path to the folder where the model is saved
    conf : str
        path to the config file
    trial : str
        trial name for saving the model, it is the trial folder that contain the VAE model
        whith this parameters you change the type of VAE (finetuning, random, lora)
    experiment : str
        name of experiment, it is refed to the type od DM used
    epoch : int
        epoch to sample, this is the epoch of cond ldm model
    guide_w : float
        guide_w for the conditional model, w=-1 [unconditional], w=0 [vanilla conditioning], w>0 [guided conditional]
    generated_mask_dir : str
        folder to save the generated mask   
    scheduler : str (optional)
        scheduler to use for the diffusion process, default is DDPM
    num_samples_step : int (optional)
        number of samples to generate, default is 1000
    """
    # Read the config file #
    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    train_config = config['train_params']
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']

    # Set the desired seed value #
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    #############################
    
    ########## Create the noise scheduler #############

    diffusion_config['scheduler'] = scheduler
    diffusion_config['num_sample_timesteps'] = num_sample_timesteps

    if diffusion_config['scheduler'] == 'linear':
        logging.info('Linear scheduler')
        scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                         beta_start=diffusion_config['beta_start'],
                                         beta_end=diffusion_config['beta_end'])
    elif diffusion_config['scheduler'] == 'ddim':
        logging.info(f"{diffusion_config['scheduler']} scheduler")
        scheduler = DDIMScheduler(beta_start=0.0001,
                                  beta_end=0.02,
                                  beta_schedule='linear',
                                  clip_sample=True,
                                  prediction_type=diffusion_config['prediction_type'])

    elif diffusion_config['scheduler'] == 'pndm':
        logging.info(f"{diffusion_config['scheduler']} scheduler")
        scheduler = PNDMScheduler(beta_start=0.0001,
                                beta_end=0.02,
                                beta_schedule='linear',
                                # clip_sample=True,
                                prediction_type=diffusion_config['prediction_type'])
                                                    
    elif diffusion_config['scheduler'] == 'ddpm':
        logging.info(f"{diffusion_config['scheduler']} scheduler")
        scheduler = DDPMScheduler(num_train_timesteps=diffusion_config['num_train_timesteps'])

    elif diffusion_config['scheduler'] == 'dpm_solver':
        logging.info(f"{diffusion_config['scheduler']} scheduler")
        scheduler = DPMSolverMultistepScheduler(beta_start=0.0001,
                                                beta_end=0.02,
                                                beta_schedule='linear',
                                                timestep_spacing='leading',
                                                prediction_type=diffusion_config['prediction_type'])
    else:
        raise ValueError(f"Scheduler {diffusion_config['scheduler']} not implemented")

    logging.info(scheduler)
    ####################################################

    ############# Load tokenizer and text model #################
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(diffusion_model_config['unet_path'], diffusion_model_config['tokenizer']))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(diffusion_model_config['unet_path'], diffusion_model_config['text_encoder']), use_safetensors=True).to(device)
    ###############################################
    
    ########## Load AUTOENCODER #############
    trial_folder = os.path.join(par_dir, 'ius', trial, split)
    assert os.listdir(trial_folder), f'No trained model found in trial folder {trial_folder}'
    logging.info(os.listdir(trial_folder))
    model_dir = os.path.join(par_dir, 'ius', trial, split, experiment)

   
    if 'vae' in os.listdir(trial_folder):
        ## VAE + conditional LDM
        type_model = 'vae'
        logging.info(f'type model {type_model}')
        logging.info(f'Load trained {os.listdir(trial_folder)[0]} model')
        # read configuration file in this foleder
        with open(os.path.join(trial_folder, 'vae', 'config.yaml'), 'r') as f:
            autoencoder_config = yaml.safe_load(f)['autoencoder_params']

        best_model = get_best_model(os.path.join(trial_folder,'vae'))
        logging.info(f'best model  epoch {best_model}')
        vae = load_autoencoder(autoencoder_config, dataset_config, device)
        vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'vae', f'vae_best_{best_model}.pth'), map_location=device))

        # conditional ldm
        model = UNet2DConditionModelCostum(diffusion_model_config).to(device)
        model.eval()
        model.load_state_dict(torch.load(os.path.join(model_dir, f'ldm_{epoch}.pth'),map_location=device), strict=False)

    #####################################

    ######### Create output directories #############
    save_folder = os.path.join(model_dir, f'w_{guide_w}', diffusion_config['scheduler'], f'samples_ep_{epoch}')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        mask_folder = os.path.join(save_folder, 'masks')
        os.makedirs(mask_folder)
        ius_folder = os.path.join(save_folder, 'ius')
        os.makedirs(ius_folder)
    else:
        overwrite = input(f"The save folder {save_folder} already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != 'y':
            print("Training aborted.")
            exit()
        else:
            mask_folder = os.path.join(save_folder, 'masks')
            ius_folder = os.path.join(save_folder, 'ius')
    
    ######## Sample from the model 
    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config, condition_config, generated_mask_dir, tokenizer, text_encoder,
               autoencoder_config, diffusion_config, dataset_config, type_model, vae, save_folder, mask_folder, ius_folder,
               guide_w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train unconditional LDM with VQVAE')
    parser.add_argument('--save_folder', type=str, default='trained_model', help='folder to save the model, default = trained_model')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--split', type=str, default='split_1', help='splitting name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the 
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--epoch', type=int, default=100, help='epoch to sample, this is the epoch of cond ldm model')
    parser.add_argument('--guide_w', type=float, default=0.0, help='guide_w for the conditional model, w=-1 [unconditional], w=0 [vanilla conditioning], w>0 [guided conditional]')
    parser.add_argument('--generated_mask_dir', type=str, default='generated_mask', help='folder to save the generated mask')
    parser.add_argument('--scheduler', type=str, default='ddpm', help='scheduler to use for the diffusion process, default is DDPM')
    parser.add_argument('--num_sample_timesteps', type=int, default=1000, help='number of samples to generate, default is 1000')
    parser.add_argument('--log', type=str, default='info', help='Logging level')
    args = parser.parse_args()

    print('Am I using GPU: ', torch.cuda.is_available())

    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    experiment_dir = os.path.join(args.save_folder, 'ius', args.trial, args.split, args.experiment)
    config = os.path.join(experiment_dir, 'config.yaml')

    infer(par_dir = args.save_folder, conf=config, trial=args.trial, split = args.split,
         experiment=args.experiment ,epoch=args.epoch, guide_w=args.guide_w, 
         generated_mask_dir=args.generated_mask_dir,
         scheduler=args.scheduler, num_sample_timesteps=args.num_sample_timesteps)
    plt.show()


"""
Sampling from trained one_step generation.
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
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor
from diffusers import UNet2DConditionModel
from accelerate import Accelerator
import torch.nn.functional as F

from intraoperative_us.diffusion.models.unet_cond_base import get_config_value, UNet2DConditionModelCostum
import intraoperative_us.diffusion.models.unet_cond_base as unet_cond_base
import intraoperative_us.diffusion.models.unet_base as unet_base
from intraoperative_us.diffusion.dataset.dataset import IntraoperativeUS, GeneratedMaskDataset
from intraoperative_us.diffusion.utils.utils import get_best_model, load_autoencoder, load_unet_model, get_number_parameter
from torch.utils.data import DataLoader
from peft import LoraConfig
from diffusers.training_utils import cast_training_params

import matplotlib.pyplot as plt
import cv2

import tqdm as tqdm



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def sample(model, scheduler, train_config, diffusion_model_config, condition_config, tokenizer, text_encoder,
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

    def tokenize_captions(current_batch_size):
        captions = [""] * current_batch_size
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    N_gen = 400
    sampling_config = 2 #train_config["ldm_batch_size_sample"]

    N_loop = int(N_gen / sampling_config)
    for btc, times_ii in enumerate(range(N_loop)):
        xt = torch.randn((sampling_config,
                        autoencoder_model_config['z_channels'],
                        im_size_h,
                        im_size_w)).to(device)

        ## add random shift to the latent space
        xt = xt - 0.05
                
        test_tokenized_captions = tokenize_captions(xt.shape[0]).to(device)
        text_embeddings = text_encoder(test_tokenized_captions)[0]         
        ################# Sampling Loop ########################
        
        scheduler.set_timesteps(diffusion_config['num_sample_timesteps'])
        for t in tqdm.tqdm(scheduler.timesteps):
            xt = scheduler.scale_model_input(xt, timestep=t)
            
            noise_pred = model(xt, t, text_embeddings, return_dict=False)[0]
            

            # Use scheduler to get x0 and xt-1
            # xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
            xt = scheduler.step(noise_pred, t, xt).prev_sample
                    
        xt = xt * (1 / vae.config.scaling_factor)
        with torch.no_grad():
            if type_model == 'vae':
                ims = vae.decode(xt).sample
        
        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        image = ims[:, 0, :, :]
        mask = ims[:, 1, :, :]
        
        image = (image + 1) / 2
        mask = (mask > 0.5).float()

        plt.figure()
        plt.imshow(image[0].numpy() * 255, cmap='gray')
        plt.axis('off')
        plt.show()
        
        for i in range(ims.shape[0]):
            cv2.imwrite(os.path.join(ius_folder, f'x0_{btc * sampling_config + i}.png'), image[i].numpy() * 255)
            cv2.imwrite(os.path.join(mask_folder, f'mask_{btc * sampling_config + i}.png'), mask[i].numpy() * 255)

def infer(par_dir, conf, trial, split, experiment, epoch, guide_w, 
          scheduler, num_sample_timesteps):
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
        scheduler = DDIMScheduler.from_pretrained(os.path.join(diffusion_config['scheduler_path'], diffusion_config['scheduler']),
                                                  beta_start=0.0001,
                                                  beta_end=0.02,
                                                  beta_schedule='linear',
                                                  clip_sample=True,
                                                  prediction_type=diffusion_config['prediction_type'])

    elif diffusion_config['scheduler'] == 'pndm':
        logging.info(f"{diffusion_config['scheduler']} scheduler")
        scheduler = PNDMScheduler.from_pretrained(os.path.join(diffusion_config['scheduler_path'], diffusion_config['scheduler']),
                                                  beta_start=0.0001,
                                                  beta_end=0.02,
                                                  beta_schedule='linear',
                                                  clip_sample=True,
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
    trial_folder = os.path.join(par_dir, args.type_image, trial, split)
    assert os.listdir(trial_folder), f'No trained model found in trial folder {trial_folder}'
    logging.info(os.listdir(trial_folder))
    model_dir = os.path.join(par_dir, args.type_image, trial, split, experiment)

   
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
        model = load_unet_model(diffusion_model_config, autoencoder_config, dataset_config, device)
        if  diffusion_model_config['initialization'] == 'lora':
            logging.info('SD1.5 initialization + loRA finetuning')
            unet_lora_config = LoraConfig(
                r=4,
                lora_alpha=4,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            model.add_adapter(unet_lora_config)
        elif diffusion_model_config['initialization'] == 'random':
            logging.info('Random initialization')
        elif diffusion_model_config['initialization'] == 'SD1.5':
            logging.info('SD1.5 initialization + extensive finetuning')
        model.eval()
        model.load_state_dict(torch.load(os.path.join(model_dir, f'ldm_{epoch}.pth'),map_location=device), strict=False)
        get_number_parameter(model)

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
    
    print('OK')
    ######## Sample from the model 
    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config, condition_config, tokenizer, text_encoder,
               autoencoder_config, diffusion_config, dataset_config, type_model, vae, save_folder, mask_folder, ius_folder,
               guide_w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train unconditional LDM with VQVAE')
    parser.add_argument('--save_folder', type=str, default='trained_model', help='folder to save the model, default = trained_model')
    parser.add_argument('--type_image', type=str, default='one_step', help='type of image to use for the training, it can be ius or us')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--split', type=str, default='split_1', help='splitting name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the 
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--epoch', type=int, default=100, help='epoch to sample, this is the epoch of cond ldm model')
    parser.add_argument('--guide_w', type=float, default=0.0, help='guide_w for the conditional model, w=-1 [unconditional], w=0 [vanilla conditioning], w>0 [guided conditional]')
    parser.add_argument('--scheduler', type=str, default='ddpm', help='scheduler to use for the diffusion process, default is DDPM')
    parser.add_argument('--num_sample_timesteps', type=int, default=1000, help='number of samples to generate, default is 1000')
    parser.add_argument('--log', type=str, default='info', help='Logging level')
    args = parser.parse_args()

    print('Am I using GPU: ', torch.cuda.is_available())

    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    experiment_dir = os.path.join(args.save_folder, args.type_image, args.trial, args.split, args.experiment)
    config = os.path.join(experiment_dir, 'config.yaml')

    infer(par_dir = args.save_folder, conf=config, trial=args.trial, split = args.split,
         experiment=args.experiment ,epoch=args.epoch, guide_w=args.guide_w, 
         scheduler=args.scheduler, num_sample_timesteps=args.num_sample_timesteps)   

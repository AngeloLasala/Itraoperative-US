"""
Sample from trained conditional latent diffusion model. the sampling follow the classifier-free guidance

w = -1 [unconditional] = the learned conditional model completely ignores the conditioner and learns an unconditional diffusion model
w = 0 [vanilla conditional] =  the model explicitly learns the vanilla conditional distribution without guidance
w > 0 [guided conditional] =  the diffusion model not only prioritizes the conditional score function, but also moves in the direction away from the unconditional score function
"""
import numpy as np
import torch
import random
import torchvision
import argparse
import yaml
import os
import logging

from torchvision.utils import make_grid
from intraoperative_us.diffusion.models.unet_cond_base import get_config_value
from intraoperative_us.diffusion.models.vqvae import VQVAE
from intraoperative_us.diffusion.models.vae import VAE 
import intraoperative_us.diffusion.models.unet_cond_base as unet_cond_base
import intraoperative_us.diffusion.models.unet_base as unet_base
from intraoperative_us.diffusion.sheduler.scheduler import LinearNoiseScheduler
from intraoperative_us.diffusion.dataset.dataset import IntraoperativeUS
from intraoperative_us.diffusion.tools.infer_vae import get_best_model
from torch.utils.data import DataLoader
from intraoperative_us.diffusion.tools.train_cond_ldm import get_text_embeddeing

import matplotlib.pyplot as plt
import cv2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, diffusion_model_config, condition_config,
           autoencoder_model_config, diffusion_config, dataset_config, type_model, vae, save_folder, guide_w, activate_cond_ldm):
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

    ## load validation dataset
    data_img = IntraoperativeUS(size= [dataset_config['im_size_h'], dataset_config['im_size_w']],
                               dataset_path= dataset_config['dataset_path'],
                               im_channels= dataset_config['im_channels'], 
                               split='train',
                               splitting_seed=dataset_config['splitting_seed'],
                               train_percentage=dataset_config['train_percentage'],
                               val_percentage=dataset_config['val_percentage'],
                               test_percentage=dataset_config['test_percentage'],
                               condition_config=condition_config,
                               data_augmentation=False
                               )                        
    logging.info(f'len of the dataset: {len(data_img)}')
    data_loader = DataLoader(data_img, batch_size=train_config['ldm_batch_size_sample'], shuffle=False, num_workers=8)

    ## if the condition is 'text' i have to load the text model
    if 'text' in condition_types:
        
        text_configuration = condition_config['text_condition_config']
        regression_model = data_img.get_model_embedding(text_configuration['text_embed_model'], text_configuration['text_embed_trial'])
        regression_model.eval()

    for btc, data in enumerate(data_loader):
        cond_input = None
        uncond_input = {}
        if condition_config is not None:
            im, cond_input = data  # im is the image (batch_size=8), cond_input is the conditional input ['image for the mask']
            for key in cond_input.keys(): ## for all the type of condition, we move the  tensor on the device
                cond_input[key] = cond_input[key].to(device)
                uncond_input[key] = torch.zeros_like(cond_input[key])
        else:
            im = data
    
        xt = torch.randn((im.shape[0],
                      autoencoder_model_config['z_channels'],
                      im_size_h,
                      im_size_w)).to(device)

        if 'text' in condition_types:
            text_condition_input = cond_input['text'].to(device)
            text_embedding = get_text_embeddeing(text_condition_input, regression_model, device).to(device)
            cond_input['text'] = text_embedding
        logging.info(cond_input[key].shape)
        
        ################# Sampling Loop ########################
        for i in reversed(range(diffusion_config['num_timesteps'])):
            # Get prediction of noise
            t = (torch.ones((xt.shape[0],)) * i).long().to(device)

            if type_model == 'vae':
                ## get the noise prediction for the conditional and unconditional model
                noise_pred_cond = model(xt, t, cond_input)
                noise_pred_uncond = model(xt, t, uncond_input)

                ## sampling the noise for the conditional and unconditional model
                noise_pred = (1 + guide_w) * noise_pred_cond - guide_w * noise_pred_uncond

            if type_model == 'cond_vae':
                if activate_cond_ldm:
                    print('double conditional ldm')
                    ## get the noise prediction for the conditional and unconditional model
                    noise_pred_cond = model(xt, t, cond_input)
                    noise_pred_uncond = model(xt, t, uncond_input)

                    ## sampling the noise for the conditional and unconditional model
                    noise_pred = (1 + guide_w) * noise_pred_cond - guide_w * noise_pred_uncond

                else:
                    print('unconditional ldm')
                    noise_pred = model(xt, t)
            
            # Use scheduler to get x0 and xt-1
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
            
            # Save x0
            if i == 0:
                # Decode ONLY the final image to save time
                if type_model == 'vae':
                    ims = vae.decode(xt)
                if type_model == 'cond_vae':
                    for key in condition_types:  ## fake for loop., for now it is only one, get only one type of condition
                        cond_input = cond_input[key].to(device)
                    ims = vae.decode(xt, cond_input)
                    pass
            else:
                ims = x0_pred
            
            ims = torch.clamp(ims, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2
        
        for i in range(ims.shape[0]):
            cv2.imwrite(os.path.join(save_folder, f'x0_{btc * train_config["ldm_batch_size_sample"] + i}.png'), ims[i].numpy()[0]*255)


def infer(par_dir, conf, trial, experiment, epoch, guide_w, activate_cond_ldm):
    # Read the config file #
    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
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
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    ###############################################
    
    
    ############# Load tokenizer and text model #################
    # Here the section of cross-attention conditioning
    # ...
    # ...
    # ...
    ###############################################
    
    ########## Load AUTOENCODER #############
    trial_folder = os.path.join(par_dir, trial)
    assert os.listdir(trial_folder), f'No trained model found in trial folder {trial_folder}'
    logging.info(os.listdir(trial_folder))

    if 'cond_vae' in os.listdir(trial_folder):
        ## Condition VAE + LDM
        type_model = 'cond_vae'
        logging.info(f'type model {type_model}')
        logging.info(f'Load trained {os.listdir(trial_folder)[0]} model')
        best_model = get_best_model(os.path.join(trial_folder,'cond_vae'))
        logging.info(f'best model  epoch {best_model}\n')
        vae = condVAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_model_config, condition_config=condition_config).to(device)
        vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'cond_vae', f'vae_best_{best_model}.pth'), map_location=device))

        
        if activate_cond_ldm:
            ## conditional ldm
            model = unet_cond_base.Unet(im_channels=autoencoder_model_config['z_channels'], model_config=diffusion_model_config).to(device)
            model.eval()
            model_dir = os.path.join(par_dir, trial, experiment)
            model.load_state_dict(torch.load(os.path.join(model_dir, f'ldm_{epoch}.pth'),map_location=device), strict=False)
        
        else:
            ## unconditional ldm
            model = unet_base.Unet(im_channels=autoencoder_model_config['z_channels'], model_config=diffusion_model_config).to(device)
            model.eval()
            model_dir = os.path.join(par_dir, trial, experiment)
            model.load_state_dict(torch.load(os.path.join(model_dir, f'ldm_{epoch}.pth'),map_location=device), strict=False)

    if 'vae' in os.listdir(trial_folder):
        ## VAE + conditional LDM
        type_model = 'vae'
        logging.info(f'type model {type_model}')
        logging.info(f'Load trained {os.listdir(trial_folder)[0]} model')
        best_model = get_best_model(os.path.join(trial_folder,'vae'))
        logging.info(f'best model  epoch {best_model}')
        vae = VAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_model_config).to(device)
        vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'vae', f'vae_best_{best_model}.pth'), map_location=device))

        # conditional ldm
        model = unet_cond_base.Unet(im_channels=autoencoder_model_config['z_channels'], model_config=diffusion_model_config).to(device)
        model.eval()
        model_dir = os.path.join(par_dir, trial, experiment)
        model.load_state_dict(torch.load(os.path.join(model_dir, f'ldm_{epoch}.pth'),map_location=device), strict=False)

    if 'vqvae' in os.listdir(trial_folder):
        logging.info(f'Load trained {os.listdir(trial_folder)[0]} model')
        vae = VQVAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_model_config).to(device)
        vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'vqvae', 'vqvae.pth'),map_location=device))
    #####################################

    ######### Create output directories #############
    save_folder = os.path.join(model_dir, f'w_{guide_w}', f'samples_ep_{epoch}')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        overwrite = input(f"The save folder {save_folder} already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != 'y':
            print("Training aborted.")
            exit()
    
    ######## Sample from the model 
    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config, condition_config,
               autoencoder_model_config, diffusion_config, dataset_config, type_model, vae, save_folder, guide_w, activate_cond_ldm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train unconditional LDM with VQVAE')
    parser.add_argument('--save_folder', type=str, default='trained_model', help='folder to save the model, default = trained_model')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the 
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--epoch', type=int, default=100, help='epoch to sample, this is the epoch of cond ldm model')
    parser.add_argument('--guide_w', type=float, default=0.0, help='guide_w for the conditional model, w=-1 [unconditional], w=0 [vanilla conditioning], w>0 [guided conditional]')
    parser.add_argument('--cond_ldm', action='store_true', help="""Choose whether or not activate the conditional ldm. Id activate enable the combo condVAE + condLDM
                                                                     Default=False that means
                                                                     'cond_vae' -> cond VAE + unconditional LDM
                                                                     'vae' -> VAE + conditional LDM""")
    parser.add_argument('--log', type=str, default='debug', help='Logging level')
    args = parser.parse_args()

    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    experiment_dir = os.path.join(args.save_folder, args.trial, args.experiment)
    config = os.path.join(experiment_dir, 'config.yaml')

    infer(par_dir = args.save_folder, conf=config, trial=args.trial, experiment=args.experiment ,epoch=args.epoch, guide_w=args.guide_w, activate_cond_ldm=args.cond_ldm)
    plt.show()


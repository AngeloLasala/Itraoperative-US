"""
Train latent diffusion model with VAE, hugginface pipeline
"""
import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
import random
import logging
import time

from diffusers import DDPMScheduler
from diffusers import UNet2DModel
from intraoperative_us.diffusion.models.unet_base import Unet
from accelerate import Accelerator
import torch.nn.functional as F


from intraoperative_us.diffusion.dataset.dataset import IntraoperativeUS, IntraoperativeUS_mask
from intraoperative_us.diffusion.utils.utils import get_best_model, load_autoencoder, get_number_parameter
from intraoperative_us.diffusion.models.unet_cond_base import get_config_value
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

def train(par_dir, conf, trial, type_image, experiment_name):
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
    
    # Set the desired seed value 
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    #############################

    # Load the dataset
    if type_image == 'ius': data_reader = IntraoperativeUS
    elif type_image == 'mask': data_reader = IntraoperativeUS_mask

    data_img = data_reader(size= [dataset_config['im_size_h'], dataset_config['im_size_w']],
                               dataset_path= dataset_config['dataset_path'],
                               im_channels= dataset_config['im_channels'],
                               splitting_json=dataset_config['splitting_json'], 
                               split='train',
                               splitting_seed=dataset_config['splitting_seed'],
                               train_percentage=dataset_config['train_percentage'],
                               val_percentage=dataset_config['val_percentage'],
                               test_percentage=dataset_config['test_percentage'],
                               condition_config=condition_config,
                               data_augmentation=True
                               )                        
    
    logging.info(f'len of the dataset: {len(data_img)}')
    data_loader = DataLoader(data_img, batch_size=train_config['ldm_batch_size'], shuffle=True, num_workers=8)

    #Create the model and scheduler
    scheduler = DDPMScheduler(num_train_timesteps=diffusion_config['num_train_timesteps'])
    print(scheduler.config)

    model = UNet2DModel(sample_size=diffusion_model_config['sample_size'],
                        in_channels=diffusion_model_config['z_channels'],
                        out_channels=diffusion_model_config['z_channels'],
                        block_out_channels=diffusion_model_config['down_channels'])
    model.train()
    

    trial_folder = trial #os.path.join(par_dir, 'trained_model', dataset_config['name'], trial)
    assert os.listdir(trial_folder), f'No trained model found in trial folder {trial_folder}'
    if 'vae' in os.listdir(trial_folder):
        type_model = 'vae'
        logging.info(f'type model {type_model}')
        logging.info(f'Load trained {os.listdir(trial_folder)[0]} model')
        # read configuration file in this foleder
        with open(os.path.join(trial_folder, 'vae', 'config.yaml'), 'r') as f:
            autoencoder_config = yaml.safe_load(f)['autoencoder_params']

        best_model = get_best_model(os.path.join(trial_folder,'vae'))
        logging.info(f'best model  epoch {best_model}')
        vae = load_autoencoder(autoencoder_config, dataset_config, device)
        # vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'vae', f'vae_best_{best_model}.pth'), map_location=device))
   
    # freeze the vae
    vae.requires_grad_(False)

    # generate save folder
    save_dir = os.path.join(trial_folder)
    if not os.path.exists(save_dir):
        if experiment_name is not None:
            save_dir = os.path.join(save_dir, experiment_name)
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = os.path.join(save_dir, 'ldm_1' )
            os.makedirs(save_dir)
    else:
        if experiment_name is not None:
            save_dir = os.path.join(save_dir, experiment_name)
            os.makedirs(save_dir, exist_ok=True)
        else:
            current_trial = len(os.listdir(save_dir))
            save_dir = os.path.join(save_dir, f'ldm_{current_trial}')
            os.makedirs(save_dir)

    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])

    accelerator = Accelerator(
        mixed_precision=train_config['mixed_precision'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'], 
    )
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    precision_dict = {"fp16": torch.float16, "bf16": torch.bfloat16, "float32": torch.float32}
    vae.to(accelerator.device, dtype=precision_dict[train_config['mixed_precision']])

    
    # Run training
    logging.info('Start training ...')
    for epoch_idx in range(num_epochs):
        train_loss = 0.0
        progress_bar = tqdm(total=len(data_loader), disable=False)
        progress_bar.set_description(f"Epoch {epoch_idx + 1}/{num_epochs}")

        time_start = time.time()
        losses = []
        for im in data_loader:
            optimizer.zero_grad()
            im = im.float()

            with accelerator.accumulate(model):
                ## convert image to latents, scaling factor is used for normalization
                latents = vae.encode(im.to(dtype=precision_dict[train_config['mixed_precision']])).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample random noise
                noise = torch.randn_like(latents).to(device)

                # sample timesteps
                bsz = latents.shape[0]
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to images according to timestep
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                ## predic the noise residual or the velocity
                if scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif scheduler.config.prediction_type == "v_prediction":
                    target = scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss or predict the velocity and compute loss
                model_pred = model(noisy_latents, timesteps, return_dict=False)[0]

                ## compute loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_config['ldm_batch_size'])).mean()
                train_loss += avg_loss.item() / train_config['gradient_accumulation_steps']

                # Backpropagate
                accelerator.backward(loss)
                losses.append(loss.item())
                optimizer.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)
            

        time_end = time.time()
        total_time = time_end - time_start
        print(f'epoch:{epoch_idx+1}/{num_epochs} | Loss : {np.mean(losses):.4f} | Time: {total_time:.4f} sec')

        # Save the model
        if (epoch_idx+1) % train_config['save_frequency'] == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'ldm_{epoch_idx+1}.pth'))
    #         if accelerator.is_main_process:
    #             accelerate_folder = os.path.join(save_dir, f'accelerator_{epoch_idx+1}')
    #             accelerator.save_state(accelerate_folder)
    # accelerator.end_training()
    
    print('Done Training ...')
    ## save the config file
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train unconditional LDM with VQVAE')
    parser.add_argument('--conf', type=str, default='eco', help='configuration file')
    parser.add_argument('--save_folder', type=str, default='trained_model', help='folder to save the model, default = trained_model')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name, here you select the trained VAE to compute the latent space')
    parser.add_argument('--type_image', type=str, default='ius', help="""dataset to use
                                                                     'ius' -> intraoperative ultrasound
                                                                     'mask -> mask (tumor) dataset""")
    parser.add_argument('--experiment_name', type=str, default=None, help='name of the experiment, i.e., ldm_1')
    parser.add_argument('--log', type=str, default='debug', help='Logging level')
    args = parser.parse_args()

    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])
    print('Am i using GPU? ', torch.cuda.is_available())

    current_directory = os.path.dirname(__file__)
    par_dir = os.path.dirname(current_directory)
    configuration = os.path.join(par_dir, 'conf', f'{args.conf}.yaml')

    train(par_dir = par_dir,
        conf = configuration, 
        type_image = args.type_image,
        trial = os.path.join(args.save_folder, args.type_image, args.trial),
        experiment_name = args.experiment_name)
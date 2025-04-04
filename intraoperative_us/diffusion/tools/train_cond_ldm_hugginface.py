"""
Train Conditional latent diffusion model with hugginface Pipeline.
The aim pf the script is to make the training as general as possible to do
- Classifier-free guideline (CFG): train conditional LDM  with cdf as condition
- MSE training: conditional LDM training with simple MSE with ControlNET as
                backbone model for the perceptual training
- MSE + Perceptual: conditional LDM training with perceptual loss without CFG
"""
import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
import random
import multiprocessing as mp
import time
import logging

from diffusers import DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel
from accelerate import Accelerator
import torch.nn.functional as F

from intraoperative_us.diffusion.models.unet_cond_base import get_config_value, UNet2DConditionModelCostum
from intraoperative_us.diffusion.scheduler.scheduler import LinearNoiseScheduler
from intraoperative_us.diffusion.dataset.dataset import IntraoperativeUS
from intraoperative_us.diffusion.utils.utils import get_best_model, load_autoencoder, get_number_parameter
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def drop_image_condition(image_condition, im, im_drop_prob):
    """
    Classifier-free guidelines. Dropping the condition image with a certain probability
    """
    if im_drop_prob > 0:
        im_drop_mask = torch.zeros((im.shape[0], 1, 1, 1), device=im.device).float().uniform_(0,1) > im_drop_prob
        return image_condition * im_drop_mask
    else:
        return image_condition


def train(par_dir, conf, trial, experiment_name):
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
    data_img = IntraoperativeUS(size= [dataset_config['im_size_h'], dataset_config['im_size_w']],
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

    trial_folder = trial
    assert os.listdir(trial_folder), f'No trained model found in trial folder {trial_folder}'

    if 'vae' in os.listdir(trial_folder):
        ## VAE + conditional LDM
        type_model = 'vae'
        logging.info(f'type model {type_model}')
        logging.info(f'Load trained {os.listdir(trial_folder)[0]} model')
        # read configuration file in selected folder
        with open(os.path.join(trial_folder, 'vae', 'config.yaml'), 'r') as f:
            autoencoder_config = yaml.safe_load(f)['autoencoder_params']

        best_model = get_best_model(os.path.join(trial_folder,'vae'))
        logging.info(f'best model  epoch {best_model}')
        vae = load_autoencoder(autoencoder_config, dataset_config, device)
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'vae', f'vae_best_{best_model}.pth'), map_location=device))

    # Unet2DConditionModel
    model = UNet2DConditionModelCostum(diffusion_model_config)
    model.train()

    ## TEXT conditioning with CLIP text model
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(diffusion_model_config['unet_path'], diffusion_model_config['tokenizer']))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(diffusion_model_config['unet_path'], diffusion_model_config['text_encoder']), use_safetensors=True)
    def tokenize_captions(current_batch_size):
        captions = [""] * current_batch_size
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    

    ## freeze the VAE and the text encoder for saving memory
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    save_folder = os.path.join(trial_folder)
    if not os.path.exists(save_folder):
        if experiment_name is not None:
            save_folder = os.path.join(trial_folder, experiment_name)
            os.makedirs(save_folder, exist_ok=True)
        else:
            save_folder = os.path.join(save_folder, 'cond_ldm_1')
            os.makedirs(save_folder)
    else:
        if experiment_name is not None:
            save_folder = os.path.join(trial_folder, experiment_name)
            os.makedirs(save_folder, exist_ok=True)
        else:
            count = 0
            for folder in os.listdir(trial_folder):
                if folder.startswith('cond_ldm'):
                    count += 1
            save_folder = os.path.join(trial_folder, f'cond_ldm_{count+1}')
            os.makedirs(save_folder)

    ## Prepare the training
    num_epochs = train_config['ldm_epochs']
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['ldm_lr'])   # optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    accelerator = Accelerator(
        mixed_precision=train_config['mixed_precision'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'], 
    )
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    precision_dict = {"fp16": torch.float16, "bf16": torch.bfloat16, "float32": torch.float32}
    text_encoder.to(accelerator.device, dtype=precision_dict[train_config['mixed_precision']])
    vae.to(accelerator.device, dtype=precision_dict[train_config['mixed_precision']])
    

    # Run training
    logging.info('Start training ...')
    torch.cuda.empty_cache()
    for epoch_idx in range(num_epochs):
        train_loss = 0.0
        progress_bar = tqdm(total=len(data_loader), disable=False)
        progress_bar.set_description(f"Epoch {epoch_idx + 1}/{num_epochs}")

        time_start = time.time()
        losses = []
        for data in data_loader:
            cond_input = None
            if condition_config is not None:
                im, cond_input = data
            else:
                im = data

            im = im.float()
            test_tokenized_captions = tokenize_captions(im.shape[0]).to(accelerator.device)


            #############  Handiling the condition input for cond LDM ########################################
            if 'image' in condition_types:
                assert 'image' in cond_input, 'Conditioning Type Image but no image conditioning input present'
                cond_input_image = cond_input['image']
                im_drop_prob = get_config_value(condition_config['image_condition_config'], 'cond_drop_prob', 0.)
                cond_input['image'] = drop_image_condition(cond_input_image, im, im_drop_prob)
            
            with accelerator.accumulate(model):
                # Convert images to latent space, scalinf factor is used to scale the latent space with the scaling factor of the VAE
                latents = vae.encode(im.to(dtype=precision_dict[train_config['mixed_precision']])).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample random noise
                # noise = torch.randn_like(im).to(device)
                noise = torch.randn_like(latents)
                if False:#noise_offset_is_true: ## TO BE IMPLEMENTED
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                # Sample timestep
                # t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(test_tokenized_captions, return_dict=False)[0]

                ## predic the noise residual or the velocity
                if scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif scheduler.config.prediction_type == "v_prediction":
                    target = scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

               # Predict the noise residual and compute loss or predict the velocity and compute loss
                model_pred = model(noisy_latents, timesteps, encoder_hidden_states, cond_input)#return_dict=False)[0]

                ## compute loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_config['ldm_batch_size'])).mean()
                train_loss += avg_loss.item() / train_config['gradient_accumulation_steps']

                # Backpropagate
                accelerator.backward(loss)
                losses.append(loss.detach().item())
                optimizer.step()
                optimizer.zero_grad()
                

                progress_bar.update(1)
                logs = {"loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)

        ## Validation - computation of the FID score between real images (train) and generated images (validation)
        # Real images: from the datasete loader of the training set
        # Generated images: from the dataset loader of the validation set on wich i apply the diffusion and the decoder
        time_end = time.time()
        total_time = time_end - time_start
        print(f'epoch:{epoch_idx+1}/{num_epochs} | Loss : {np.mean(losses):.4f} | Time: {total_time:.4f} sec')

        # Save the model
        if (epoch_idx+1) % train_config['save_frequency'] == 0:
            torch.save(model.state_dict(), os.path.join(save_folder, f'ldm_{epoch_idx+1}.pth'))
    if accelerator.is_main_process:
        accelerate_folder = os.path.join(save_folder, f'accelerator_{epoch_idx+1}')
        accelerator.save_state(accelerate_folder)
    accelerator.end_training()

    logging.info('Done Training ...')
    ## save the config file
    with open(os.path.join(save_folder, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train unconditional LDM with VQVAE')
    parser.add_argument('--conf', type=str, default='eco', help='configuration file')
    parser.add_argument('--save_folder', type=str, default='trained_model', help='folder to save the model, default = trained_model')
    parser.add_argument('--type_image', type=str, default='ius', help='type of image, ius or mri')
    parser.add_argument('--experiment_name', type=str, default=None, help='name of the experiment, i.e., cond_ldm_1')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name, here you select the trained VAE to compute the latent space')
    parser.add_argument('--log', type=str, default='warning', help='Logging level')
    args = parser.parse_args()

    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])
    print('Am i using GPU? ', torch.cuda.is_available())

    current_directory = os.path.dirname(__file__)
    par_dir = os.path.dirname(current_directory)
    print(f'Parent directory: {par_dir}')
    configuration = os.path.join(par_dir, 'conf', f'{args.conf}.yaml')

    train(par_dir = par_dir,
        conf = configuration,
        trial = os.path.join(args.save_folder, args.type_image, args.trial),
        experiment_name = args.experiment_name)
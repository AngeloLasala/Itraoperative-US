"""
Train Conditional latent diffusion model with hugginface Pipeline.
The aim pf the script is to make the training as general as possible to do
- Classifier-free guideline (CFG): train conditional LDM  with cdf as condition
- MSE training: conditional LDM training with simple MSE with ControlNET as
                backbone model for the perceptual training
- MSE + Perceptual: conditional LDM training with perceptual loss without CFG

To DO: make this model general also for costum model
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

from diffusers import DDIMScheduler, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel
# from diffusers.optimization import get_scheduler
from accelerate import Accelerator
import torch.nn.functional as F

from intraoperative_us.diffusion.models.unet_cond_base import get_config_value, UNet2DConditionModelCostum
# import intraoperative_us.diffusion.models.unet_cond_base as unet_cond_base, 
# import intraoperative_us.diffusion.models.unet_base as unet_base
from intraoperative_us.diffusion.scheduler.scheduler import LinearNoiseScheduler
from intraoperative_us.diffusion.dataset.dataset import IntraoperativeUS
from intraoperative_us.diffusion.utils.utils import get_best_model, load_autoencoder, get_number_parameter
from torch.utils.data import DataLoader

# mp.set_start_method('spawn', force=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.empty_cache()

def drop_image_condition(image_condition, im, im_drop_prob):
    if im_drop_prob > 0:
        im_drop_mask = torch.zeros((im.shape[0], 1, 1, 1), device=im.device).float().uniform_(0,1) > im_drop_prob
        return image_condition * im_drop_mask
    else:
        return image_condition


def train(par_dir, conf, trial, activate_cond_ldm=False):
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

    else:
        raise ValueError(f"Scheduler {diffusion_config['scheduler']} not implemented")


    trial_folder = trial
    assert os.listdir(trial_folder), f'No trained model found in trial folder {trial_folder}'


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
        # vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'vae', f'vae_best_{best_model}.pth'), map_location=device))

    # Unet2DConditionModel
    # model = UNet2DConditionModel.from_pretrained(os.path.join(diffusion_model_config['unet_path'], diffusion_model_config['unet']),
    #                                             sample_size=diffusion_model_config['sample_size'],
    #                                             in_channels=autoencoder_config['z_channels'],
    #                                             out_channels=autoencoder_config['z_channels'],
    #                                             block_out_channels=diffusion_model_config['down_channels'],
    #                                             low_cpu_mem_usage=False,
    #                                             use_safetensors=True,
    #                                             ignore_mismatched_sizes=True)
    model = UNet2DConditionModelCostum(diffusion_model_config)
    model.train()

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

    save_folder = os.path.join(trial_folder, 'cond_ldm_1')
    if not os.path.exists(save_folder):
        save_folder = os.path.join(trial_folder, 'cond_ldm_1')
        os.makedirs(save_folder)
    else:
        ## count how many folder start with cond_ldm
        count = 0
        for folder in os.listdir(trial_folder):
            if folder.startswith('cond_ldm'):
                count += 1
        save_folder = os.path.join(trial_folder, f'cond_ldm_{count+1}')
        os.makedirs(save_folder)

    ## Prepare the training
    num_epochs = train_config['ldm_epochs']
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['ldm_lr'])   # optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()
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
    ## clear the gpu memory
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
                noise = torch.randn_like(im).to(device)
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
                losses.append(loss.item())
                optimizer.step()
                optimizer.zero_grad()
                

                progress_bar.update(1)
                logs = {"loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)


        ## save state with accelerator
    #     # end of the epoch

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
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name, here you select the trained VAE to compute the latent space')
    parser.add_argument('--cond_ldm', action='store_true', help="""Choose whether or not activate the conditional ldm. Id activate enable the combo condVAE + condLDM
                                                                     Default=False that means
                                                                     'cond_vae' -> cond VAE + unconditional LDM
                                                                     'vae' -> VAE + conditional LDM""")
    parser.add_argument('--log', type=str, default='warning', help='Logging level')
    args = parser.parse_args()

    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])
    print('Am i using GPU? ', torch.cuda.is_available())

    current_directory = os.path.dirname(__file__)
    par_dir = os.path.dirname(current_directory)
    configuration = os.path.join(par_dir, 'conf', f'{args.conf}.yaml')
    # save_folder = os.path.join(args.save_folder, args.trial)
    train(par_dir = par_dir,
        conf = configuration,
        trial = os.path.join(args.save_folder, 'ius', args.trial),
        activate_cond_ldm=args.cond_ldm)
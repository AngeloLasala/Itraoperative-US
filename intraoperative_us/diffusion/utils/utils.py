import torch
import logging
import os

def get_best_model(trial_folder):
    """
    Get the best model from the trial folder
    """
    best_model = 0

    # in the folder give me only the file with extention '.pth'
    for i in os.listdir(trial_folder):
        if '.pth' in i and i.split('_')[0] == 'vae':
            model = i.split('_')[-1].split('.')[0]
            if int(model) > best_model:
                best_model = int(model)
    return best_model

def get_number_parameter(model):
    """
    Get the number of parameters of the model

    Parameters
    ----------
    model : torch.nn.Module
        The model to get the number of parameters

    Returns
    -------
    int
        The number of parameters of the model, in B
    """
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    print(f"Total parameters: {num_params:.5f} B")
    print(f"Trainable parameters: {trainable_params:.5f} B")

    # return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_autoencoder(autoencoder_config, dataset_config, device):
    """
    Ausiliar function to load Autoencoder for VAE and LDM

    Parameters
    ----------
    autoencoder_config : dict
        Configuration of the autoencoder
    
    dataset_config : dict
        Configuration of the dataset


    Returns
    -------
    model : torch.nn.Module
        The autoencoder model moved on available device
    """
    ## here for avoid circlular import
    from intraoperative_us.diffusion.models.vae import VAE
    from diffusers import AutoencoderKL     
    
    
    autoencoder_type = autoencoder_config['autoencoder_type']
    initialization = autoencoder_config['initialization']

    ## my implementation - only random initialization
    if autoencoder_type == 'scratch':
        logging.info('Training VAE from scratch')
        model = VAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)

    ## model from Hugginface 
    else:
        ## set the type of generation -one_step or two_step
        if 'type_of_generation' in autoencoder_config:
            type_of_generation = autoencoder_config['type_of_generation']
            if type_of_generation == 'one_step':
                in_channels = dataset_config['im_channels'] * 2
                out_channels = dataset_config['im_channels'] * 2
            elif type_of_generation == 'two_step':
                in_channels = dataset_config['im_channels']
                out_channels = dataset_config['im_channels']

        else: ## defaoult is two_step
            type_of_generation = 'two_step'
            in_channels = dataset_config['im_channels']
            out_channels = dataset_config['im_channels']

        ## random initialization
        if initialization == 'random':
            logging.info('Training VAE with random initialization of Hugginface model')
            model = AutoencoderKL(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    sample_size=dataset_config['im_size_h'],
                    block_out_channels=autoencoder_config['down_channels'],
                    latent_channels=autoencoder_config['z_channels'],  # Default is 4
                    down_block_types=autoencoder_config.get('down_block_types', [
                        "DownEncoderBlock2D",
                        "DownEncoderBlock2D",
                        "DownEncoderBlock2D",
                        "DownEncoderBlock2D"
                    ]),
                    up_block_types=autoencoder_config.get('up_block_types', [
                        "UpDecoderBlock2D",
                        "UpDecoderBlock2D",
                        "UpDecoderBlock2D",
                        "UpDecoderBlock2D"
                    ])
                ).to(device)

        ## fine-tuning only the decoder
        elif initialization == 'only_D':
            logging.info('Training VAE with only decoder weights of Hugginface model')
            
            model = AutoencoderKL.from_pretrained(
            autoencoder_config['autoencoder_type'],
            in_channels=in_channels,
            out_channels=out_channels,
            latent_channels=autoencoder_config['z_channels'],
            sample_size=dataset_config['im_size_h'],
            block_out_channels=autoencoder_config['down_channels'],
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True).to(device)

            # Freeze the encoder by disabling gradient updates
            for param in model.encoder.parameters():
                param.requires_grad = False
            
            decoder_filter = filter(lambda p: p.requires_grad, model.parameters())

        ## exstensive fine-tuning  
        else:
            # pretrained weights
            logging.info('Training VAE with pretrained Hugginface model SDv1.5')
            model = AutoencoderKL.from_pretrained(
            autoencoder_config['autoencoder_type'],
            in_channels=in_channels,
            out_channels=out_channels,
            latent_channels=autoencoder_config['z_channels'],
            sample_size=dataset_config['im_size_h'],
            block_out_channels=autoencoder_config['down_channels'],
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True).to(device)
    
    return model


def load_unet_model(diffusion_model_config, autoencoder_config, dataset_config, device):
    """
    Ausiliar functio to load the UNet model

    Parameters
    ----------
    diffusion_model_config : dict
        Configuration of the diffusion model, main configuration

    autoencoder_config : dict
        Configuration of the autoencoder, for retrieving the latent space dimention and the number of channels
    
    dataset_config : dict
        Configuration of the dataset

    device : torch.device
        The device where to move the
    

    Returns
    -------
    model : torch.nn.Module
        The UNet model moved on available device
    """
    from diffusers import UNet2DConditionModel


    initialization = diffusion_model_config['initialization']

    if initialization == 'random':
        logging.info('Training UNet with random initialization')
        model = UNet2DConditionModel(
            sample_size=diffusion_model_config['sample_size'],
            in_channels=autoencoder_config['z_channels'],
            out_channels=autoencoder_config['z_channels'],
            block_out_channels=diffusion_model_config['down_channels'],
            cross_attention_dim=diffusion_model_config['cross_attention_dim'],

        ).to(device)

    elif initialization == 'SD1.5' or initialization == 'lora':
        logging.info('Training UNet with pretrained Hugginface model SDv1.5')
        model = UNet2DConditionModel.from_pretrained(os.path.join(diffusion_model_config['unet_path'], diffusion_model_config['unet']),
                                                sample_size=diffusion_model_config['sample_size'],
                                                in_channels=autoencoder_config['z_channels'],
                                                out_channels=autoencoder_config['z_channels'],
                                                block_out_channels=diffusion_model_config['down_channels'],
                                                low_cpu_mem_usage=False,
                                                use_safetensors=True,
                                                ignore_mismatched_sizes=True).to(device)

    return model

    
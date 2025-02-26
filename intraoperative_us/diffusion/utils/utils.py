import torch
import logging
import osW

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
    print(f"Total parameters: {num_params:.3f} B")
    print(f"Trainable parameters: {trainable_params:.3f} B")

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
        ## random initialization
        if initialization == 'random':
            logging.info('Training VAE with random initialization of Hugginface model')
            model = AutoencoderKL(
                    in_channels=dataset_config['im_channels'],
                    out_channels=dataset_config['im_channels'],
                    sample_size=dataset_config['im_size_h'],
                    block_out_channels=autoencoder_config['down_channels'],
                    latent_channels=autoencoder_config.get('latent_channels', 4),  # Default is 4
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
            in_channels=dataset_config['im_channels'],
            out_channels=dataset_config['im_channels'],
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
            in_channels=dataset_config['im_channels'],
            out_channels=dataset_config['im_channels'],
            sample_size=dataset_config['im_size_h'],
            block_out_channels=autoencoder_config['down_channels'],
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True).to(device)
    
    return model
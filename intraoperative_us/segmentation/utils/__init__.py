from ..models.unet import UNet_up

def load_model(model_config, dataset_config):
    """
    Load the UNet model from a specified path.

    Parameters:
    -----------
    model_config (dict): 
        Configuration dictionary containing model parameters.

    Returns:
    --------
    model (torch.nn.Module): 
        The loaded UNet model.
    """
    if model_config['model_type'] ==  'unet_up':
        model = UNet_up(dataset_config['im_channels'], num_classes=1)
    
    return model
    print("Loading UNet model...")
    
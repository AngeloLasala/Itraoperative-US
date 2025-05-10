"""
utils file for load dataset and model
"""
import torch
from ..models.unet import UNet_up
import logging

def load_dataset(dataset_config):
    """
    Load train and validation datasets based on the provided configuration:
    - Train with only real data experiment  : real data
    - Replacement experiment                : generated data
    - Real data with generated data         : real + generated data

    Parameters:
    -----------
    dataset_config (dict): 
        Configuration dictionary containing dataset parameters.

    Returns:    
    --------
    train_dataset (Dataset): 
        The training dataset.
    val_dataset (Dataset):
        The validation dataset.
    """
    from intraoperative_us.diffusion.dataset.dataset import IntraoperativeUS

    ## Val dataset is equal for all the experiment
    val_data = IntraoperativeUS(size= [dataset_config['im_size_h'], dataset_config['im_size_w']],
                                dataset_path= dataset_config['dataset_path'],
                                im_channels= dataset_config['im_channels'], 
                                splitting_json=dataset_config['splitting_json'],
                                split='val',
                                splitting_seed=dataset_config['splitting_seed'],
                                train_percentage=dataset_config['train_percentage'],
                                val_percentage=dataset_config['val_percentage'],
                                test_percentage=dataset_config['test_percentage'],
                                condition_config=dataset_config['condition_config'],
                                data_augmentation=False)

    data = IntraoperativeUS(size= [dataset_config['im_size_h'], dataset_config['im_size_w']],
                            dataset_path= dataset_config['dataset_path'],
                            im_channels= dataset_config['im_channels'], 
                            splitting_json=dataset_config['splitting_json'],
                            split='train',
                            splitting_seed=dataset_config['splitting_seed'],
                            train_percentage=dataset_config['train_percentage'],
                            val_percentage=dataset_config['val_percentage'],
                            test_percentage=dataset_config['test_percentage'],
                            condition_config=dataset_config['condition_config'],
                            data_augmentation=True)

    if dataset_config['dataset_type'] == 'only_real':
        logging.info('Experiment: only real data')
        logging.info(f'len data {len(data)} - len val_data {len(val_data)}')
        logging.info('')
        return data, val_data

    elif dataset_config['dataset_type'] == 'only_gen':
        from intraoperative_us.diffusion.dataset.dataset import GenerateDataset

        data_gen = GenerateDataset(par_dir = dataset_config['par_dir'],
                                   trial = dataset_config['trial'], 
                                   split = dataset_config['split'], 
                                   experiment = dataset_config['experiment'],
                                   guide_w = dataset_config['guide_w'], 
                                   scheduler = dataset_config['scheduler'],
                                   epoch = dataset_config['epoch'],
                                   size=[dataset_config['im_size_h'], dataset_config['im_size_w']], 
                                   input_channels=dataset_config['im_channels'],
                                   mask=True,
                                   num_of_images=len(data),
                                   data_augmentation=True,
                                   task='segmentation')
                        
        logging.info('Experiment: Replacement experiment')
        logging.info(f'len data {len(data_gen)} - len val_data {len(val_data)}')
        logging.info('')
        return data_gen, val_data

    elif dataset_config['dataset_type'] == 'real_and_gen':
        from intraoperative_us.diffusion.dataset.dataset import GenerateDataset

        data_gen = GenerateDataset(par_dir = dataset_config['par_dir'],
                                   trial = dataset_config['trial'], 
                                   split = dataset_config['split'], 
                                   experiment = dataset_config['experiment'],
                                   guide_w = dataset_config['guide_w'], 
                                   scheduler = dataset_config['scheduler'],
                                   epoch = dataset_config['epoch'],
                                   size=[dataset_config['im_size_h'], dataset_config['im_size_w']], 
                                   input_channels=dataset_config['im_channels'],
                                   mask=True,
                                   num_of_images=len(data),
                                   data_augmentation=True,
                                   task='segmentation')
        ## concatenate the two datasets
        data_gen_real = torch.utils.data.ConcatDataset([data, data_gen])

        logging.info('Experiment: Augmentation experiment')
        logging.info(f'len data {len(data_gen_real)} - len val_data {len(val_data)}')
        logging.info('')

        return data_gen_real, val_data

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
    
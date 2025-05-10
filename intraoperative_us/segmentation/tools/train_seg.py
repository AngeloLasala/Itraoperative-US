"""
Train segmentation model
models
- Unet
- nn Unet
- TransUnet (see paper)
"""
import os
import argparse
import logging
import torch
from torchvision import transforms
import time
import tqdm
import random
import yaml

import json
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data.dataloader import DataLoader
from intraoperative_us.segmentation.models.losses import FocalDiceLoss
from intraoperative_us.segmentation.utils import load_model, load_dataset

## deactivate the warning of the torch
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
def train(conf, save_folder, trial_name=None):
    """
    Train segmentation model for intraoperative brain tumor US

    Parameters
    ----------
    conf : str
        Path to the configuration file
    
    save_folder : str
        Path to the folder where to save the model

    trial_name : str
        Name of the trial
    """

    ## device and reproducibility  
    seed = 42  
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    ## read the configuration file
    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            logging.warning(exc)
    
    dataset_config = config['dataset_params']
    train_config = config['train_params']
    model_config = config['model_params']

    ## Read dataset
    data, val_data = load_dataset(dataset_config)
    
    data_loader = DataLoader(data, batch_size=train_config['batch_size'], shuffle=True, num_workers=0, timeout=0)
    val_data_loader = DataLoader(val_data, batch_size=train_config['batch_size'], shuffle=True, num_workers=0, timeout=0)
    
    # generate save folder
    save_dir = os.path.join(save_folder)
    trial_name = os.path.join(trial_name, f"split_{dataset_config['splitting_json'].split('.')[0].split('_')[-1]}", dataset_config['dataset_type'])
    if not os.path.exists(save_dir):
        if trial_name is not None:
            save_dir = os.path.join(save_dir, trial_name,  'model')
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = os.path.join(save_dir, trial_name, 'model')
            os.makedirs(save_dir)
    else:
        if trial_name is not None:
            save_dir = os.path.join(save_dir, trial_name, 'model')
            os.makedirs(save_dir, exist_ok=True)
        else:
            current_trial = len(os.listdir(save_dir))
            save_dir = os.path.join(save_dir, f'trial_{current_trial + 1}', 'model')
            os.makedirs(save_dir)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))

    ## Load model
    model = load_model(model_config, dataset_config).to(device)

    ## Loss and optimizer
    num_epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']
    best_vloss = 1_000_000.

    loss_fn = FocalDiceLoss(alpha=train_config['alpha'], 
                            gamma=train_config['gamma'],
                            focal_weight=train_config['focal_weight'],
                            dice_weight=train_config['dice_weight'],
                            reduction=train_config['reduction'])
    logging.info(f'Loss function...')
    logging.info(f'alpha = {train_config["alpha"]}')
    logging.info(f'gamma = {train_config["gamma"]}')
    logging.info(f'focal_weight = {train_config["focal_weight"]}')
    logging.info(f'dice_weight = {train_config["dice_weight"]}')
    logging.info(f'reduction = {train_config["reduction"]}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['learning_rate'])
    
    ## Start training
    logging.info('Training...')
    avg_vloss = 0.
    train_loss, val_loss = [], []
    for epoch_idx in range(num_epochs):
        progress_bar = tqdm.tqdm(total=len(data_loader), disable=False)
        progress_bar.set_description(f"Epoch {epoch_idx + 1}/{num_epochs}")
        time_start = time.time()

        running_loss = 0. #torch.tensor(0.).to(device)
        loss = 0.
        for data in data_loader:
            img = data[0].to(device)
            labels = data[1]['image'].to(device)

            optimizer.zero_grad()                    # Zero your gradients for every batch!
            outputs = model(img).to(device)          # predict the output
            loss = loss_fn(outputs, labels.float())  # Compute the loss 
            loss.backward()                          # Compute the gradients
            optimizer.step()                         # Adjust learning weights
            
            running_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({'train_loss': loss.item(), 'val_loss': avg_vloss})  

        last_loss = running_loss / len(data_loader)

        ## validation
        running_vloss = 0.0
        model.eval() # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization.
        with torch.no_grad():
            for i, vdata in enumerate(val_data_loader):
                vimg = vdata[0].to(device)
                vlabels = vdata[1]['image'].to(device)
            
                voutputs = model(vimg).to(device)
                vloss = loss_fn(voutputs, vlabels).item()
                running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        val_loss.append(avg_vloss)
        train_loss.append(last_loss)
        progress_bar.set_postfix({'train_loss': last_loss, 'val_loss': avg_vloss})
        ## save the model
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'best_model.pth'
            torch.save(model.state_dict(), os.path.join(save_dir, model_path))
        
        ## writer tensorboard
        writer.add_scalar('Loss/train', last_loss, epoch_idx)
        writer.add_scalar('Loss/validation', avg_vloss, epoch_idx)

        fig, ax  = plt.subplots(1, 3, figsize=(10, 5))
        ax[0].imshow(vimg[0].cpu().numpy().transpose(1, 2, 0))
        ax[0].set_title('Image')
        ax[0].axis('off')
        ax[1].imshow(vlabels[0].cpu().numpy().transpose(1, 2, 0))
        ax[1].set_title('Label')
        ax[1].axis('off')
        ax[2].imshow(voutputs[0].cpu().numpy().transpose(1, 2, 0))
        ax[2].set_title('Output')
        ax[2].axis('off')
        writer.add_figure('Image', fig, epoch_idx)

    # save last epoch model
    model_path = f'last_model.pth'
    torch.save(model.state_dict(), os.path.join(save_dir, model_path))
    np.save(os.path.join(save_dir, 'train_loss.npy'), np.array(train_loss))
    np.save(os.path.join(save_dir, 'val_loss.npy'), np.array(val_loss))

        
    config_file = os.path.join(save_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE on MNIST or CelebA-HQ')
    parser.add_argument('--conf', type=str, default='conf', help='yaml configuration file')  
    parser.add_argument('--save_folder', type=str, default='trained_model', help='folder to save the model, default = trained_model')
    parser.add_argument('--trial_name', type=str, default=None, help='name of the trial')
    parser.add_argument('--log', type=str, default='info', help='Logging level')
    args = parser.parse_args()
    
    
    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])
    print(f'Am I using GPU: {torch.cuda.is_available()}')

    current_directory = os.path.dirname(__file__)
    par_dir = os.path.dirname(current_directory)

    configuration = os.path.join(par_dir, 'conf', f'{args.conf}.yaml')
    save_folder = os.path.join(par_dir, args.save_folder, 'segmentation')
    train(conf = configuration, save_folder=save_folder, trial_name=args.trial_name)
    
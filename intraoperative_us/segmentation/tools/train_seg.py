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
from intraoperative_us.diffusion.dataset.dataset import IntraoperativeUS
from intraoperative_us.segmentation.models.losses import FocalLoss
from intraoperative_us.segmentation.utils import load_model

## deactivate the warning of the torch
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(training_loader, model, loss, optimizer, device, tb_writer = None):
    """
    Funtion that performe the training of the model for one epoch
    """
    running_loss = 0. #torch.tensor(0.).to(device)
    loss = 0.           ## this have to be update with the last_loss
    time_load_start = time.time()
    for i, (inputs, labels) in enumerate(training_loader):
        inputs, labels = inputs.to(device), labels.to(device)       # Every data instance is an input + label pair
        optimizer.zero_grad()                           # Zero your gradients for every batch!
        outputs = model(inputs)                         # Make predictions for this batch
        loss = loss_fn(outputs.float(), labels.float()) # Compute the loss and its gradientsù
        # print(f'loss {i}: {loss.device}')
        loss.backward()
        
        optimizer.step() # Adjust learning weights
        
        running_loss += loss.item()
        
    last_loss = running_loss / len(training_loader)
        # time_end = time.time()

    return last_loss

def fit(training_loader, validation_loader, model, loss_fn, optimizer, epochs=5, device='cpu', save_dir='./'):
    """
    Fit function to train the model

    Parameters
    ----------
    training_loader : torch.utils.data.DataLoader
        DataLoader object that contains the training dataset

    validation_loader : torch.utils.data.DataLoader
        DataLoader object that contains the validation dataset

    model : torch.nn.Module
        Model to train

    loss_fn : torch.nn.Module
        Loss function to use

    optimizer : torch.optim.Optimizer
        Optimizer to use

    epochs : int
        Number of epochs to train the model
    
    device : torch.device
        Device to use for training
    """
    EPOCHS = epochs
    best_vloss = 1_000_000.     # initialize the current best validation loss with a large value

    losses = {'train': [], 'valid': []}
    for epoch in range(EPOCHS):
        start = time.time()
        epoch += 1
        # print(f'EPOCH {epoch}')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        print(f'Starting epoch {epoch}/{EPOCHS}')
        start_one_epoch = time.time()
        avg_loss = train_one_epoch(training_loader, model, loss_fn, optimizer, device=device)
        print(f'Epoch {epoch}/{EPOCHS} | Time: {time.time() - start_one_epoch:.2f}s')

        running_vloss = 0.0 
        model.eval() # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization.
        
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)

                voutputs = model(vinputs)
                if len(voutputs) == 2: voutputs = voutputs[-1]
                voutputs = voutputs.to(device)
                vloss = loss_fn(voutputs, vlabels).item()
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        #convert the torch tensor  to float

        losses['train'].append(avg_loss)
        losses['valid'].append(avg_vloss)

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            # print('best model found')
            best_vloss = avg_vloss
            model_path = f'model_{epoch}'
            torch.save(model.state_dict(), os.path.join(save_dir, model_path))
        print(f'Epoch {epoch}/{EPOCHS} | Train Loss: {avg_loss:.6f} | Validation Loss: {avg_vloss:.6f} | Time: {time.time() - start:.2f}s\n')
    return losses
        
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
    print(dataset_config['condition_config'])

    ## Read dataset
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
    logging.info('DATASET')
    logging.info(f'len data {len(data)} - len val_data {len(val_data)}')
    logging.info('')

    data_loader = DataLoader(data, batch_size=train_config['batch_size'], shuffle=True, num_workers=0, timeout=0)
    val_data_loader = DataLoader(val_data, batch_size=train_config['batch_size'], shuffle=True, num_workers=0, timeout=0)
    
    # generate save folder
    save_dir = os.path.join(save_folder)
    if not os.path.exists(save_dir):
        if trial_name is not None:
            save_dir = os.path.join(save_dir, trial_name, 'model')
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = os.path.join(save_dir, 'trial_1', 'model')
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

    loss_fn = FocalLoss(alpha=train_config['alpha'], 
                        gamma=train_config['gamma'],
                        reduction=train_config['reduction'])
    ## loss BCE 
    # loss_fn = torch.nn.BCELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['learning_rate'])
    
    ## Start training
    logging.info('Training...')
    avg_vloss = 0.
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
                vimg = data[0].to(device)
                vlabels = data[1]['image'].to(device)
            
                voutputs = model(vimg).to(device)
                vloss = loss_fn(voutputs, vlabels).item()
                running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        progress_bar.set_postfix({'train_loss': last_loss, 'val_loss': avg_vloss})
        ## save the model
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'best_model.pth'
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
    
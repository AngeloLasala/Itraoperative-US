"""
Test the segmentation model
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
from surface_distance import metrics

from torch.utils.data.dataloader import DataLoader
from intraoperative_us.segmentation.models.losses import FocalDiceLoss
from intraoperative_us.segmentation.utils import load_model
from intraoperative_us.diffusion.dataset.dataset import IntraoperativeUS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics(mask_gt, mask_pred):
    """
    Compute the metrics for the segmentation
    """
    ## binary mask into boolean mask
    mask_gt = mask_gt > 0.5
    mask_pred = mask_pred > 0.5

    ## compute the metrics
    dsc = metrics.compute_dice_coefficient(mask_gt, mask_pred)
    hausdorff = metrics.compute_robust_hausdorff(metrics.compute_surface_distances(mask_gt, mask_pred, spacing_mm=(1, 1)),
                                                95)
    return dsc, hausdorff

def infer(experiment_dir, config, dataset_split):
    """
    Infer the segmentation model for intraoperative brain tumor US

    Paramerters
    ----------
    experiment_dir : str
        Path to the experiment folder
    config : str
        Path to the configuration file
    dataset_split : str

    Returns
    -------
    """

    ## Configuration
    with open(config, 'r') as file:
        try:
            config = json.load(file)
        except json.JSONDecodeError as exc:
            logging.warning(exc)
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    ## Test Dataset
    test_data = IntraoperativeUS(size= [dataset_config['im_size_h'], dataset_config['im_size_w']],
                                dataset_path= dataset_config['dataset_path'],
                                im_channels= dataset_config['im_channels'], 
                                splitting_json=dataset_config['splitting_json'],
                                split=dataset_split,
                                splitting_seed=dataset_config['splitting_seed'],
                                train_percentage=dataset_config['train_percentage'],
                                val_percentage=dataset_config['val_percentage'],
                                test_percentage=dataset_config['test_percentage'],
                                condition_config=dataset_config['condition_config'],
                                data_augmentation=False)

    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    logging.info(f'len test_data {len(test_data)}')

    ## Load model
    model = load_model(model_config, dataset_config).to(device)
    model.load_state_dict(torch.load(os.path.join(experiment_dir, 'model', 'best_model.pth')))
    model.eval()

    ## save result folder
    save_result_dir = os.path.join(experiment_dir, f'{dataset_split}_result')
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    ## Test
    dsc_list, hausdorff_list = [], []
    for ii, data in enumerate(test_data_loader):
        img = data[0].to(device)
        label = data[1]['image'].to(device)
        output = model(img).to(device)
        output = (output > 0.5).float()

        mask_gt = label[0, 0].cpu().detach().numpy().astype(np.uint8)
        mask_pred = output[0, 0, :, :].cpu().detach().numpy().astype(np.uint8)

        # plot image and labels
        fig, ax = plt.subplots(1, 3, figsize=(16, 6), tight_layout=True)
        ax[0].imshow(img[0, 0, :, :].cpu().detach().numpy(), cmap='gray')
        ax[0].set_title('Image', fontsize=18)
        ax[0].axis('off')
        ax[1].imshow(mask_gt, cmap='gray')
        ax[1].set_title('Label', fontsize=18)
        ax[1].axis('off')
        ax[2].imshow(mask_pred, cmap='gray')
        ax[2].set_title('Prediction', fontsize=18)
        ax[2].axis('off')
        plt.savefig(os.path.join(save_result_dir, f'image_{ii}.png'), dpi=300)

        dsc, haus = compute_metrics(mask_gt, mask_pred)
        dsc_list.append(dsc)
        hausdorff_list.append(haus)
    
    ## save the metrics
    dsc_list = np.array(dsc_list)
    hausdorff_list = np.array(hausdorff_list)
    np.save(os.path.join(save_result_dir, 'dsc.npy'), dsc_list)
    
    print(f'DSC: {np.mean(dsc_list)} [{np.quantile(dsc_list, 0.25)}, {np.quantile(dsc_list, 0.75)}]') 
    print(f'Hausdorff: {np.mean(hausdorff_list)} [{np.quantile(hausdorff_list, 0.25)}, {np.quantile(hausdorff_list, 0.75)}]')

        


        

    

  
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train segmentation model for tumor detection')
    parser.add_argument('--save_folder', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model/segmentation", help='folder to save the model, default = trained_model')
    parser.add_argument('--trial', type=str, default='focal_loss', help='name of the trial')
    parser.add_argument('--split', type=str, default='split_1', help='split to use for training')
    parser.add_argument('--experiment', type=str, default='only_real', help='configuration file to use')
    parser.add_argument('--dataset_split', type=str, default='val', help='dataset split to use for testing, val or test')
    parser.add_argument('--log', type=str, default='info', help='Logging level')
    args = parser.parse_args()
    
    
    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])
    print(f'Am I using GPU: {torch.cuda.is_available()}')

    experiment_dir = os.path.join(args.save_folder, args.trial, args.split, args.experiment)
    config = os.path.join(experiment_dir, 'model', 'config.json')

    infer(experiment_dir=experiment_dir, config=config, dataset_split=args.dataset_split)
    
    

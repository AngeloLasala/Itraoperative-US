"""
Compute the FID score
"""
"""
Compute the fid score to evaluate the quality of the generated images.
Modify from (https://github.com/mseitzer/pytorch-fid/tree/master?tab=readme-ov-file)
"""
import os
import argparse
import yaml
import torch
import logging
import numpy as np
import json
import pathlib
from torchvision import transforms
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import matplotlib.pyplot as plt
from pytorch_fid.inception import InceptionV3

from intraoperative_us.diffusion.evaluation.fid import ImagePathDataset, get_activations, calculate_frechet_distance, calculate_activation_statistics, calculate_fid_given_paths


def fid_experiment(dataset_path,  splitting_json, experiment_dir, batch_size=50, device=None, dims=2048, num_workers=4):
    """
    FID score of cond LDM model for all epochs.
    
    Parameters
    ----------
    config: str
        path to the configuration file

    experiment_dir: str
        path to the experiment directory

    Returns
    -------
    fid_values: list
        FID scores for all epochs

    """
    folder_list = [folder for folder in os.listdir(experiment_dir) if folder.startswith("samples_ep")]
    epoch_list = [int(folder.split('_')[-1]) for folder in folder_list]
    epoch_list.sort()

    data_real = []
    with open(os.path.join(os.path.dirname(dataset_path), splitting_json), 'r') as file:
        splitting_dict = json.load(file)
    subjects_files = splitting_dict['train']
    data_real = [os.path.join(dataset_path, subject, 'volume') for subject in subjects_files]
       
    # Fake images validation
    fid_values = {}
    for epoch in epoch_list:
        data_fake = [os.path.join(experiment_dir, f'samples_ep_{epoch}', 'ius')]
        fid_value = calculate_fid_given_paths([data_real, data_fake], batch_size, device, dims, num_workers)
        fid_values[epoch] = fid_value

    return fid_values



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID score.")
    parser.add_argument('--save_folder', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model",
                                                   help='folder to save the model')
    parser.add_argument('--dataset_path', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/RESECT_iUS_dataset/dataset",
                                                    help='path to the real dataset') 
    parser.add_argument('--splitting_json', type=str, default="splitting.json", help='json file containing the splitting of the dataset')
    parser.add_argument('--type_image', type=str, default='ius', help='type of image to evaluate, ius or mask')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the 
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--guide_w', type=float, default=0.0, help='guide_w for the conditional model, w=-1 [unconditional], w=0 [vanilla conditioning], w>0 [guided conditional]')
    parser.add_argument('--show_plot', action='store_true', help="show and save the FID plot, default=False")
    parser.add_argument('--log', type=str, default='info', help='Logging level')
    args = parser.parse_args()

    print('Am I using GPU: ', torch.cuda.is_available())
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")


    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    experiment_dir = os.path.join(args.save_folder, args.type_image, args.trial, args.experiment)
    experiment_dir_w = os.path.join(experiment_dir, f'w_{args.guide_w}')

    fid = fid_experiment(args.dataset_path, args.splitting_json, experiment_dir_w, device=device)
    for key, value in fid.items():
        print(f'Epoch: {key}, FID: {value}')    
    
    ## save the FID score
    with open(os.path.join(experiment_dir, f'w_{args.guide_w}', 'FID_score.txt'), 'w') as f:
        for key, value in fid.items():
            f.write(f'Epoch: {key}, FID: {value}\n')
        
    if args.show_plot:
        fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5), num=f'FID score', tight_layout=True)
        ax.plot(list(fid.keys()), list(fid.values()), marker='o', color='b')
        ax.set_xlabel('Epoch', fontsize=20)
        ax.set_ylabel('FID', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid('dotted')
        plt.savefig(os.path.join(experiment_dir, 'FID_score.png'))
        plt.show()
 

   
    
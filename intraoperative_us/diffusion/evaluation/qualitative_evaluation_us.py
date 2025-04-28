"""
Qualitative comparison between real image (training) and generated image 
"""
import argparse
import glob
import os
import pickle
import logging

import torch
import torchvision
import yaml
import numpy as np
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from intraoperative_us.diffusion.evaluation.investigate_vae import get_config_value, get_best_model
from intraoperative_us.diffusion.dataset.dataset import IntraoperativeUS, GenerateDataset
from intraoperative_us.diffusion.utils.utils import get_best_model, load_autoencoder, get_number_parameter

from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
from scipy.spatial.distance import cdist
from PIL import Image
from torchvision import transforms
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## set reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def infer(par_dir, conf, trial, split, experiment, epoch, guide_w, scheduler, show_gen_mask):
    ######## Read the config file #######
    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    dataset_config = config['dataset_params']
    autoencoder_model_config = config['autoencoder_params']
    condition_config = get_config_value(autoencoder_model_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']



    # Train dataset
    data_img = IntraoperativeUS(size= [dataset_config['im_size_h'], dataset_config['im_size_w']],
                               dataset_path= dataset_config['dataset_path'],
                               im_channels= dataset_config['im_channels'],
                               splitting_json=dataset_config['splitting_json'], 
                               split='train',
                               splitting_seed=dataset_config['splitting_seed'],
                               train_percentage=dataset_config['train_percentage'],
                               val_percentage=dataset_config['val_percentage'],
                               test_percentage=dataset_config['test_percentage'],
                               condition_config=config['ldm_params']['condition_config'],
                               data_augmentation=False)
    logging.info(f'len train data {len(data_img)}')
    data_loader = DataLoader(data_img, batch_size=1, shuffle=False, num_workers=8)


    data_gen = GenerateDataset(par_dir, trial, split, experiment, guide_w, scheduler, epoch, size=[dataset_config['im_size_h'], dataset_config['im_size_w']], input_channels=dataset_config['im_channels'])
    data_loader_gen = DataLoader(data_gen, batch_size=1, shuffle=False, num_workers=8)
    logging.info(f'len gen data {len(data_gen)}')

    if show_gen_mask:
        data_gen_mask = GenerateDataset(par_dir, trial, split, experiment, guide_w, scheduler, epoch, 
                                        size=[dataset_config['im_size_h'], dataset_config['im_size_w']], input_channels=dataset_config['im_channels'],
                                        mask=True)
        for i, (gen_img, mask) in enumerate(data_gen_mask):
            ## plt the generated image and the mask
            print(gen_img.shape, mask.shape)
            plt.figure(figsize=(25,12), num=f'gen_mask_{i}', tight_layout=True)
            plt.subplot(1,3,2)
            plt.imshow(gen_img[0,:,:].cpu().numpy(), cmap='gray')
            plt.title('Generated image', fontsize=30)
            plt.axis('off')
            plt.subplot(1,3,1)
            plt.imshow(mask[0,:,:].cpu().numpy(), cmap='gray')
            plt.title('Mask', fontsize=30)
            plt.axis('off')
            plt.subplot(1,3,3)
            plt.imshow(gen_img[0,:,:].cpu().numpy(), cmap='gray')
            tumor_mask = mask[0,:,:].cpu().numpy()
            plt.imshow(np.ma.masked_where(tumor_mask == 0, tumor_mask), cmap='ocean', alpha=0.3)
            plt.title('Generated image with mask', fontsize=30)
            plt.axis('off')
            plt.show() 


    trial_folder = os.path.join(par_dir, trial, split)
    assert os.listdir(trial_folder), f'No trained model found in trial folder {trial_folder}'

    if 'vae' in os.listdir(trial_folder):
        type_model = 'vae'
        logging.info(f'type model {type_model}')
        logging.info(f'Load trained {os.listdir(trial_folder)[0]} model')
        # read configuration file in selected folder
        with open(os.path.join(trial_folder, 'vae', 'config.yaml'), 'r') as f:
            autoencoder_config = yaml.safe_load(f)['autoencoder_params']

        best_model = get_best_model(os.path.join(trial_folder,'vae'))
        logging.info(f'best model  epoch {best_model}')
        vae = load_autoencoder(autoencoder_config, dataset_config, device)
        vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'vae', f'vae_best_{best_model}.pth'), map_location=device))


    # evalaute the recon loss on test set
    recon_criterion = torch.nn.MSELoss()            # L1/L2 loss for Reconstruction
    encoded_output_list, encoded_val_list, gen_encoded_output_list = [], [], []

    
    with torch.no_grad():
        ## Real data
        for im, cond in tqdm(data_loader):
            im = im.float().to(device)
            for key in cond.keys(): ## for all the type of condition, we move the  tensor on the device
                cond[key] = cond[key].to(device)

            ## Encoder - get the latent space
            encoded_output = vae.encode(im).latent_dist.sample()

            encoded_output = encoded_output[0,:,:,:].flatten()
            encoded_output_list.append(encoded_output.cpu().numpy())

    encoded_output_list = np.array(encoded_output_list)
      
    with torch.no_grad():
        for gen_img in tqdm(data_loader_gen):
            gen_img = gen_img.float().to(device)

            ## Encoder - get the latent space
            gen_encoded_output = vae.encode(gen_img).latent_dist.sample()

            gen_encoded_output = gen_encoded_output[0,:,:,:].flatten()
            gen_encoded_output_list.append(gen_encoded_output.cpu().numpy())
        gen_encoded_output_list = np.array(gen_encoded_output_list)
         

    real_gen_stack = np.vstack((encoded_output_list, gen_encoded_output_list))
    real_gen_label_stack = np.concatenate((np.ones(encoded_output_list.shape[0]), 
                                        #    np.ones(encoded_val_list.shape[0])*2,
                                           np.zeros(gen_encoded_output_list.shape[0])))
   
    pca = PCA(n_components=50)
    real_gen_stack = pca.fit_transform(real_gen_stack)

    np.random.seed(42)
    seeds = np.random.choice(np.arange(1000), 1, replace=False)
    logging.info(f'seeds TSNE: {seeds}')
    print('TSNE embedding...')

    metrics = {'silhouette_score':[], 'davies_bouldin_score':[], 'centroid_distance':[]}
    progress_bar = tqdm(total=len(seeds), disable=False)
    progress_bar.set_description(f"TSNE random")
    for s in seeds:
        print(s)
        tsne = TSNE(n_components=2, random_state=s, init="pca", method='exact')
        encoded_output_tsne = tsne.fit_transform(real_gen_stack)
        real_gen_stack_tsne = encoded_output_tsne.copy()

        # real is where real_gen_label_stack == 1
        real_gen_stack_tsne = (real_gen_stack_tsne - np.min(real_gen_stack_tsne, axis=0)) / (np.max(real_gen_stack_tsne, axis=0) - np.min(real_gen_stack_tsne, axis=0))
        real_tsne = real_gen_stack_tsne[real_gen_label_stack==1]
        val_tsne = real_gen_stack_tsne[real_gen_label_stack==2]
        gen_tsne = real_gen_stack_tsne[real_gen_label_stack==0]
        
    
        ## compute the silouhette score between the real and generated data
        silhouette_score_all = silhouette_score(real_gen_stack_tsne, real_gen_label_stack)
        
        # compute the DB values for the real and generated data
        davies_bouldin_score_all = davies_bouldin_score(real_gen_stack_tsne, real_gen_label_stack)

        ## compute the centroids of the real and generated data
        centroids = np.array([np.mean(real_tsne, axis=0), np.mean(gen_tsne, axis=0)])
        centroid_distance = np.linalg.norm(centroids[0] - centroids[1])
        metrics['silhouette_score'].append(silhouette_score_all)
        metrics['davies_bouldin_score'].append(davies_bouldin_score_all)
        metrics['centroid_distance'].append(centroid_distance)

        progress_bar.update(1)
        logs = {"Silouette": silhouette_score_all, "DB": davies_bouldin_score_all, "centroid_distance": centroid_distance}
        progress_bar.set_postfix(**logs)
        
    print('silhouette_score', np.mean(metrics['silhouette_score']), np.std(metrics['silhouette_score']))
    print('davies_bouldin_score', np.mean(metrics['davies_bouldin_score']), np.std(metrics['davies_bouldin_score']))
    print('centroid_distance', np.mean(metrics['centroid_distance']), np.std(metrics['centroid_distance']))

    ######################## PLOT #################################################################

    ## plt the 2d scatter plot of tsne
    plt.figure(figsize=(10,10), num=f'iUS tsne model', tight_layout=True)   

    plt.scatter(real_tsne[:,0], real_tsne[:,1], c='blue', label='Train data', s=100)
    plt.scatter(val_tsne[:,0], val_tsne[:,1], c='green', label='Val data', s=100)
    plt.scatter(gen_tsne[:,0], gen_tsne[:,1], c='lightgreen', label='Gen data', s=100)

    plt.xlabel('TSNE 1', fontsize=30)
    plt.ylabel('TSNE 2', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    # plt.legend(fontsize=34)
    plt.show()


    ##################################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Invastigate the latent space')
    parser.add_argument('--save_folder', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model",
                                                   help='folder to save the model')
    parser.add_argument('--type_image', type=str, default='ius', help='type of image to evaluate, ius or mask')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--split', type=str, default='split_1', help='split among the 5 fold, default=split_1')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the 
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--guide_w', type=float, default=0.0, help='guide_w for the conditional model, w=-1 [unconditional], w=0 [vanilla conditioning], w>0 [guided conditional]')
    parser.add_argument('--scheduler', type=str, default='ddpm', help='sheduler used for sampling, i.e. ddpm, pndm')
    parser.add_argument('--epoch', type=int, default=99, help='epoch to sample, this is the epoch of cond ldm model') 
    parser.add_argument('--show_gen_mask', action='store_true', help="show the generative and mask images, default=False")
    parser.add_argument('--log', type=str, default='debug', help='Logging level')
    args = parser.parse_args()

    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    experiment_dir = os.path.join(args.save_folder, args.type_image, args.trial, args.split)
    if 'vae' in os.listdir(experiment_dir): config = os.path.join(experiment_dir, 'vae', 'config.yaml')

    infer(par_dir = os.path.join(args.save_folder, args.type_image), conf=config, trial=args.trial, split=args.split,
         experiment=args.experiment, epoch=args.epoch, guide_w=args.guide_w, scheduler=args.scheduler, show_gen_mask=args.show_gen_mask)
    plt.show()
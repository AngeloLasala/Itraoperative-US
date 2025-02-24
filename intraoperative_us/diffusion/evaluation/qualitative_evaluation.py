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
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt

from intraoperative_us.diffusion.evaluation.investigate_vae import get_config_value, get_best_model
# from intraoperative_us.diffusion.evaluation.hypertropy_eval import GenerateDataset
from intraoperative_us.diffusion.dataset.dataset import IntraoperativeUS
from intraoperative_us.diffusion.models.vae import VAE

from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
from scipy.spatial.distance import cdist
from PIL import Image
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GenerateDataset(torch.utils.data.Dataset):
    """
    Dataset of generated image loaded from the path
    """
    def __init__(self, par_dir, trial, experiment, guide_w, epoch, size, input_channels):
        self.par_dir = par_dir
        self.trial = trial
        self.experiment = experiment
        self.guide_w = guide_w
        self.epoch = epoch
        self.size = size
        self.input_channels = input_channels

        self.data_ius= self.get_eco_path()
        self.files_data = [os.path.join(self.data_ius, f'x0_{i}.png') for i in range(len(os.listdir(self.data_ius)))]

    def __len__(self):
        return len(self.files_data)

    def __getitem__(self, idx):
        image_path = self.files_data[idx]

        # read the image wiht PIL
        image = Image.open(image_path)
        resize = transforms.Resize(size=self.size)
        image = resize(image)
        if self.input_channels == 1: image = image.convert('L')
        image = transforms.functional.to_tensor(image)
        image = (2 * image) - 1 

        return image

    def get_eco_path(self):
        """
        retrive the path 'eco' from current directory
        """
        data_ius = os.path.join(self.par_dir, self.trial, self.experiment, f'w_{self.guide_w}', f'samples_ep_{self.epoch}','ius')
        return data_ius

    def get_mask_images(self):
        pass


def infer(par_dir, conf, trial, experiment, epoch, guide_w, compute_real, compute_gen):
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

    # To Do: create the dataset and the dataloader of the generated images
    data_val = IntraoperativeUS(size= [dataset_config['im_size_h'], dataset_config['im_size_w']],
                               dataset_path= dataset_config['dataset_path'],
                               im_channels= dataset_config['im_channels'],
                               splitting_json=dataset_config['splitting_json'], 
                               split='val',
                               splitting_seed=dataset_config['splitting_seed'],
                               train_percentage=dataset_config['train_percentage'],
                               val_percentage=dataset_config['val_percentage'],
                               test_percentage=dataset_config['test_percentage'],
                               condition_config=config['ldm_params']['condition_config'],
                               data_augmentation=False)
    logging.info(f'len val data {len(data_val)}')
    data_loader_val = DataLoader(data_val, batch_size=1, shuffle=False, num_workers=8)

    data_gen = GenerateDataset(par_dir, trial, experiment, guide_w, epoch, size=[dataset_config['im_size_h'], dataset_config['im_size_w']], input_channels=dataset_config['im_channels'])
    data_loader_gen = DataLoader(data_gen, batch_size=1, shuffle=False, num_workers=8)
    logging.info(f'len gen data {len(data_gen)}')


    trial_folder = os.path.join(par_dir, trial)
    assert os.listdir(trial_folder), f'No trained model found in trial folder {trial_folder}'

    if 'vae' in os.listdir(trial_folder):
        type_model = 'vae'
        logging.info(f'Load trained {os.listdir(trial_folder)[0]} model')
        best_model = get_best_model(os.path.join(trial_folder,'vae'))
        logging.info(f'best model  epoch {best_model}')
        vae = VAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_model_config).to(device)
        vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'vae', f'vae_best_{best_model}.pth'), map_location=device))

    if 'vqvae' in os.listdir(trial_folder):
        logging.info(f'Load trained {os.listdir(trial_folder)[0]} model')
        vae = VQVAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_model_config).to(device)
        vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'vqvae', 'vqvae.pth'),map_location=device))

    ## evalaute the recon loss on test set
    recon_criterion = torch.nn.MSELoss()            # L1/L2 loss for Reconstruction
    encoded_output_list, encoded_val_list, gen_encoded_output_list = [], [], []

    if compute_real:
        with torch.no_grad():
            ## Real data
            for im, cond in tqdm(data_loader):
                im = im.float().to(device)
                for key in cond.keys(): ## for all the type of condition, we move the  tensor on the device
                    cond[key] = cond[key].to(device)

                ## Encoder - get the latent space
                encoded_output, _ = vae.encode(im)

                encoded_output = encoded_output[0,:,:,:].flatten()
                encoded_output_list.append(encoded_output.cpu().numpy())
       
        encoded_output_list = np.array(encoded_output_list)

        with torch.no_grad():
            ## Real data
            for im, cond in tqdm(data_loader_val):
                im = im.float().to(device)
                for key in cond.keys(): ## for all the type of condition, we move the  tensor on the device
                    cond[key] = cond[key].to(device)

                ## Encoder - get the latent space
                encoded_output_val, _ = vae.encode(im)

                encoded_output_val = encoded_output_val[0,:,:,:].flatten()
                encoded_val_list.append(encoded_output_val.cpu().numpy())
       
        encoded_val_list = np.array(encoded_val_list)
  
    else:
        tsne_folder = os.path.join(trial_folder, experiment, 'tsne')        
        encoded_output_list = np.load(os.path.join(tsne_folder, 'encoded_output_list.npy'))




    if compute_gen:    
        with torch.no_grad():
            for gen_img in tqdm(data_loader_gen):
                gen_img = gen_img.float().to(device)

                ## Encoder - get the latent space
                gen_encoded_output, _ = vae.encode(gen_img)

                gen_encoded_output = gen_encoded_output[0,:,:,:].flatten()
                gen_encoded_output_list.append(gen_encoded_output.cpu().numpy())
            gen_encoded_output_list = np.array(gen_encoded_output_list)
         
    else:
        tsne_folder = os.path.join(trial_folder, experiment, 'tsne')        
        gen_encoded_output_list = np.load(os.path.join(tsne_folder, 'gen_encoded_output_list.npy'))

    real_gen_stack = np.vstack((encoded_output_list, encoded_val_list, gen_encoded_output_list))
    real_gen_label_stack = np.concatenate((np.ones(encoded_output_list.shape[0]), 
                                           np.ones(encoded_val_list.shape[0])*2,
                                           np.zeros(gen_encoded_output_list.shape[0])))
   
    pca = PCA(n_components=50)
    real_gen_stack = pca.fit_transform(real_gen_stack)

    np.random.seed(42)
    seeds = np.random.choice(np.arange(1000), 30, replace=False)
    logging.info(f'seeds TSNE: {seeds}')
    print('TSNE embedding...')

    metrics = {'silhouette_score':[], 'davies_bouldin_score':[], 'centroid_distance':[]}
    progress_bar = tqdm(total=len(seeds), disable=False)
    progress_bar.set_description(f"TSNE random")
    for s in seeds:
        tsne = TSNE(n_components=2, perplexity=3, random_state=s)
        encoded_output_tsne = tsne.fit_transform(real_gen_stack)
        real_gen_stack_tsne = encoded_output_tsne.copy()

        # real is where real_gen_label_stack == 1
        real_gen_stack_tsne = (real_gen_stack_tsne - np.min(real_gen_stack_tsne, axis=0)) / (np.max(real_gen_stack_tsne, axis=0) - np.min(real_gen_stack_tsne, axis=0))
        real_tsne = real_gen_stack_tsne[real_gen_label_stack==1]
        val_tsne = real_gen_stack_tsne[real_gen_label_stack==2]
        gen_tsne = real_gen_stack_tsne[real_gen_label_stack==0]
        
    
        ## compute the silouhette score between the real and generated data
        # real_gen_stack_tsne = (real_gen_stack_tsne - np.min(real_gen_stack_tsne, axis=0)) / (np.max(real_gen_stack_tsne, axis=0) - np.min(real_gen_stack_tsne, axis=0))
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

    plt.scatter(real_tsne[:,0], real_tsne[:,1], c='blue', label='Train data', s=60)
    plt.scatter(val_tsne[:,0], val_tsne[:,1], c='green', label='Val data', s=60)
    plt.scatter(gen_tsne[:,0], gen_tsne[:,1], c='lightgreen', label='Gen data', s=60)

    plt.xlabel('TSNE 1', fontsize=22)
    plt.ylabel('TSNE 2', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=24)
    plt.show()


    ##################################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Invastigate the latent space')
    parser.add_argument('--save_folder', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model",
                                                   help='folder to save the model')
    parser.add_argument('--type_image', type=str, default='ius', help='type of image to evaluate, ius or mask')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the 
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--guide_w', type=float, default=0.0, help='guide_w for the conditional model, w=-1 [unconditional], w=0 [vanilla conditioning], w>0 [guided conditional]')
    parser.add_argument('--epoch', type=int, default=99, help='epoch to sample, this is the epoch of cond ldm model') 
    parser.add_argument('--compute_encodings_real', action='store_true', help="compite the embeddings, default=False")
    parser.add_argument('--compute_encodings_gen', action='store_true', help="compite the embeddings, default=False")
    parser.add_argument('--log', type=str, default='debug', help='Logging level')
    args = parser.parse_args()

    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    experiment_dir = os.path.join(args.save_folder, args.type_image, args.trial)
    if 'vae' in os.listdir(experiment_dir): config = os.path.join(experiment_dir, 'vae', 'config.yaml')
    if 'vqvae' in os.listdir(experiment_dir): config = os.path.join(experiment_dir, 'vqvae', 'config.yaml')
    if 'cond_vae' in os.listdir(experiment_dir): config = os.path.join(experiment_dir, 'cond_vae', 'config.yaml')


    infer(par_dir = os.path.join(args.save_folder, args.type_image), conf=config, trial=args.trial, 
         experiment=args.experiment, epoch=args.epoch, guide_w=args.guide_w, compute_real=args.compute_encodings_real, compute_gen=args.compute_encodings_gen)
    plt.show()
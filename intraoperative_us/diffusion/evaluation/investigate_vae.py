"""
Invastigate haw the VAE encode the PLAX, in other words, is the latent space of the VAE able to capture the PLAX information?
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

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from intraoperative_us.diffusion.dataset.dataset import IntraoperativeUS, IntraoperativeUS_mask
from intraoperative_us.diffusion.models.vqvae import VQVAE
from intraoperative_us.diffusion.utils.utils import load_autoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_config_value(config, key, default_value):
    """
    Get the value of a key from the config dictionary
    """
    return config[key] if key in config else default_value


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


def plot_image_latent(image, latent):
    """
    Plot the latent space of the image
    """
    latent_dict = {}

    ## plot the latent space
    latent_original = latent[0,:,:,:].cpu().permute(1,2,0).numpy()
    print(latent_original.shape)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
    ax.set_title('Image - latent space', fontsize=20)
    ax.axis('off')
    ax.imshow(latent_original, cmap='jet')

    ## plot only the original images
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
    ax.set_title('Image', fontsize=20)
    ax.axis('off')
    ax.imshow(image[0, 0, :, :].cpu().numpy(), cmap='gray')



    laten_img = torchvision.transforms.Resize((image.shape[2], image.shape[3]))(latent)

    fig, ax = plt.subplots(1, laten_img.shape[1], figsize=(21, 8), tight_layout=True)
    for i in range(laten_img.shape[1]):
        ax[i].set_title(f'Image - ch latent {i}', fontsize=20)
        ax[i].imshow(image[0, 0, :, :].cpu().numpy(), cmap='gray')
        ax[i].imshow(laten_img[0, i, :, :].cpu().numpy(), cmap='jet', alpha=0.2)

    for axis in ax:
        axis.axis('off')

    ##
    if latent.shape[1] == 4:
        latent_1 = laten_img[0, 0, :, :].cpu().numpy() + laten_img[0, 2, :, :].cpu().numpy()
        latent_2 = laten_img[0, 1, :, :].cpu().numpy() + laten_img[0, 3, :, :].cpu().numpy()
        #normalize the latent in the range 0-1
        latent_1 = (latent_1 - np.min(latent_1)) / (np.max(latent_1) - np.min(latent_1))
        latent_2 = (latent_2 - np.min(latent_2)) / (np.max(latent_2) - np.min(latent_2))

        fig, ax = plt.subplots(1, 2, figsize=(21, 8), tight_layout=True)
        ax[0].imshow(image[0, 0, :, :].cpu().numpy(), cmap='gray')
        ax[0].imshow(latent_1, cmap='jet', alpha=0.2)
        ax[0].set_title('Image - ch latent 0 + ch latent 2', fontsize=20)
        ax[0].axis('off')

        ax[1].imshow(image[0, 0, :, :].cpu().numpy(), cmap='gray')
        ax[1].imshow(latent_2, cmap='jet', alpha=0.2)
        ax[1].set_title('Image - ch latent 1 + ch latent 3', fontsize=20)
        ax[1].axis('off')

def plot_im_cond_rec(im, rec, cond):
    """
    Plot the image, condition and the reconstructed image
    """
    im = (im + 1) / 2
    rec = (rec + 1) / 2
    fig, ax = plt.subplots(1, 3, figsize=(21, 8), num=get_hypertrophy_from_one_hot(cond), tight_layout=True)
    real_img = (im[0, 0, :, :].cpu().numpy() * 255) + 0.00003
    rec_img = (rec[0, 0, :, :].cpu().numpy() * 255) + 0.00003

    ax[0].imshow(real_img, cmap='gray')
    ax[0].set_title('Original image', fontsize=20)
    ax[0].axis('off')


    ax[1].imshow(rec_img, cmap='gray')
    ax[1].set_title('Reconstructed Image', fontsize=20)
    ax[1].axis('off')

    ax[2].imshow(np.abs(real_img - rec_img) / rec_img, cmap='hot')
    ax[2].set_title('Difference', fontsize=20)

def plot_difference_matrix(original, out_1, out_2, out_3, out_4):
    """
    Plot the difference matrix
    """
    original = (original + 1) / 2
    out_1 = (out_1 + 1) / 2
    out_2 = (out_2 + 1) / 2
    out_3 = (out_3 + 1) / 2
    out_4 = (out_4 + 1) / 2

    generations = [original.cpu().numpy(), out_1.cpu().numpy(), out_2.cpu().numpy(), out_3.cpu().numpy(), out_4.cpu().numpy()]
    generations = [(i * 255) + 0.00003 for i in generations]
    label = ['Original_cond', 'CH', 'CR', 'EC', 'Normal geometry']
    fig, ax = plt.subplots(5, 5, figsize=(30, 20), num='Difference between cond', tight_layout=True)
    for i,data_1 in enumerate(generations):
        for j, data_2 in enumerate(generations):
            ax[i, j].imshow(np.abs(generations[i][0,0,:,:] - generations[j][0,0,:,:])/ generations[0][0,0,:,:], cmap='hot')
            ax[i, j].set_title(f'{label[i]} - {label[j]}', fontsize=20)
            ax[i, j].axis('off')

def infer(par_dir, conf, trial, type_image, show_plot=False):
    ######## Read the config file #######
    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    train_config = config['train_params']
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    initialization = autoencoder_config['initialization']
    diffusion_model_config = config['ldm_params']      ## here for taking the mask while loading the dataset
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']

    # Create the dataset and dataloader
    if type_image == 'ius':
        data = IntraoperativeUS(size= [dataset_config['im_size_h'], dataset_config['im_size_w']],
                                dataset_path= dataset_config['dataset_path'],
                                im_channels= dataset_config['im_channels'],
                                splitting_json=dataset_config['splitting_json'], 
                                split='train',
                                splitting_seed=dataset_config['splitting_seed'],
                                train_percentage=dataset_config['train_percentage'],
                                val_percentage=dataset_config['val_percentage'],
                                test_percentage=dataset_config['test_percentage'],
                                condition_config=config['autoencoder_params']['condition_config'],
                                data_augmentation=False)
    elif type_image == 'mask':
        data = IntraoperativeUS_mask(size= [dataset_config['im_size_h'], dataset_config['im_size_w']],
                            dataset_path= dataset_config['dataset_path'],
                            im_channels= dataset_config['im_channels'], 
                            splitting_json=dataset_config['splitting_json'],
                            split='train',
                            splitting_seed=dataset_config['splitting_seed'],
                            train_percentage=dataset_config['train_percentage'],
                            val_percentage=dataset_config['val_percentage'],
                            test_percentage=dataset_config['test_percentage'],
                            data_augmentation=False)

    elif type_image == 'one_step':
        data = IntraoperativeUS(size= [dataset_config['im_size_h'], dataset_config['im_size_w']],
                                dataset_path= dataset_config['dataset_path'],
                                im_channels= dataset_config['im_channels'],
                                splitting_json=dataset_config['splitting_json'], 
                                split='train',
                                splitting_seed=dataset_config['splitting_seed'],
                                train_percentage=dataset_config['train_percentage'],
                                val_percentage=dataset_config['val_percentage'],
                                test_percentage=dataset_config['test_percentage'],
                                condition_config=condition_config,
                                data_augmentation=False)

    logging.info(f'len data {len(data)}')
    data_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=8)

    num_images = train_config['num_samples']
    ngrid = train_config['num_grid_rows']

    trial_folder = os.path.join(par_dir, trial)
    assert os.listdir(trial_folder), f'No trained model found in trial folder {trial_folder}'
    logging.info(os.listdir(trial_folder))

    if 'vae' in os.listdir(trial_folder):
        type_model = 'vae'
        logging.info(f'Load trained {os.listdir(trial_folder)[0]} model')
        best_model = get_best_model(os.path.join(trial_folder,'vae'))
        vae = load_autoencoder(autoencoder_config, dataset_config, device)

        vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'vae', f'vae_best_{best_model}.pth'), map_location=device))



    if 'vqvae' in os.listdir(trial_folder):
        logging.info(f'Load trained {os.listdir(trial_folder)[0]} model')
        vae = VQVAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_config).to(device)
        vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'vqvae', 'vqvae.pth'),map_location=device))

    ## evalaute the recon loss on test set
    recon_criterion = torch.nn.MSELoss()            # L1/L2 loss for Reconstruction
    encoded_output_list = []

    with torch.no_grad():
        test_recon_losses = []
        test_perceptual_losses = []

        progress_bar = tqdm(total=len(data_loader), disable=False)
        progress_bar.set_description('Loop over the dataset')
        for nn, im in enumerate(data_loader):
            if type_image == 'one_step':
                img = im[0]
                mask = im[1]['image']
                im = torch.cat((im[0], mask), dim=1).float().to(device)
                
            else:
                im = im.float().to(device)

            ## Encoder - get the latent space
            if initialization == 'scratch':
                encoded_output, _ = vae.encode(im)
                output = vae.decode(encoded_output)

            else:
                encoded_output = vae.encode(im).latent_dist.sample()           # Encode image
                output = vae.decode(vae.encode(im).latent_dist.sample()).sample
            
            
            recon_loss = recon_criterion(output, im)
            encoded_output = torch.clamp(encoded_output, -1., 1.)
            encoded_output = (encoded_output + 1) / 2

            progress_bar.update(1)

            if show_plot and nn%20 == 0:
                encoded_plt = encoded_output[0,:,:,:].cpu().permute(1,2,0).numpy()
                print(f'min {encoded_output.min()}, MAX: {encoded_output.max()}')

                ## plot the real encoded_output
                for i in range(encoded_plt.shape[2]):
                    plt.figure(figsize=(8, 8), num=f'Latent channel {i}')
                    plt.imshow(encoded_plt[:,:,i])
                    plt.axis('off')
                plt.axis('off')

                ## plot the image, latent space and the reconstructed
                if type_image == 'one_step':
                    plt.figure(num='reconstructed', figsize=(10, 10), tight_layout=True)
                    plt.subplot(2, 2, 1)
                    plt.title('Real image', fontsize=20)
                    plt.imshow(im[0,0,:,:].cpu().numpy(), cmap='gray')
                    plt.axis('off')
                    plt.subplot(2, 2, 2)
                    plt.title('Real Mask', fontsize=20)
                    plt.imshow(im[0,1,:,:].cpu().numpy(), cmap='gray')
                    plt.axis('off')
                    plt.subplot(2, 2, 3)
                    plt.title('Reconstructed image', fontsize=20)
                    plt.imshow(output[0,0,:,:].cpu().numpy(), cmap='gray')
                    plt.axis('off')
                    plt.subplot(2, 2, 4)
                    plt.title('Reconstructed Mask', fontsize=20)
                    plt.imshow(np.clip(output[0,1,:,:].cpu().numpy(), 0.,1.), cmap='gray')
                    plt.axis('off')

                ## plot the image, latent space and the reconstructed image
                plot_image_latent(im, encoded_output)
                plt.show()

            ## calculate the perceptual loss
            encoded_output = encoded_output[0,:,:,:].flatten()
            encoded_output_list.append(encoded_output.cpu().numpy())

    encoded_output_list = np.array(encoded_output_list)

    # Reduce dimensionality with PCA
    logging.info("PCA reduction...")
    pca = PCA(n_components=3)
    encoded_output_pca = pca.fit_transform(encoded_output_list)
    encoded_output_pca = (encoded_output_pca - np.min(encoded_output_pca)) / (np.max(encoded_output_pca) - np.min(encoded_output_pca))
    explained_variance_ratio = pca.explained_variance_ratio_
    logging.info(f"Explained Variance Ratio PCA: {explained_variance_ratio}\n")

    ## reduce the dimensionality of the latent space with tsne
    logging.info("TSNE reduction...")
    tsne = TSNE(n_components=2, random_state=0, perplexity=3)
    encoded_output_tsne = tsne.fit_transform(encoded_output_list)
    encoded_output_tsne = (encoded_output_tsne - np.min(encoded_output_tsne)) / (np.max(encoded_output_tsne) - np.min(encoded_output_tsne))
    logging.info(f'tsne_shape {encoded_output_tsne}')

    ######################## PLOT #################################################################
    # plot the 3D scatter plot of each point
    fig = plt.figure(figsize=(10,10), num=f'{type_model} - 3D PCA of latent space of PLAX')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(encoded_output_pca[:,0], encoded_output_pca[:,1], encoded_output_pca[:,2])
    ax.set_xlabel('PCA 1', fontsize=16)
    ax.set_ylabel('PCA 2', fontsize=16)
    ax.set_zlabel('PCA 3', fontsize=16)
    ax.tick_params(labelsize=14)


    # plot the 2D scatter plot of each point
    plt.figure(figsize=(8,8), num=f'{type_model} - PCA of latent space of PLAX')
    plt.scatter(encoded_output_pca[:,0], encoded_output_pca[:,1])
    plt.xlabel('PCA 1', fontsize=20)
    plt.ylabel('PCA 2', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    ## plt the 2d scatter plot of tsne
    plt.figure(figsize=(10,10), num=f'{type_model} - TSNE of latent space of PLAX')
    plt.scatter(encoded_output_tsne[:,0], encoded_output_tsne[:,1], color='blue', s=40)
    plt.xlabel('TSNE 1', fontsize=22)
    plt.ylabel('TSNE 2', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()


    ##################################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Invastigate the latent space')
    parser.add_argument('--save_folder', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model",
                                                   help='folder to save the model')
    parser.add_argument('--type_image', type=str, default='ius', help='type of image to investigate, ius or mask')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--show_plot', action='store_true', help="show the latent space imgs, default=False")
    parser.add_argument('--log', type=str, default='debug', help='Logging level')
    args = parser.parse_args()

    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    experiment_dir = os.path.join(args.save_folder, args.type_image, args.trial)
    if 'vae' in os.listdir(experiment_dir): config = os.path.join(experiment_dir, 'vae', 'config.yaml')
    if 'vqvae' in os.listdir(experiment_dir): config = os.path.join(experiment_dir, 'vqvae', 'config.yaml')


    infer(par_dir = os.path.join(args.save_folder, args.type_image), conf=config, trial=args.trial, type_image=args.type_image, show_plot=args.show_plot)
    plt.show()
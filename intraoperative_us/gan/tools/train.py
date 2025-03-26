
"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
import logging
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import torch
from torch.utils.data import DataLoader
import numpy as np

from intraoperative_us.gan.options.train_options import TrainOptions
from intraoperative_us.diffusion.dataset.dataset import IntraoperativeUS
from intraoperative_us.gan.models import create_model
from intraoperative_us.gan.utils.visualizer import Visualizer

def visualize_results(model, epoch, save_folder):
    """
    Visualize the results of the model
    """
    img = model.real_B[0].cpu().detach()
    img = torch.clamp(img, -1., 1.).numpy().transpose(1, 2, 0)
    img = (img + 1) / 2

    gen_img = model.fake_B[0].cpu().detach()
    gen_img = torch.clamp(gen_img, -1., 1.).numpy().transpose(1, 2, 0)
    gen_img = (gen_img + 1) / 2

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), num=f'Results_epch_{epoch}', tight_layout=True)
    ax[0].set_title('Tumor mask ', fontsize=28)
    ax[0].imshow(model.real_A[0].cpu().detach().numpy().transpose(1, 2, 0), cmap='gray')
    ax[0].axis('off')
    ax[1].set_title('Real Image', fontsize=28)
    ax[1].imshow(img, cmap='gray')
    ax[1].axis('off')
    ax[2].set_title('Generated Image', fontsize=28)
    ax[2].imshow(gen_img, cmap='gray')
    ax[2].axis('off')

    plt.savefig(os.path.join(opt.checkpoints_dir, opt.name, 'web', f'epoch_{epoch}.png'))

if __name__ == '__main__':
    ## Training Configuration
    opt = TrainOptions().parse()

    ## Load dataset
    # Load the dataset
    condition_config = {'condition_types': 'image',
                        'image_condition_config': {'image_condition_input_channels': 1,  'image_condition_output_channels': 3,  ## for gan is useless
                                                   'image_condition_h': 256, 'image_condition_w': 256}}
    data_img = IntraoperativeUS(size= [opt.crop_size, opt.crop_size],
                               dataset_path = opt.dataset_path,
                               im_channels = opt.input_nc,
                               splitting_json=opt.splitting_json,
                               split='train',
                               splitting_seed=opt.splitting_seed,
                               train_percentage=opt.train_percentage,
                               val_percentage=opt.val_percentage,
                               test_percentage=opt.test_percentage,
                               condition_config=condition_config,
                               data_augmentation=True
                               )

    dataset = DataLoader(data_img, batch_size=opt.batch_size, shuffle=True, num_workers=8)
    dataset_size = len(data_img)
    logging.info(f'len of the dataset: {dataset_size}')

    ## Create visualizer
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots 

    ## Create model
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    save_folder = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_folder_images = os.path.join(save_folder, 'web')
    if not os.path.exists(save_folder_images):
        os.makedirs(save_folder_images)
    save_folder_models = os.path.join(save_folder, 'models')
    if not os.path.exists(save_folder_models):
        os.makedirs(save_folder_models)
    
    losses_dict = {'G_GAN': [], 'G_L1': [], 'D_real': [], 'D_fake': []}
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        #self.optimizers[0].param_groups[0]['lr']

        progress_bar = tqdm(total=dataset_size, disable=False)
        progress_bar.set_description(f"Epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay}")
        g_gan, g_l1, d_real, d_fake = [], [], [], [] 
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % (opt.display_freq * dataset_size) == 0:   # display images on visdom and save images to a HTML file
                visualize_results(model, epoch, save_folder_images)
             
            progress_bar.update(1)
            losses = model.get_current_losses()
            g_gan.append(losses['G_GAN'])
            g_l1.append(losses['G_L1'])
            d_real.append(losses['D_real'])
            d_fake.append(losses['D_fake'])
            logs = {"G_GAN": losses['G_GAN'], "D_real": losses['D_real'], "D_fake": losses['D_fake'], "lr": model.optimizers[0].param_groups[0]['lr']}
            progress_bar.set_postfix(**logs)

        losses_dict['G_GAN'].append(np.mean(g_gan))
        losses_dict['G_L1'].append(np.mean(g_l1))
        losses_dict['D_real'].append(np.mean(d_real))
        losses_dict['D_fake'].append(np.mean(d_fake))
        
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            torch.save(model.netG.state_dict(), os.path.join(save_folder_models, f'netG_{epoch}.pth'))
            torch.save(model.netD.state_dict(), os.path.join(save_folder_models, f'netD_{epoch}.pth'))

    with open(os.path.join(save_folder, 'losses.json'), 'w') as f:
        json.dump(losses_dict, f, indent=4)
    logging.info(f"Training finished")
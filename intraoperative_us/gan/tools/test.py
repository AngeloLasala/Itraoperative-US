"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
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
import cv2

from intraoperative_us.gan.options.train_options import TrainOptions
from intraoperative_us.gan.options.test_options import TestOptions
from intraoperative_us.gan.models import create_model
from intraoperative_us.diffusion.dataset.dataset import IntraoperativeUS, GeneratedMaskDataset


if __name__ == '__main__':
    ## Test parameters
    opt = TrainOptions().parse()  # get test options

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    ## Load dataset - Tumor mask dataset for generating iUS images
    data_img = GeneratedMaskDataset(par_dir = opt.dataset_path,
                                    size=[opt.crop_size, opt.crop_size],
                                    input_channels=opt.input_nc)
    dataset = DataLoader(data_img, batch_size=1, shuffle=False, num_workers=8)
    logging.info(f"Dataset loaded with {len(dataset)} generated masks")

    ## Load model
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    ## load pretrained model manually
    model.netG.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'models', f'netG_{opt.epoch}.pth')))
    model.netG.eval()

    save_folder = os.path.join(opt.checkpoints_dir, opt.name, 'w_-1.0', f'samples_ep_{opt.epoch}')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    mask_folder = os.path.join(save_folder, 'masks')
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
    ius_folder = os.path.join(save_folder, 'ius')
    if not os.path.exists(ius_folder):
        os.makedirs(ius_folder)

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    # if opt.eval:
    #     model.eval()
    for i, data in tqdm(enumerate(dataset)):
        data = data.to(model.device)
        fake = model.netG(data)           

        ims = torch.clamp(fake, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        mask = data.detach().cpu()

        cv2.imwrite(os.path.join(ius_folder, f'x0_{i}.png'), ims[0].numpy()[0]*255)
        cv2.imwrite(os.path.join(mask_folder, f'mask_{i}.png'), mask[0].numpy()[0]*255)
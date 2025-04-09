"""
This code provide an application of MedSAM for automatica extrapolate the segmentation maps of tumour 
from real and genereted image.
MedSAM: https://github.com/bowang-lab/MedSAM/tree/main?tab=readme-ov-file

The reasoning is to compare the ERROR of segmatation with MedSAM on real images and the generated images 
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import logging
import yaml

from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
from torch.utils.data import DataLoader
import surface_distance
from surface_distance import metrics

from intraoperative_us.diffusion.evaluation.investigate_vae import get_config_value, get_best_model
from intraoperative_us.diffusion.dataset.dataset import IntraoperativeUS, GenerateDataset
from intraoperative_us.diffusion.utils.utils import get_best_model, load_autoencoder, get_number_parameter

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

def get_bbox_from_mask(mask):
    """
    Get the bounding box [x0, y0, h, w] from the binary mask
    """
    y, x = np.where(mask > 0)
    ##
    y0, x0 = np.min(y), np.min(x)
    y1, x1 = np.max(y), np.max(x)
    return np.array([x0, y0, x1, y1])

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    """
    Perform inference using the MedSAM model.
    Code from oginal repo of MedSAM
    """
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :] # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False,
        )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

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

def infer(par_dir, conf, trial, experiment, epoch, guide_w, scheduler, 
          medsam_path,
          show_gen_mask, device):
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


    data_gen = GenerateDataset(par_dir, trial, experiment, guide_w, scheduler, epoch,
                               size=[dataset_config['im_size_h'], dataset_config['im_size_w']], input_channels=dataset_config['im_channels'],
                               mask=True)
    data_loader_gen = DataLoader(data_gen, batch_size=1, shuffle=False, num_workers=8)
    logging.info(f'len gen data {len(data_gen)}')

    medsam_model = sam_model_registry['vit_b'](checkpoint=medsam_path)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()

    logging.info('Real images...')
    real_dsc, real_hausdorff = [], []
    for i, data in enumerate(data_loader):
        img = data[0].squeeze(0).permute(1, 2, 0).cpu().numpy()  # shape: (H, W, C)
        W, H = img.shape[0], img.shape[1]
        img_resized = transform.resize(img, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.float32)  
        img_resized = (img_resized - img_resized.min()) / np.clip(img_resized.max() - img_resized.min(), a_min=1e-8, a_max=None)
        img_tensor = torch.tensor(img_resized).permute(2, 0, 1).unsqueeze(0).repeat(1,3,1,1).to(device)

        mask = data[1]['image']
        if mask[0, 0].sum() > 0:
            bbox = np.array([get_bbox_from_mask(mask[0, 0].cpu().numpy())])
            box_1024 = bbox / np.array([W, H, W, H]) * 1024
    
            with torch.no_grad():
                image_embedding = medsam_model.image_encoder(img_tensor)

            
            medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)

            dsc, hausdorff = compute_metrics(mask[0, 0].detach().cpu().numpy(),
                                                medsam_seg)
            real_dsc.append(dsc)
            real_hausdorff.append(hausdorff)
        
        else:
            print(f'Mask {i} is empty')

        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[0].imshow(img, cmap='gray')
        # show_box(bbox[0], ax[0])
        # ax[0].set_title("Input Image and Bounding Box")
        # ax[0].axis('off')
        # ax[1].imshow(img, cmap='gray')
        # show_mask(medsam_seg, ax[1])
        # show_mask(mask[0, 0].cpu().numpy(), ax[1], random_color=True)
        # show_box(bbox[0], ax[1])
        # ax[1].axis('off')
        # ax[1].set_title("MedSAM vs Manual Segmentation")
        # plt.show()

    print()
    ## Generated image
    logging.info('Generated images...')
    gen_dsc, gen_hausdorff = [], []
    for i, data in enumerate(data_loader_gen):
        
        img = data[0].squeeze(0).permute(1, 2, 0).cpu().numpy()  # shape: (H, W, C)
        W, H = img.shape[0], img.shape[1]
        img_resized = transform.resize(img, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.float32)  
        img_resized = (img_resized - img_resized.min()) / np.clip(img_resized.max() - img_resized.min(), a_min=1e-8, a_max=None)
        img_tensor = torch.tensor(img_resized).permute(2, 0, 1).unsqueeze(0).repeat(1,3,1,1).to(device)
        
        mask = data[1]
        if mask[0, 0].sum() > 0:
            bbox = np.array([get_bbox_from_mask(mask[0, 0].cpu().numpy())])
            box_1024 = bbox / np.array([W, H, W, H]) * 1024

            with torch.no_grad():
                image_embedding = medsam_model.image_encoder(img_tensor)

            
            medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)

            dsc, hausdorff = compute_metrics(mask[0, 0].detach().cpu().numpy(),
                                                medsam_seg)
            gen_dsc.append(dsc)
            gen_hausdorff.append(hausdorff)
        else:
            print(f'Mask {i} is empty')

        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[0].imshow(img, cmap='gray')
        # show_box(bbox[0], ax[0])
        # ax[0].set_title("Input Image and Bounding Box")
        # ax[0].axis('off')
        # ax[1].imshow(img, cmap='gray')
        # show_mask(medsam_seg, ax[1])
        # show_mask(mask[0, 0].cpu().numpy(), ax[1], random_color=True)
        # show_box(bbox[0], ax[1])
        # ax[1].axis('off')
        # ax[1].set_title("MedSAM vs Manual Segmentation")
        # plt.show()


    ## Statistics
    real_dsc = np.array(real_dsc)
    real_hausdorff = np.array(real_hausdorff)
    gen_dsc = np.array(gen_dsc)
    gen_hausdorff = np.array(gen_hausdorff)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), num='Boxplot SamMED')
    ax[0].boxplot([real_dsc, gen_dsc], labels=['Real', 'Generated'])
    ax[0].set_title('DSC')

    ax[1].boxplot([real_hausdorff, gen_hausdorff], labels=['Real', 'Generated'])
    ax[1].set_title('Hausdorff')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anatomical evaluation of echogenicity')
    parser.add_argument('--save_folder', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model",
                                                   help='folder to save the model')
    parser.add_argument('--type_image', type=str, default='ius', help='type of image to evaluate, ius or mask')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the 
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--guide_w', type=float, default=0.0, help='guide_w for the conditional model, w=-1 [unconditional], w=0 [vanilla conditioning], w>0 [guided conditional]')
    parser.add_argument('--scheduler', type=str, default='ddpm', help='sheduler used for sampling, i.e. ddpm, pndm')
    parser.add_argument('--epoch', type=int, default=99, help='epoch to sample, this is the epoch of cond ldm model') 
    parser.add_argument('--medsam_path', type=str, default='/home/angelo/Documenti/MedSAM/work_dir/MedSAM/medsam_vit_b.pth', help='path to the MedSAM model')
    parser.add_argument('--show_gen_mask', action='store_true', help="show the generative and mask images, default=False")
    parser.add_argument('--log', type=str, default='debug', help='Logging level')
    args = parser.parse_args()

    # Check if the device is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Am I using GPU? {torch.cuda.is_available()}')

     ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    experiment_dir = os.path.join(args.save_folder, args.type_image, args.trial)
    if 'vae' in os.listdir(experiment_dir): config = os.path.join(experiment_dir, 'vae', 'config.yaml')

    infer(par_dir = os.path.join(args.save_folder, args.type_image), conf=config, trial=args.trial, 
         experiment=args.experiment, epoch=args.epoch, guide_w=args.guide_w, scheduler=args.scheduler,
         medsam_path=args.medsam_path, show_gen_mask=args.show_gen_mask, device=device)
    plt.show()
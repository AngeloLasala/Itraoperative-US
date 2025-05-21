"""
Implemented the Kernel Inseption Distances (KID) metric for evaluating the quality of generated images.
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
from PIL import Image
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.inception import InceptionV3
from scipy.spatial.distance import pdist, cdist, squareform
from torch import randint
from torchmetrics.image.kid import KernelInceptionDistance

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = img.resize((299, 299))
        if self.transform:
            img = self.transform(img)
        return img

def get_data(path, num_workers=4):
    """
    Get the data 
    """

    ## Get Dataset    
    files = []
    for p in path:
        p_path = pathlib.Path(p)
        file_list = sorted(
            [file for ext in {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"} for file in p_path.glob("*.{}".format(ext))]
        )
        files += file_list
    
    ds = ImagePathDataset(files, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=len(ds),
                                         shuffle=False, num_workers=num_workers)
    data = [i for i in loader]                                    
    return data[0]


def compute_kid(paths, batch_size, device, dims, num_workers):
    """Compute KID between two lists of image paths."""
    assert len(paths) == 2, "Need [real_paths, fake_paths]"
    for p in paths[0] + paths[1]:
        if not os.path.exists(p):
            raise RuntimeError(f"Invalid path: {p}")

    kid = KernelInceptionDistance(subset_size=50, normalize=True)
    
    real_data = get_data(paths[0], num_workers=num_workers)
    fake_data = get_data(paths[1], num_workers=num_workers)
    kid.update(real_data, real=True)
    kid.update(fake_data, real=False)
    kid_values = kid.compute() 
    return kid_values[0], kid_values[1]

def kid_experiment(conf, experiment_dir, batch_size=50, device=None, dims=2048, num_workers=4):
    """Compute KID across epochs similarly to fid_experiment."""
    folder_list = [folder for folder in os.listdir(experiment_dir) if folder.startswith("samples_ep")]
    epoch_list = [int(folder.split('_')[-1]) for folder in folder_list]
    epoch_list.sort()

    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    dataset_config = config['dataset_params']
    dataset_path = dataset_config['dataset_path']
    splitting_json = dataset_config['splitting_json']
    
    data_real = []
    with open(os.path.join(os.path.dirname(dataset_path), splitting_json), 'r') as file:
        splitting_dict = json.load(file)
    subjects_files = splitting_dict['train']
    data_real = [os.path.join(dataset_path, subject, 'volume') for subject in subjects_files]
    
    kid_vals = {}
    for ep in epoch_list:
        fake_dir = os.path.join(experiment_dir, f'samples_ep_{ep}', 'ius')
        kid_mean, kid_std = compute_kid([data_real, [fake_dir]], batch_size, device, dims, num_workers)
        kid_vals[ep] = (kid_mean, kid_std)
    return kid_vals

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute KID score.")
    parser.add_argument('--save_folder', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model",
                                                   help='folder to save the model')
    parser.add_argument('--type_image', type=str, default='ius', help='type of image to evaluate, ius or mask')
    parser.add_argument('--split', type=str, default='split_1', help='splitting name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the 
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--guide_w', type=float, default=0.0, help='guide_w for the conditional model, w=-1 [unconditional], w=0 [vanilla conditioning], w>0 [guided conditional]')
    parser.add_argument('--scheduler', type=str, default='ddpm', help='sheduler used for sampling, i.e. ddpm, pndm')
    parser.add_argument('--show_plot', action='store_true', help="show and save the FID plot, default=False")
    parser.add_argument('--log', type=str, default='info', help='Logging level')
    args = parser.parse_args()

    print('Am I using GPU: ', torch.cuda.is_available())
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    experiment_dir = os.path.join(args.save_folder, args.type_image, args.trial, args.split, args.experiment)
    config = os.path.join(experiment_dir, 'config.yaml')
    experiment_dir_w = os.path.join(experiment_dir, f'w_{args.guide_w}', args.scheduler)

    kid_scores = kid_experiment(config, experiment_dir_w, batch_size=50, device=device)

    for ep, (mean, std) in kid_scores.items():
        print(f"Epoch {ep}: KID = {mean:.6f} ± {std:.6f}")

    ## save the KID score
    with open(os.path.join(experiment_dir, f'w_{args.guide_w}', args.scheduler, f'KID_score_{args.scheduler}.txt'), 'w') as f:
        for ep, (mean, std) in kid_scores.items():
            f.write(f"Epoch {ep}: KID = {mean:.6f} ± {std:.6f}\n")

    if args.show_plot:
        import matplotlib.pyplot as plt
        epochs = list(kid_scores.keys())
        means = [v[0] for v in kid_scores.values()]
        stds = [v[1] for v in kid_scores.values()]
        plt.errorbar(epochs, means, yerr=stds, fmt='-o')
        plt.xlabel('Epoch')
        plt.ylabel('KID')
        plt.title('Kernel Inception Distance over Epochs')
        plt.grid(True)
        plt.savefig(os.path.join(experiment_dir, f'w_{args.guide_w}', args.scheduler, f'KID_plot_{args.scheduler}.png'))
        # plt.show()

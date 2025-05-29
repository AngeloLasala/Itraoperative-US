"""
Create the folders sctructure to train nnUnet model for real and real_and gen segmentation experiment
"""
import glob
import os
import yaml
import argparse
import logging

from PIL import Image
from tqdm import tqdm
import json
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import re
from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import tqdm

def extract_case_number(case_str):
    match = re.match(r"Case(\d+)", case_str)
    if match:
        return int(match.group(1))
    return None

def extract_slice_number(filename):
    match = re.search(r'_(\d+)\.png$', filename)
    if match:
        return int(match.group(1))
    return None


def main(args):
    """
    Create the nnUNet dataset structure from the original dataset
    """
    ## create save folders
    nnUnet_raw = os.path.join(args.dataset_path, 'nnUNet_raw')
    
    # here I read only yht split 0 because train+val are equal acroos the splitting
    count = len(os.listdir(nnUnet_raw))
    json_path = os.path.join(args.dataset_path, f'splitting_0.json')
    count += 1
    dataset_name = f"Dataset{count:03d}_{args.name_dataset}"

    ## create the folder
    dataset_path = os.path.join(nnUnet_raw, dataset_name)
    os.makedirs(dataset_path, exist_ok=True)
    train_path = os.path.join(dataset_path, 'imagesTr')
    labels_path = os.path.join(dataset_path, 'labelsTr')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    ## create test folders
    test_path = os.path.join(dataset_path, 'imagesTs')
    test_labels_path = os.path.join(dataset_path, 'labelsTs')
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(test_labels_path, exist_ok=True)
            
    with open(json_path, 'r') as f:
        split_info = json.load(f)

    n_train, ntest = 0, 0
    logging.info(f"Dataset Real...")
    for key in split_info.keys():
        for sbj in split_info[key]:
            subject_id = extract_case_number(sbj.split('_')[0])
            subject_path = os.path.join(args.dataset_path, 'dataset', sbj)
            image_path = os.path.join(subject_path, 'volume')
            mask_path = os.path.join(subject_path, 'tumor')

            # read images and masks
            images = sorted(glob.glob(os.path.join(image_path, '*.png')))
            masks = sorted(glob.glob(os.path.join(mask_path, '*.png')))

            for i, j in zip(images, masks):
                image_slice = extract_slice_number(i.split('/')[-1])
                mask_slice = extract_slice_number(j.split('/')[-1])
                if image_slice != mask_slice:
                    raise ValueError(f"Slice mismatch: image '{i}' has slice {image_slice}, but mask '{j}' has slice {mask_slice}")

                image_name = f"Case{subject_id:03d}_{image_slice:03d}_0000.png"
                mask_name = f"Case{subject_id:03d}_{mask_slice:03d}.png"

                image = Image.open(i).convert('L')
                mask = Image.open(j).convert('L')
                image = image.resize((256, 256), Image.BILINEAR)
                mask = mask.resize((256, 256), Image.NEAREST)
                mask_np = np.array(mask)
                mask_np[mask_np > 125] = 1
                mask = Image.fromarray(mask_np.astype(np.uint8))

                if key == 'train' or key == 'val':
                    image.save(os.path.join(train_path, image_name))
                    mask.save(os.path.join(labels_path, mask_name))
                    n_train += 1

                elif key == 'test':
                    image.save(os.path.join(test_path, image_name))
                    mask.save(os.path.join(test_labels_path, mask_name))
                    ntest += 1

    if args.name_dataset != 'Real':
        # raise and error if thhe name doen not start with RealGen
        if not args.name_dataset.startswith('RealGenx'):
            raise ValueError("Dataset name must start with 'RealGenx' for generated datasets.")

        real_dataset_len = {'split_0': 353, 'split_1': 354, 'split_2': 345, 'split_3': 344, 'split_4': 350}
        epochs_dict = {'split_0': 5000, 'split_1': 6000, 'split_2': 5000, 'split_3': 5000, 'split_4': 5000}
        degrees_of_augmentation = int(args.name_dataset.split('x')[-1])

        logging.info(f"Dataset RealGenx{degrees_of_augmentation}...")
        for split in real_dataset_len.keys():  
            num_of_images = real_dataset_len[split] * degrees_of_augmentation
            data_ius = os.path.join(args.save_folder, 'ius', args.trial, split, args.experiment, 
                              f'w_3.0_all', 'ddpm', f'samples_ep_{epochs_dict[split]}')

            images_path = [os.path.join(data_ius, 'ius', f'x0_{i}.png') for i in range(num_of_images)]
            masks_path = [os.path.join(data_ius, 'masks', f'mask_{i}.png') for i in range(num_of_images)]

            for i, j in zip(images_path, masks_path):
                nn = int(i.split("/")[-1].split('.')[0].split('_')[-1])
                tt = int(j.split("/")[-1].split('.')[0].split('_')[-1])
                if nn != tt:
                    raise ValueError(f"Image {i} and mask {j} do not match in naming convention.")
            
                image_name = f"CaseG{split.split('_')[-1]}_{nn:04d}_0000.png"
                mask_name = f"CaseG{split.split('_')[-1]}_{nn:04d}.png"

                

                image = Image.open(os.path.join(data_ius, i)).convert('L')
                mask = Image.open(os.path.join(data_ius, j)).convert('L')
                image = image.resize((256, 256), Image.BILINEAR)
                mask = mask.resize((256, 256), Image.NEAREST)

                # convert to numpy array image
                image_np = np.array(image)
                mask_np = np.array(mask)
                mask_np[mask_np > 125] = 1
                mask = Image.fromarray(mask_np.astype(np.uint8))

                image.save(os.path.join(train_path, image_name))
                mask.save(os.path.join(labels_path, mask_name))
                n_train += 1


    generate_dataset_json(output_folder=os.path.join(nnUnet_raw, dataset_name),
                            channel_names={0: "L"},
                            labels={"background": 0, "tumor": 1},
                            num_training_cases=n_train,
                            file_ending=".png",
                            dataset_name=dataset_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create nnUNet dataset')
    parser.add_argument('--dataset_path', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/RESECT_iUS_dataset",
                         help='Path to the original dataset')
    parser.add_argument('--name_dataset', type=str, default='Real', help='Name of the create dataset, i.e Dataset001_#--name_dataset')
    parser.add_argument('--save_folder', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model", help='folder to save the model, default = trained_model')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the 
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--log', type=str, default='info', help='Logging level')
    args = parser.parse_args()

    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    main(args)



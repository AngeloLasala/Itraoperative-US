"""
Create the folders sctructure to train nnUnet model for real and real_and gen segmentation experiment
"""
import glob
import os
import yaml
import argparse

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
    split_list = [0]         # list of CV splitting

    ## create save folders
    nnUnet_raw = os.path.join(args.dataset_path, 'nnUNet_raw')

    count = 0
    for split in split_list:
        json_path = os.path.join(args.dataset_path, f'splitting_{split}.json')
        count += 1
        dataset_name = f"Dataset{count:03d}_Real"

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
        for key in split_info.keys():
            print(f"Split {split} - Key: {key}")
            for sbj in split_info[key]:
                subject_id = extract_case_number(sbj.split('_')[0])
                subject_path = os.path.join(args.dataset_path, 'dataset', sbj)
                image_path = os.path.join(subject_path, 'volume')
                mask_path = os.path.join(subject_path, 'tumor')
                print(f"  Subject {sbj}: images = {len(os.listdir(image_path))}, masks = {len(os.listdir(mask_path))}")

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
    args = parser.parse_args()

    main(args)



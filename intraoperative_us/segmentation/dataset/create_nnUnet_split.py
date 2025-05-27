"""
Create the json file form manual split 
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
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

def get_dataset_name_from_number(d):
    """
    Get the dataset name from the dataset number
    """
    # get the list of dataset in nnUNet_preprocessed
    dataset_list = os.listdir(nnUNet_preprocessed)
    # find the string in the list that sart with 'Dataset001'
    for dataset in dataset_list:
        if dataset.startswith(f'Dataset{d:03d}_'):
            return dataset
        else:
            raise ValueError(f"Dataset {d} not found in nnUNet_preprocessed folder. Available datasets: {dataset_list}")

def main(args):
    """
    Create the json for splitting
    """
    dataset_name = get_dataset_name_from_number(args.d)
    print(dataset_name)
    exit()

    # read the json file with splitting
    for split in [0, 1, 2, 3, 4]:
        json_path = os.path.join(args.dataset_path, f'splitting_{split}.json')
        with open(json_path, 'r') as f:
            split_data = json.load(f)

        print(split_data.keys())
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create nnUNet dataset structure from manual split")
    parser.add_argument('--dataset_path', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/RESECT_iUS_dataset", help='Path to the dataset')
    parser.add_argument('-d', type=int, default=1, help='number of the dataset')
    args = parser.parse_args()
    
    main(args)  


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
from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

def main(args):
    """
    Create the json for splitting
    """
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
    args = parser.parse_args()
    
    main(args)  


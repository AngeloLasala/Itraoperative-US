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

from intraoperative_us.segmentation.dataset.create_nnUnet_dataset import extract_case_number, extract_slice_number

def get_dataset_name_from_number(d):
    """
    Get the dataset name from the dataset number
    """
    # get the list of dataset in nnUNet_preprocessed
    dataset_list = os.listdir(nnUNet_preprocessed)
    # find the string in the list that sart with 'Dataset001'
    dataset_return = None
    for dataset in dataset_list:
        if dataset.startswith(f'Dataset{d:03d}_'):
            dataset_return = dataset

    if dataset_return is None:
        raise ValueError(f"Dataset {d} not found in {nnUNet_preprocessed}")

    return dataset_return

def filter_cases_by_ids(filenames, case_ids):
    """
    Filter filenames to include only those that start with specified case IDs.
    """
    prefixes = [f"Case{case_id:03d}_" for case_id in case_ids]
    return [filename.replace('_0000.png', '') for filename in filenames if any(filename.startswith(prefix) for prefix in prefixes)]

def filter_casesG_by_split(filenames, split):
    """
    Filter filenames to include only those that start with specified split.
    """
    return [filename.replace('_0000.png', '') for filename in filenames if filename.startswith(f'CaseG{split}')]

def main(args):
    """
    Create the json for splitting
    """
    dataset_name = get_dataset_name_from_number(args.d)

    list_dataset_raw = os.listdir(os.path.join(nnUNet_raw, dataset_name, 'imagesTr'))

    # read the json file with splitting
    split_json_list = []
    for split in [0, 1, 2, 3, 4]:
        split_json = {}
        json_path = os.path.join(args.dataset_path, f'splitting_{split}.json')
        with open(json_path, 'r') as f:
            split_data = json.load(f)

        train_sbj = [extract_case_number(i) for i in split_data['train']]
        val_sbj = [extract_case_number(i) for i in split_data['val']]

        train_sbj_list = filter_cases_by_ids(list_dataset_raw, train_sbj)
        val_sbj_list = filter_cases_by_ids(list_dataset_raw, val_sbj)
        
        split_json['train'] = train_sbj_list
        split_json['val'] = val_sbj_list
        split_json_list.append(split_json)

        if dataset_name != 'Dataset001_Real':
            train_sbj_list_gen = filter_casesG_by_split(list_dataset_raw, split)
            split_json['train'] += train_sbj_list_gen
            
    ## save the list with json
    with open(os.path.join(nnUNet_preprocessed, dataset_name, 'splits_final.json'), 'w') as f:
        json.dump(split_json_list, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create nnUNet dataset structure from manual split")
    parser.add_argument('--dataset_path', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/RESECT_iUS_dataset", help='Path to the dataset')
    parser.add_argument('-d', type=int, default=1, help='number of the dataset')
    args = parser.parse_args()
    
    main(args)  


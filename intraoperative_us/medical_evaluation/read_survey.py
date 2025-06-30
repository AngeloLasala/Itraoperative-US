"""
Read the syrvey and return the CM
"""
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import json

def main(args):
    """
    Read the survey and compute the confusion matrix
    """
    ## read the gt joson file
    gt_file = os.path.join(args.dict_path, args.gt_file)
    with open(gt_file, 'r') as f:
        gt_dict = json.load(f)
    print(f"Ground truth file: {gt_file}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read survey and compute confusion matrix')
    parser.add_argument('--dict_path', type=str, defaut="/home/angelo/Documenti/Itraoperative-US/intraoperative_us/medical_evaluation/form_images/dict", help='Path to the dict folder')
    parser.add_argument('--gt_file', type=str, required=True, help='file to the ground true file')
    parser.add_argument('--answer_file', type=str, required=True, help='Path to the answers CSV file')
    args = parser.parse_args()

    main(args)

   
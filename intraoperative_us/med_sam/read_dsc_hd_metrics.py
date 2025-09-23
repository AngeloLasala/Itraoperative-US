"""
File for reading the Alligment pre-computed alligment metrics
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import yaml
import pandas as pd
import seaborn as sns
import cv2


def main(metrics_folder, experiment_list):
    """
    Main function to read the Alligment pre-computed alligment metrics
    
    Paramerters
    ----------
    metrics_folder : str
        Path to the folder containing the metrics

    experiment_list : list
        List of experiments to be compared
    """
    

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read the Alligment pre-computed alligment metrics")
    parser.add_argument("--metrics_folder", type=str, default="DSC_HD_metrics", help="folder of the metrics")
    args = parser.parse_args()

    experiment_list = [ 'pix2pix', 'proposed', 'Controlnet', 'onestep']

    main(args.metrics_folder, experiment_list)
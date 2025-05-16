"""
File for reading the Segmentation performance
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


def main(save_folder, trial_list, split_list, experiment_list, dataset_split):
    """
    Main function to read the Alligment pre-computed alligment metrics
    
    Paramerters
    ----------
    save_folder : str
        Path to the folder containing the metrics
    experiment_list : list
        List of experiments to be compared
    trial_list : list
        List of trials to be compared
    dataset_split : str
        Dataset split to be used for testing, val or test
    """
    
    for trial in trial_list:
        print(f"Trial: {trial}")
        for split in split_list:
            print(f"Split: {split} - {dataset_split}")
            fig, ax = plt.subplots(1, 2, figsize=(15, 8), tight_layout=True, num=f"Loss {trial} {split}")

            for i, experiment in enumerate(experiment_list):
                dsc = np.load(os.path.join(save_folder, trial, split, experiment, f'{dataset_split}_result', 'dsc.npy'))
                haus = np.load(os.path.join(save_folder, trial, split, experiment, f'{dataset_split}_result', 'hausdorff.npy'))
                haus[haus == np.inf] = 256

                train_loss = np.load(os.path.join(save_folder, trial, split, experiment, 'model', 'train_loss.npy'))
                val_loss = np.load(os.path.join(save_folder, trial, split, experiment, 'model', 'val_loss.npy'))

                # Plot loss
                ax[i].plot(train_loss, label=f"train", lw=2)
                ax[i].plot(val_loss, label=f"val", lw=2)
                # set the grid
                ax[i].grid(linestyle=':')                
                ax[i].set_title(f"{experiment}", fontsize=20)

                ax[i].set_xlabel("Epoch", fontsize=20)
                ax[i].set_ylabel("Loss", fontsize=20)
                ax[i].legend(fontsize=20)
                ax[i].tick_params(axis='x', labelsize=20)
                ax[i].tick_params(axis='y', labelsize=20)
                ax[i].set_ylim(0, 1.0)

                print(f"DSC: {np.mean(dsc):.4f} [{np.quantile(dsc, 0.25):.4f}, {np.quantile(dsc, 0.75):.4f}]")
                print(f"Hausdorff: {np.mean(haus):.4f} [{np.quantile(haus, 0.25):.4f}, {np.quantile(haus, 0.75):.4f}]")
            print()
            plt.show()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read the Alligment pre-computed alligment metrics")
    parser.add_argument('--save_folder', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model/segmentation", help='folder to save the model, default = trained_model')
    parser.add_argument('--dataset_split', type=str, default='val', help='dataset split to use for testing, val or test')
    args = parser.parse_args()

    save_folder = args.save_folder
    split_list = ['split_0', 'split_1', 'split_2', 'split_3', 'split_4']
    trial_list = ['bce_loss']
    experiment_list = ['only_real', 'real_and_gen']

    main(save_folder, trial_list, split_list, experiment_list, args.dataset_split)
"""
Read the prediction of nnUnet for downstream tasks of segmentation and return the final analysis
"""
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import surface_distance
from surface_distance import metrics
import seaborn as sns
import json

def read_summary(summary_file):
    """
    Read the summary file 

    Parameters
    ----------
    summary : dict
        Dictionary containing the summary of the predictions.
    
    Returns
    -------
    list 
        List of tuples containing the name of the experiment and the summary.
    """
    ## read the summary file with json
    with open(summary_file, 'r') as f:
        summary = json.load(f)

    # metrics per case
    metric_per_case = summary["metric_per_case"]
    print(metric_per_case.keys())  




def main(args):
    """
    Compute the results of prediction of nnUnet and ground truth.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments containing the save folder and results folder.
        - save_folder : str
            Folder to save the model, default is "trained_model".
        - results_folder : str
            Folder to save the results, default is "nnUNet_results".
    
    Returns
    -------
    None
        The function saves the results in the specified folder.
        - box plot of prediction and statistical analysis.
    """

    list_experiment = os.listdir(os.path.join(args.save_folder, args.results_folder))
    # loop over the dataset
    for exp in list_experiment:
        name_of_exp = exp.split("_")[-1]
        print(f"Experiment: {name_of_exp}")
        
        prediction_path = os.path.join(args.save_folder, args.results_folder, exp,
                                       "nnUNetTrainer__nnUNetPlans__2d", "crossval_results_folds_0_1_2_3_4")

        ## read summary file
        summary_output = read_summary(os.path.join(prediction_path, "summary.json"))
        summary_file = os.path.join(prediction_path, "summary.json")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read the prediction of nnUnet for downstream tasks of segmentation")
    parser.add_argument('--save_folder', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model", help='folder to save the model, default = trained_model')
    parser.add_argument('--results_folder', type=str, default="nnUNet_results", help='folder to save the results, default = nnUNet_results')

    args = parser.parse_args()
    
    main(args)

    



# path = "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model/nnUNet_results/Dataset001_Real/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation"
# print(os.listdir(path))
# img = Image.open(os.path.join(path, "Case015_232.png"))
# # convert to numpy array
# img = np.array(img)
# print(img.shape)
# print(np.max(img))
# plt.figure(figsize=(10, 10))
# plt.imshow(img, cmap='gray')
# plt.axis('off')
# plt.show()

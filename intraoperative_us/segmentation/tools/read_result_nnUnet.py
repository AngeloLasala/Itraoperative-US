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
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
import json

def read_inference(prediction_path):
    """
    Read the preditction on the inferece path and compare with ground truth.

    Parameters
    ----------
    prediction_path : str
        Path to the prediction folder containing the summary file.

    Returns
    -------
    dsc : float
        Dice Similarity Coefficient between the prediction and ground truth.
    
    hausdorff : float
        Hausdorff distance between the prediction and ground truth.
    """
    # read the json with info
    json_file = os.path.join(prediction_path, "predict_from_raw_data_args.json")
    with open(json_file, 'r') as f:
        info_test = json.load(f)
    test_img_path = info_test["list_of_lists_or_source_folder"]
    test_mask_path = os.path.join(os.path.dirname(test_img_path), 'labelsTs')

    # for i in os.listdir(test_img_path):
    #     print(i)
    #     img = Image.open(os.path.join(test_img_path, i))
    #     mask = Image.open(os.path.join(test_mask_path, i.replace("_0000.png", ".png")))
    #     img = np.array(img)
    #     mask = np.array(mask)

    dsc_list, hd_list = [], []
    for pred in os.listdir(prediction_path):
        if pred.endswith(".png"):
            mask_pred = Image.open(os.path.join(prediction_path, pred))
            mask_gt = Image.open(os.path.join(test_mask_path, pred))
            mask_pred = np.array(mask_pred)
            mask_gt = np.array(mask_gt)

            dsc = metrics.compute_dice_coefficient(mask_gt > 0.5, mask_pred > 0.5)
            hausdorff = metrics.compute_robust_hausdorff(metrics.compute_surface_distances(mask_gt > 0.5, mask_pred > 0.5, spacing_mm=(1, 1)), 95)
            dsc_list.append(dsc)
            hd_list.append(hausdorff)   
    print()    
    return dsc_list, hd_list     


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
    experiment_dict = {}
    for exp in list_experiment:
        name_of_exp = exp.split("_")[-1]
        print(f"Experiment: {name_of_exp}")
        prediction_path = os.path.join(args.save_folder, args.results_folder, exp, "inference")

        ## read summary file
        dsc_list, hd_list = read_inference(prediction_path)
        print(f"DSC: {np.mean(dsc_list):.4f} +/- {np.std(dsc_list):.4f}")
        print(f"Hausdorff: {np.mean(hd_list):.4f} +/- {np.std(hd_list):.4f}")
        print()
        experiment_dict[name_of_exp] = {"dsc": dsc_list,"hd": hd_list}

    ## STATISTICAL ANALYSIS
    # Omnibus test
    print("OMNIBUS test:") 
    dsc_scores = [experiment_dict[exp]["dsc"] for exp in experiment_dict]
    friedman_dsc = friedmanchisquare(*dsc_scores)
    print(f"Friedman test DSC: statistic={friedman_dsc.statistic:.4f}, p-value={friedman_dsc.pvalue:.4f}")

    hd_scores = [experiment_dict[exp]["hd"] for exp in experiment_dict]
    friedman_hd = friedmanchisquare(*hd_scores)
    print(f"Friedman test Hausdorff: statistic={friedman_hd.statistic:.4f}, p-value={friedman_hd.pvalue:.4f}")

    if friedman_dsc.pvalue < 0.05:
        print("!! Significant differences !! Friedman test indicates significant differences in DSC scores across experiments.")
    else:
        print("!! NO significant differences !! Friedman test indicates no significant differences in DSC scores across experiments.")
    if friedman_hd.pvalue < 0.05:
        print("!! Significant differences !! Friedman test indicates significant differences in Hausdorff scores across experiments.")
    else:
        print("!! NO significant differences !! Friedman test indicates no significant differences in Hausdorff scores across experiments.")
    print()

    # Pairwise comparisons
    print("POST HOC:")
    real_dsc = experiment_dict["Real"]["dsc"]
    print(f"Real DSC: {np.mean(real_dsc):.4f} +/- {np.std(real_dsc):.4f}")
    dsc_p_values = {}
    for exp in experiment_dict:
        if exp == "Real":
            continue
        generated_dsc = experiment_dict[exp]["dsc"]
        stat, p_value = wilcoxon(real_dsc, generated_dsc)
        dsc_p_values[exp] = p_value
        print(f"Wilcoxon test between Real and {exp} DSC: statistic={stat:.4f}, p-value={p_value:.4f}")
        if p_value < 0.05:
            print(f"!! Significant differences !! Real vs {exp} DSC scores.")
        else:
            print(f"!! NO significant differences !! Real vs {exp} DSC scores.")
        print()
    
    # pairwise hausdorff
    real_hd = experiment_dict["Real"]["hd"]
    print(f"Real Hausdorff: {np.mean(real_hd):.4f} +/- {np.std(real_hd):.4f}")
    hd_p_values = {}
    for exp in experiment_dict:
        if exp == "Real":
            continue
        generated_hd = experiment_dict[exp]["hd"]
        stat, p_value = wilcoxon(real_hd, generated_hd)
        hd_p_values[exp] = p_value
        # print(f"HD: {np.median(real_hd):.4f} - {np.median(generated_hd):.4f} ")
        print(f"Wilcoxon test between Real and {exp} Hausdorff: statistic={stat:.4f}, p-value={p_value:.4f}")
        if p_value < 0.05:
            print(f"!! Significant differences !! Real vs {exp} Hausdorff scores.")
        else:
            print(f"!! NO significant differences !! Real vs {exp} Hausdorff scores.")
        print()

    # Prepare data for plotting
    data = experiment_dict
    plot_data = {
        "Score": [],
        "Metric": [],
        "Type": [],
        "Experiment": []
    }

    for exp_name, exp_data in data.items():
        # DSC scores
        for score in exp_data["dsc"]:
            plot_data["Score"].append(score)
            plot_data["Metric"].append("DSC")
            plot_data["Type"].append("Real" if exp_name == "Real" else "Generated")
            plot_data["Experiment"].append(exp_name)

        # Hausdorff scores
        for score in exp_data["hd"]:
            plot_data["Score"].append(score)
            plot_data["Metric"].append("Hausdorff")
            plot_data["Type"].append("Real" if exp_name == "Real" else "Generated")
            plot_data["Experiment"].append(exp_name)

    df = pd.DataFrame(plot_data)

    # Set style
    sns.set(style="whitegrid")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), tight_layout=True) # Increased figure size for better readability

    # Colors
    dsc_color_real = (65/255, 105/255, 225/255)
    dsc_color_generated = (173/255, 216/255, 230/255)
    haus_color_real = (65/255, 105/255, 225/255)
    haus_color_generated = (173/255, 216/255, 230/255)

    # Common settings for annotations
    y_offset_dsc = 0.02 # Adjust this for vertical position of line/asterisk
    y_offset_hd = 5 # Adjust this for vertical position of line/asterisk
    line_height = 0.01 # Length of the vertical part of the bracket
    text_offset = 0.01 # Vertical offset for the asterisk

    experiments = list(experiment_dict.keys())
    
    # DSC plot
    sns.violinplot(x="Experiment", y="Score", hue="Type", data=df[df["Metric"] == "DSC"],
                ax=axes[0], inner=None, palette={"Real": dsc_color_real, "Generated": dsc_color_generated})
    sns.stripplot(x="Experiment", y="Score", hue="Type", data=df[df["Metric"] == "DSC"],
                ax=axes[0], color="black", size=6, jitter=True, alpha=0.6)
    axes[0].set_title("DSC Scores by Experiment", fontsize=14)
    axes[0].set_ylabel("DSC Score", fontsize=20)
    axes[0].set_xlabel("Experiment Type", fontsize=20)
    axes[0].tick_params(axis='x', rotation=30)
    # Adjust legend to be outside the plot
    # axes[0].legend(title="Data Type", loc='upper left', bbox_to_anchor=(1, 1))

    if friedman_dsc.pvalue < 0.05:
        base_y = df[df["Metric"] == "Hausdorff"]["Score"].max()
        line_height = 5.0      # distanza verticale della linea
        text_offset = 2.0      # distanza dell'asterisco sopra la linea

        for i, exp1 in enumerate(experiments[:-1]):
            for j, exp2 in enumerate(experiments[i+1:], start=i+1):
                if hd_p_values[exp2] < 0.05:
                    y = base_y + (j - i) * line_height

                    # SOLO linea orizzontale da i a j
                    axes[1].plot([i, j], [y, y], color='black', lw=1.5)

                    # Asterisco centrato sopra la linea
                    axes[1].text((i + j) / 2, y + text_offset, '*', ha='center', va='bottom', fontsize=18)

    # Hausdorff plot
    sns.violinplot(x="Experiment", y="Score", hue="Type", data=df[df["Metric"] == "Hausdorff"],
                ax=axes[1], inner=None, palette={"Real": haus_color_real, "Generated": haus_color_generated})
    sns.stripplot(x="Experiment", y="Score", hue="Type", data=df[df["Metric"] == "Hausdorff"],
                ax=axes[1], color="black", size=6, jitter=True, alpha=0.6)
    axes[1].set_title("Hausdorff 95th Percentile Scores by Experiment", fontsize=14)
    axes[1].set_ylabel("Hausdorff 95th Percentile", fontsize=22)
    axes[1].set_xlabel("Experiment Type", fontsize=22)
    axes[1].tick_params(axis='x', rotation=30)
    # Adjust legend to be outside the plot
    # axes[1].legend(title="Data Type", loc='upper left', bbox_to_anchor=(1, 1))

    if friedman_hd.pvalue < 0.05:
        base_y = df[df["Metric"] == "Hausdorff"]["Score"].max()
        line_height = 18.0       # distanza verticale base della linea orizzontale
        text_offset = 0.0        # offset verticale per l'asterisco
        left_height = 2.0        # altezza della stanghetta sinistra
        right_height = 12.0      # altezza della stanghetta destra

        for i, exp1 in enumerate(experiments[:-1]):
            for j, exp2 in enumerate(experiments[i+1:], start=i+1):
                if hd_p_values[exp2] < 0.05:
                    if hd_p_values[exp2] < 0.001: asterisk = "***"
                    elif hd_p_values[exp2] < 0.01: asterisk = "**"
                    elif hd_p_values[exp2] < 0.05: asterisk = "*"

                    y = base_y + (j - i) * line_height

                    # Linea orizzontale
                    axes[1].plot([i, j], [y, y], color='black', lw=2.5)

                    # Stanghetta verticale sinistra (asimmetrica)
                    axes[1].plot([i, i], [y - left_height, y], color='black', lw=2.5)

                    # Stanghetta verticale destra (asimmetrica)
                    axes[1].plot([j, j], [y - right_height, y], color='black', lw=2.5)

                    # Asterisco centrato sopra la linea
                    axes[1].text((i + j) / 2, y + text_offset, asterisk, ha='center', va='bottom', fontsize=25)
                                
    # plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout to make space for the legends
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
    axes[0].legend_.remove()
    axes[1].legend_.remove()
    # for the second axis set the y limit
    axes[1].set_ylim(-10, 115)  # Adjust this limit based on your data range
    plt.show()


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

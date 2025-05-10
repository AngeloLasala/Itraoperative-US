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
    data = {
        "Score": [],
        "Metric": [],
        "Experiment": []
    }

    # Carica dati reali
    dsc_real = np.load(os.path.join(metrics_folder, experiment_list[0], 'real_dsc.npy')).tolist()
    haus_real = np.load(os.path.join(metrics_folder, experiment_list[0], 'real_hausdorff.npy')).tolist()

    data["Score"].extend(dsc_real)
    data["Metric"].extend(["DSC"] * len(dsc_real))
    data["Experiment"].extend(["Real"] * len(dsc_real))

    data["Score"].extend(haus_real)
    data["Metric"].extend(["Hausdorff"] * len(haus_real))
    data["Experiment"].extend(["Real"] * len(haus_real))

    # Carica esperimenti
    for i, experiment in enumerate(experiment_list):
        dsc_generated = np.load(os.path.join(metrics_folder, experiment, 'gen_dsc.npy')).tolist()
        haus_generated = np.load(os.path.join(metrics_folder, experiment, 'gen_hausdorff.npy')).tolist()

        label = f"Experiment {i+1}"

        data["Score"].extend(dsc_generated)
        data["Metric"].extend(["DSC"] * len(dsc_generated))
        data["Experiment"].extend([label] * len(dsc_generated))

        data["Score"].extend(haus_generated)
        data["Metric"].extend(["Hausdorff"] * len(haus_generated))
        data["Experiment"].extend([label] * len(haus_generated))

    df = pd.DataFrame(data)

    # Palette personalizzata
    base_palette = {}
    all_experiments = df["Experiment"].unique()
    for exp in all_experiments:
        if exp == "Real":
            base_palette[exp] = (65/255, 105/255, 225/255)  # Royal Blue
        else:
            base_palette[exp] = (173/255, 216/255, 230/255)  # Light Blue

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    for ax, metric in zip(axes, ["DSC", "Hausdorff"]):
        # Violin + Strip
        sns.violinplot(x="Experiment", y="Score", data=df[df["Metric"] == metric],
                       ax=ax, inner=None, palette=base_palette)
        sns.stripplot(x="Experiment", y="Score", data=df[df["Metric"] == metric],
                      ax=ax, color="black", size=4, jitter=True, alpha=0.2)

        # Mappa posizione x reale
        xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        experiment_pos_map = {label: pos for pos, label in enumerate(xtick_labels)}

        # Calcolo e plot di mediana, Q1, Q3
        grouped = df[df["Metric"] == metric].groupby("Experiment")["Score"]
        for exp, scores in grouped:
            if exp in experiment_pos_map:
                x = experiment_pos_map[exp]
                q1 = scores.quantile(0.25)
                median = scores.mean()
                q3 = scores.quantile(0.75)

                ax.plot([x - 0.2, x + 0.2], [median, median], color='black', linewidth=2)
                ax.plot([x - 0.2, x + 0.2], [q1, q1], color='black', linewidth=1.5, linestyle='dotted')
                ax.plot([x - 0.2, x + 0.2], [q3, q3], color='black', linewidth=1.5, linestyle='dotted')

        # Font size
        ax.set_ylabel("DSC" if metric == "DSC" else "Hausdorff 95 percentile", fontsize=26)
        ax.set_xlabel("", fontsize=24)
        ax.tick_params(axis='x', labelsize=24)
        ax.tick_params(axis='y', labelsize=24)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read the Alligment pre-computed alligment metrics")
    parser.add_argument("--metrics_folder", type=str, default="metrics", help="folder of the metrics")
    args = parser.parse_args()

    experiment_list = [ 
                        # "VAE_finetuning_split_1_cond_ldm_finetuning_w_3.0_ddpm_6000",
                        # "VAE_finetuning_split_1_cond_ldm_finetuning_w_3.0_dpm_solver_5000",
                        # "VAE_random_split_1_cond_ldm_finetuning_w_3.0_ddpm_8000",
                        # "VAE_random_split_1_cond_ldm_finetuning_w_3.0_dpm_solver_8000",
                        # "VAE_finetuning_split_1_Controlnet_finetuning_empty_text_w_5.0_ddpm_7000",
                        # "VAE_finetuning_split_1_Controlnet_finetuning_empty_text_w_5.0_dpm_solver_7000",
                        "VAE_finetuning_split_1_Controlnet_lora_empty_text_w_3.0_ddpm_5000"
                        ]

    main(args.metrics_folder, experiment_list)
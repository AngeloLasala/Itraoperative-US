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
from scipy.stats import wilcoxon, friedmanchisquare
import cv2
from statannotations.Annotator import Annotator
from statsmodels.stats.multitest import multipletests


def bonferroni_wilcoxon(data_dict, metric, pairs):
    """Calcola p-value Wilcoxon per tutte le coppie e li corregge con Bonferroni"""
    raw_pvals = []
    for a, b in pairs:
        stat, p = wilcoxon(data_dict[a][metric], data_dict[b][metric])
        raw_pvals.append(p)
    _, pvals_corr, _, _ = multipletests(raw_pvals, alpha=0.05, method='bonferroni')
    return pvals_corr

def main(args, experiment_list):
    """
    Main function to read the Alligment pre-computed alligment metrics
    
    Paramerters
    ----------
    metrics_folder : str
        Path to the folder containing the metrics

    experiment_list : list
        List of experiments to be compared
    """
    print(os.listdir(args))
    split_min = {0:396, 1:400, 2:400, 3:398, 4:399}
    for s in range(5):
        print(f'Split {s}')

        split_dict = {}
        for experiment in experiment_list:
            experiment_name = experiment + f'_{s}'
            dsc_list = np.load(os.path.join(args, experiment_name, 'gen_dsc.npy'))[:split_min[s]]
            hd_list = np.load(os.path.join(args, experiment_name, 'gen_hausdorff.npy'))[:split_min[s]]
            print(f'  {experiment} DSC: {np.median(dsc_list):.3f} [{np.quantile(dsc_list, 0.25):.3f}, {np.quantile(dsc_list, 0.75):.3f}] - HD: {np.median(hd_list):.2f} [{np.quantile(hd_list, 0.25):.2f}, {np.quantile(hd_list, 0.75):.2f}]')
            split_dict[experiment] = {"dsc": dsc_list.tolist(), "hd": hd_list.tolist()}    

        ## Statistical test - omnibus fiedman test
        print()
        print('Statistical test') 
        dsc_data = [split_dict[experiment]["dsc"] for experiment in experiment_list]
        hd_data = [split_dict[experiment]["hd"] for experiment in experiment_list]
        stat_dsc, p_dsc = friedmanchisquare(*dsc_data)
        stat_hd, p_hd = friedmanchisquare(*hd_data)
        print(f'  DSC: stat={stat_dsc:.3f}, p={p_dsc:.3e}')
        print(f'  HD: stat={stat_hd:.3f}, p={p_hd:.3e}')
        print()
        if p_dsc < 0.05:
            print(' DSC - Post-hoc Wilcoxon signed-rank test with Bonferroni correction')
            aplfa = 0.05
            alpha_corrected = aplfa / (len(experiment_list) * (len(experiment_list) - 1) / 2)
            print( f'  corrected alpha is {alpha_corrected:.3e}' )
            for i in range(len(experiment_list)):
                for j in range(i+1, len(experiment_list)):
                    stat, p = wilcoxon(split_dict[experiment_list[i]]["dsc"], split_dict[experiment_list[j]]["dsc"])
                    if p < alpha_corrected:
                        print(f'    {experiment_list[i]} vs {experiment_list[j]}: stat={stat:.3f}, p={p:.3f}, Significant')
                    else:
                        print(f'    {experiment_list[i]} vs {experiment_list[j]}: stat={stat:.3f}, p={p:.3f}, Not Significant')

        else:
            print(' No significant difference between groups')
        if p_hd < 0.05:
            # print(f'  HD: stat={stat:.3f}, p={p:.3e}')
            print(' HD - Post-hoc Wilcoxon signed-rank test with Bonferroni correction')
            aplfa = 0.05
            alpha_corrected = aplfa / (len(experiment_list) * (len(experiment_list) - 1) / 2)
            print( f'  corrected alpha is {alpha_corrected:.3e}' )
            for i in range(len(experiment_list)):
                for j in range(i+1, len(experiment_list)):
                    stat, p = wilcoxon(split_dict[experiment_list[i]]["hd"], split_dict[experiment_list[j]]["hd"])
                    if p < alpha_corrected:
                        print(f'    {experiment_list[i]} vs {experiment_list[j]}: stat={stat:.3f}, p={p:.3f}, Significant')
                    else:
                        print(f'    {experiment_list[i]} vs {experiment_list[j]}: stat={stat:.3f}, p={p:.3f}, Not Significant')
        else:
            print(' No significant difference between groups')

        
        df_dsc = pd.DataFrame({exp: split_dict[exp]["dsc"] for exp in experiment_list})
        df_hd  = pd.DataFrame({exp: split_dict[exp]["hd"] for exp in experiment_list})

        df_dsc_melt = df_dsc.melt(var_name="Experiment", value_name="DSC")
        df_hd_melt  = df_hd.melt(var_name="Experiment", value_name="HD")

        fig, axes = plt.subplots(2, 1, figsize=(11, 10), num=f'Morf split {s}', sharex=True, tight_layout=True)

        # --- Plot DSC ---
        sns.violinplot(data=df_dsc_melt, x="Experiment", y="DSC", ax=axes[0],
                    inner=None, color="lightblue")
        sns.stripplot(data=df_dsc_melt, x="Experiment", y="DSC", ax=axes[0],
                    color="black", size=6, jitter=True, alpha=0.15)
        axes[0].set_title("DSC")

        # --- Plot HD ---
        sns.violinplot(data=df_hd_melt, x="Experiment", y="HD", ax=axes[1],
                    inner=None, color="lightblue")
        sns.stripplot(data=df_hd_melt, x="Experiment", y="HD", ax=axes[1],
                    color="black", size=6, jitter=True, alpha=0.15)
        axes[1].set_title("HD")

        # --- Definizione delle coppie ---
        pairs = [(experiment_list[i], experiment_list[j])
                for i in range(len(experiment_list))
                for j in range(i+1, len(experiment_list))]

        # DSC
        if p_dsc < 0.05:
            pvals_dsc_corr = bonferroni_wilcoxon(split_dict, "dsc", pairs)
            print(pvals_dsc_corr)
            annotator = Annotator(axes[0], pairs, data=df_dsc_melt,
                                x="Experiment", y="DSC")
            annotator.configure(text_format='star', loc='inside', verbose=0,
                                line_width=1.5,
                                fontsize=20) 
            annotator.set_pvalues(pvals_dsc_corr)
            annotator.annotate()

        # HD
        if p_hd < 0.05:
            pvals_hd_corr = bonferroni_wilcoxon(split_dict, "hd", pairs)
            print(pvals_hd_corr)
            annotator = Annotator(axes[1], pairs, data=df_hd_melt,
                                x="Experiment", y="HD")
            annotator.configure(text_format='star', loc='inside', verbose=0,
                                line_width=1.5,
                                fontsize=20) 
            annotator.set_pvalues(pvals_hd_corr)
            annotator.annotate()

        axes[1].tick_params(axis='x', labelsize=22)
        axes[0].tick_params(axis='x', labelsize=22)
        axes[0].tick_params(axis='y', labelsize=18)
        axes[1].tick_params(axis='y', labelsize=18)
        axes[0].yaxis.label.set_size(24)
        axes[1].yaxis.label.set_size(24)
        # set y 
        # axes[0].set_ylim(0.10, 1.99)
        # axes[1].set_ylim(-10, 130)
        
        axes[0].yaxis.grid(linestyle=':')
        axes[1].yaxis.grid(linestyle=':')
        
        # deactivate x labels for the first plot
        axes[0].set_xlabel('')
        axes[1].set_xlabel('')

        # deativate title
        axes[0].set_title('')
        axes[1].set_title('')
        

        plt.show()
        print('===============================================================')
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read the Alligment pre-computed alligment metrics")
    parser.add_argument("--metrics_folder", type=str, default="DSC_HD_metrics", help="folder of the metrics")
    args = parser.parse_args()

    experiment_list = [ 'Pix2pix', 'proposed', 'Controlnet', 'one-step']

    main(args.metrics_folder, experiment_list)
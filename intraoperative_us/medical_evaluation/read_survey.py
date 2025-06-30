"""
Read the syrvey and return the CM
"""
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def get_info_from_answer(answer_df):
    """
    Extract information from the answers DataFrame.

    Parameters
    ----------
    answer_df : pd.DataFrame
        DataFrame containing the survey answers.

    Returns
    -------
    clinical_specialization : str
        The clinical specialization of the respondent.
    experience : str
        The experience level with intraoperative imaging.
    answer_dict : dict
        Dictionary mapping question columns to binary answers (0 for 'Real', 1 for 'Fake').
    additional_info : np.ndarray
        Additional information provided by the respondent.
    """
    clinical_specialization = answer_df['What is your profession and clinical specialization?'].unique()[0]
    experience = answer_df['What is your experience with intraoperative imaging?'].unique()[0]
    print(f"Clinical Specialization: {clinical_specialization}")
    print(f"Experience with intraoperative imaging: {experience}")
    
    answer_dict = {}
    for col in answer_df.columns[3:-1]:
        if answer_df[col].unique()[0] == 'Real': a = 0
        elif answer_df[col].unique()[0] == 'Fake': a = 1

        answer_dict[str(col)] = a
    
    additional_info = answer_df['What were the key factors that influenced your decision?'].unique()
    print(f"Additional Information: {additional_info}")
      
    return clinical_specialization, experience, answer_dict, additional_info

def main(args):
    """
    Read the survey and compute the confusion matrix
    - 0: real
    - 1: generated
    """
    ## read the gt joson file
    gt_file = os.path.join(args.dict_path, args.gt_file)
    with open(gt_file, 'r') as f:
        gt_dict = json.load(f)

    ## read the answers CSV file
    answer_file = os.path.join(args.dict_path, args.answer_file)
    answer_df = pd.read_csv(answer_file)
    role, experience, answer_dict, additional_info = get_info_from_answer(answer_df)

    # get the confusion matrix and calassification report
    gt_labels, answers_labels = [], []
    for key in gt_dict.keys():
        gt_label = gt_dict[key]
        an_labels = answer_dict[key]
        gt_labels.append(gt_label[0])
        answers_labels.append(an_labels)

    cm = confusion_matrix(gt_labels, answers_labels)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(gt_labels, answers_labels, target_names=['Real', 'Generated']))

    ## plot the confusion matrix
    plt.figure(figsize=(8, 6))
    custom_cmap = LinearSegmentedColormap.from_list(
                    'custom_deepskyblue',
                    ['white','deepskyblue', 'black'],
                    N=256  # number of levels
                )
    sns.heatmap(cm, annot=True, fmt='d', cmap=custom_cmap, xticklabels=['Real', 'Generated'], yticklabels=['Real', 'Generated'], annot_kws={"size": 24})
    plt.xlabel('Predicted', fontsize=24)
    plt.ylabel('Ground True', fontsize=24)
    # increst the font size of the labels
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # increase the font size of the numer inside the cm
    plt.tick_params(labelsize=20)

    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read survey and compute confusion matrix')
    parser.add_argument('--dict_path', type=str, default="/home/angelo/Documenti/Itraoperative-US/intraoperative_us/medical_evaluation/form_images/dict", help='Path to the dict folder')
    parser.add_argument('--gt_file', type=str, default='gt_dict.json', help='file to the ground true file')
    parser.add_argument('--answer_file', type=str, default='Giulio_survey.csv', help='Path to the answers CSV file')
    args = parser.parse_args()

    main(args)

   
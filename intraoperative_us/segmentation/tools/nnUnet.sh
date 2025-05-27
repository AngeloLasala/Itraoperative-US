#!/bin/bash
# export the varibla path
export nnUNet_raw="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/RESECT_iUS_dataset/nnUNet_raw"
export nnUNet_preprocessed="/home/angelo/Documenti/Itraoperative-US/intraoperative_us/segmentation/nnUNet_preprocessed"
export nnUNet_results="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model/nnUNet_results"

# create dataset Real Real_Gex_x#numeber
# python -m intraoperative_us.segmentation.dataset.create_nnUnet_dataset

# # analyse raw data
# nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

# create split
python -m intraoperative_us.segmentation.dataset.create_nnUnet_split

# train the model
# nnUNetv2_train 1 2d 0 -device cuda
#!/bin/bash
# export the varibla path
export nnUNet_raw="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/RESECT_iUS_dataset/nnUNet_raw"
export nnUNet_preprocessed="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model/nnUNet_preprocessed"
export nnUNet_results="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model/nnUNet_results"

# create dataset Real Real_Gex_x#numeber
# python -m intraoperative_us.segmentation.dataset.create_nnUnet_dataset --save_folder "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model/"\
#                                                                        --name_dataset "RealGenx3" \
#                                                                        --trial 'VAE_finetuning' \
#                                                                        --experiment cond_ldm_finetuning

# # analyse raw data
# nnUNetv2_plan_and_preprocess -d 4 --verify_dataset_integrity

# create split
# python -m intraoperative_us.segmentation.dataset.create_nnUnet_split -d 4

# # train the model
# for split in 4 ; do
#     nnUNetv2_train 4 2d $split -device cuda
# done

# best configuration
# nnUNetv2_find_best_configuration 4

# predict
# nnUNetv2_predict -d 4\
#                  -i "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/RESECT_iUS_dataset/nnUNet_raw/Dataset001_Real/imagesTs" \
#                  -o "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model/nnUNet_results/Dataset004_RealGenx3/inference" \
#                  -f  0 1 2 3 4 \
#                  -tr nnUNetTrainer \
#                  -c 2d \
#                  -p nnUNetPlans

# postprocess
nnUNetv2_apply_postprocessing -i "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model/nnUNet_results/Dataset004_RealGenx3/inference" \
                              -o "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model/nnUNet_results/Dataset004_RealGenx3/inference/post_process" \
                              -pp_pkl_file "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model/nnUNet_results/Dataset004_RealGenx3/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl" \
                               -np 8 \
                               -plans_json "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model/nnUNet_results/Dataset004_RealGenx3/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json"
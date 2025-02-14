"""
Read images form slicer
"""
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging
import argparse
import tqdm
import seaborn as sns
import pandas as pd

def read_xls(dataset_path, *selected_patients):
    """
    Read the xls file with the information of the dataset

    Parameters
    ----------
    excel_path : str
        Path of the xls file
    selected_patients : list
        List of selected patients

    Returns
    -------
    dataset_dict : dict
        Dictionary with the information of the dataset
    """
    ## check an exel file in the dataset path
    for item in os.listdir(dataset_path):
        if item.endswith(".xlsx"):
            excel_path = os.path.join(dataset_path, item)
        
    
    dataset_info = {}
    dataset_pd = pd.read_excel(excel_path, index_col=False)
    # sub dataset with only yher column of interest
    dataset_pd = dataset_pd[['Case Number', 'Age', 'Sex', 'WHO Grade']]

    ## selcet a dataset of a given 'Case Number' entry list
    dataset_pd = dataset_pd[dataset_pd['Case Number'].isin(selected_patients)]
    dataset_pd["Case Number"] = dataset_pd["Case Number"].astype(str).str.zfill(3)
    ## convert all the case number item in string
    dataset_pd['Case Number'] = dataset_pd['Case Number'].astype(str)
    dataset_pd['Case Number'] = 'ReMIND-'+dataset_pd['Case Number']+'_pre_dura'
    return dataset_pd


    # for i in range(len(dataset)):
    #     dataset_dict[dataset['Subject'][i]] = {'volume':dataset['Volume'][i], 'tumor':dataset['Tumor'][i], 'sulci':dataset['Sulci'][i], 'falx':dataset['Falx'][i]}
    # return dataset_dict

def check_subject_volume(dataset_path):
    """
    Given the path of the RESECT dataset iUS, return the list of subject with volume
    """
    subjects_with_volume = []
    for item in os.listdir(dataset_path):
        # chek if the sting item finished with .nii
        if item.endswith(".nii.gz"):
            subjects_with_volume.append(item.split('.')[0])
    return subjects_with_volume 

def subject_label_dict(dataset_path, save_dict=False):
    """
    Create the dictionary with the subject and the label
    """
    ## subject with volume
    subjects_with_volume = check_subject_volume(dataset_path)

    dataset_dict = {}
    for subject in subjects_with_volume:
        dataset_dict[subject] = {'volume':f'{subject}.nii', 'tumor':[], 'sulci':None, 'falx':None}
       
    count = 0
    for item in os.listdir(dataset_path):
        if item.endswith(".nrrd"):
            subject = item.split('_')[0].split('-')[0] + '-' + item.split('_')[0].split('-')[1]
            dataset_dict[f'{subject}_pre_dura']['tumor'] = item
            count += 1
    logging.info(f'Number of segmentation: {count}')

    if save_dict:
        # save dict ad json
        with open(os.path.join('dataset_dict.json'), 'w') as json_file:
            json.dump(dataset_dict, json_file)
    return dataset_dict
       
def bar_chart_dataset(dataset_path):
    """
    Create a bar chart of the datset, for each label
    """
    dataset_dict = subject_label_dict(dataset_path)
    subject = list(dataset_dict.keys())

    labels = ['volume', 'tumor', 'sulci', 'falx']
    labels_count = {'volume':0, 'tumor':0, 'sulci':0, 'falx':0}
    for key, value in dataset_dict.items():
        for label in labels:
            if value[label] is not None:
                labels_count[label] += 1

    color = ['blue', 'green', 'green', 'green']
    fig, ax = plt.subplots(figsize=(5, 5), num='Label distribution')
    ax.bar(labels, [(labels_count[label]/len(subject))*100 for label in labels])
    # set the color of the bar based on color list
    for i in range(len(labels)):
        ax.patches[i].set_facecolor(color[i])
    ax.set_ylabel('Percentage')
    # set the orizontal grid
    ax.yaxis.grid(linestyle=':')

def compute_volume_mm3(segmentation_map):
    """
    Compute the volume in mm^3 of the segmentation maps

    Parameters
    ----------
    volume_array : 
        Segmentation map, SimpleITK image

    Returns
    -------
    volume_mm3 : float
        Volume in mm^3
    """
    spacing = segmentation_map.GetSpacing()
    volume_mm3 = np.sum(sitk.GetArrayFromImage(segmentation_map)) * np.prod(spacing)
    return volume_mm3, spacing

def register_volume_label(volume, label):
    """
    Register the volume and the label
    """
    # Resample mask to iUS geometry
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(volume)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_mask = resampler.Execute(label)

    # Visualize
    reg_volume = sitk.GetArrayFromImage(volume)
    reg_label = sitk.GetArrayFromImage(resampled_mask)
    return reg_volume, reg_label

def read_volume_label_subject(dataset_path, dataset_dict, subject, show_plot=True):
    """
    Read the volume and the label of a SINGLE subject

    Parameters
    ----------
    dataset_path : str
        Path of the dataset
    dataset_dict : dict
        Dictionary with the subject and the label, from the function subject_label_dict
    subject : str
        Subject to read, 'Case1-US-before'
    """

    value = dataset_dict[subject]

    logging.info(f'subject: {subject}')
    if value['volume'] is not None : 
        volume = os.path.join(dataset_path, value['volume'])
        volume = sitk.ReadImage(volume)
        volume_array = sitk.GetArrayFromImage(volume)  # Shape: (Z, Y, X)
        logging.info(f'volume size from volume: {volume.GetSize()}')
        spacing = volume.GetSpacing()
        logging.info(f'volume spacing: {spacing}')
        logging.info(f'volume shape: {volume_array.shape}')
    else:
        logging.info('No volume for this subject')

    if value['tumor'] is not None :
        tumor = os.path.join(dataset_path, value['tumor'])
        tumor = sitk.ReadImage(tumor)
        tumor_size, spacing_t = compute_volume_mm3(tumor)
        logging.info(f'tumor spacing: {spacing_t} [MRI space]')
        tumor_array = sitk.GetArrayFromImage(tumor)
    else:
        tumor_array = None
        logging.info('No tumor for this subject')

    if value['sulci'] is not None : 
        sulci = os.path.join(dataset_path, value['sulci'])
        sulci = sitk.ReadImage(sulci)
        sulci_array = sitk.GetArrayFromImage(sulci)
        # print('sulci shape:', sulci_array.shape)
    else:
        sulci_array = None
        logging.info('No sulci for this subject')

    if value['falx'] is not None : 
        falx = os.path.join(dataset_path, value['falx'])
        falx = sitk.ReadImage(falx)
        falx_array = sitk.GetArrayFromImage(falx)

    else:
        falx_array = None
        logging.info('No falx for this subject')

    ## Registration of the volume and the label
    volume_array, tumor_array = register_volume_label(volume, tumor)
    

    # for i in np.arange(0, volume_array.shape[0], 20):
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(volume_array[i, :, :], cmap='gray')
    #     plt.title("iUS Volume")
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(volume_array[i, :, :], cmap='gray')
    #     plt.imshow(tumor_array[i, :, :], alpha=0.3, cmap='jet')
    #     plt.title("Aligned Mask")
    # plt.show()

    # in tumor, find the slice with the maximun number of 1 pixel
    max_tumor = 0
    for i in range(tumor_array.shape[0]):
        if np.sum(tumor_array[i, :, :]) > max_tumor:
            max_tumor = np.sum(tumor_array[i, :, :])
            slice_tumor = i
    max_tumor_surface = np.sum(max_tumor) * spacing[0] * spacing[1]
    logging.info(f'tumor size: {max_tumor_surface/100.0:.4f} [cm^2] in slice {slice_tumor}')
    logging.info('======================================')

    if show_plot:
        ## plot the volume with the tumor in the max_tumor slice 
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), num=f'{subject} max tumor slice: {slice_tumor}') 
        ax.imshow(volume_array[slice_tumor, :, :], cmap="gray")
        ax.imshow(tumor_array[slice_tumor, :, :], alpha=0.2, cmap="jet")  # Overlay
        ax.axis('off')

    return volume_array, tumor_array, sulci_array, falx_array, tumor_size, spacing, slice_tumor

def slicing_volume(volume_array, tumor_array, sulci_array, falx_array, spacing, slice_tumor, slice_spacing=1, slicing_window=35):
    """
    Slice the volume each 5 mm upward and backward from
    the slice of tumor that contains the maximum number of 1 pixel, 'bigger surface'

    Parameters
    ----------
    volume_array : 
        Volume array, numpy or None
    tumor_array :
        Tumor array, numpy or None
    sulci_array :
        Sulci array, numpy or None
    falx_array :
        Falx array, numpy or None   
    spacing :
        Spacing of the volume in mm, 3d array
    slice_tumor :
        Slice of the tumor that contains the maximum number of 1 pixel, bigger surface
    slice_spacing :
        Spacing between the slices in mm, default 1 mm
    slicing_window :
        Windowto performe slicing, default 35 mm
    """
    ## surface of the max tumor slice
    max_tumor_surface = np.sum(tumor_array[slice_tumor, :, :]) * spacing[0] * spacing[1]
    # print(f'tumor size: {max_tumor_surface/100.0:.4f} [cm^2] in slice {slice_tumor} [{slice_tumor * spacing[0]:.4f}mm],')
    # print(f'depht of the volume: {volume_array.shape[0]* spacing[0]:.4f} mm')
    slicing_window = tumor_array.shape[0] * spacing[2] // 4

    int_slicing = int(slice_spacing/spacing[2]) #int of slice
    int_window = int(slicing_window/spacing[2]) #int of window)
    logging.info(f'int_slicing: {int_slicing}, int_window: {int_window}')  
    tumor_surface_list = []
    for slice_i in np.arange(slice_tumor - (int_window//2), slice_tumor + (int_window//2) + 1 ,int_slicing):
        # print(f'tumor surface: {np.sum(tumor_array[slice_i, :, :]) * spacing[0] * spacing[1]:.4f}')
        tumor_surface_list.append(np.sum(tumor_array[slice_i, :, :]) * spacing[0] * spacing[1])
    #     plt.figure()
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(volume_array[slice_i, :, :], cmap='gray')
    #     plt.imshow(tumor_array[slice_i, :, :], alpha=0.2, cmap='jet')
    #     plt.axis('off')

    #     plt.subplot(1, 2, 2)
    #     plt.imshow(volume_array[slice_i, :, :], cmap='gray')
    #     plt.axis('off')
    # plt.show()
    return tumor_surface_list

  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read the dataset of the RESECT iUS dataset')
    parser.add_argument('--dataset_path', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/", help='Path to the dataset')
    parser.add_argument('--log', type=str, default='DEBUG', help='Logging level')
    args = parser.parse_args()

    ## set the logger
    logging_dict = {'DEBUG':logging.DEBUG, 'INFO':logging.INFO, 'WARNING':logging.WARNING, 'ERROR':logging.ERROR, 'CRITICAL':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    # read the xls file
    dataset_info = read_xls(args.dataset_path, 1, 2, 6, 11, 13, 18, 20, 26, 27, 28,
                                               37, 39, 40, 45, 46, 48, 56, 58, 59, 65, 66,
                                               71, 74, 77, 85, 88, 101, 91, 94, 96, 108, 110, 114)

    # subject with volume
    subjects_with_volume = check_subject_volume(args.dataset_path)
    selected_patients = dataset_info['Case Number'].values
    logging.info(f'Number of subject with volume: {len(subjects_with_volume)}')
    logging.info(f'Number of selected patients: {len(selected_patients)}')

    

    # Create the dictionary with the subject and the label
    dataset_dict = subject_label_dict(args.dataset_path)

    # Read the volume and the label of a SINGLE subject
    subject = subjects_with_volume[np.random.randint(0, len(selected_patients))]
    subject = 'ReMIND-001_pre_dura'
    read_volume_label_subject(args.dataset_path, dataset_dict, subject, show_plot=True)
    
    # Read the entire dataset
    # volume_list, depht_list = [], []
    # for subject in tqdm.tqdm(selected_patients):
    #     volume, tumor, sulci, flax, tumor_size, spacing, slice_tumor = read_volume_label_subject(args.dataset_path, dataset_dict, subject, show_plot=False)
    #     max_tumor = slicing_volume(volume, tumor, sulci, flax, spacing, slice_tumor, slice_spacing=1.0, slicing_window=25)
    #     depht_list.append(max_tumor)
    #     volume_list.append(tumor_size/1000.0)
    #     print(f'Subject: {subject}, tumor size: {tumor_size/1000.0:.4f} [cm^3]')

    # # ## get a single list of item from the list of list of dehpt_list
    # depht_list = [item/100.0 for sublist in depht_list for item in sublist]
    # print(f'Number of 2D slice: {len(depht_list)}')

    # ## plot the histogram of the tumor size
    # plt.figure('Tumor size', figsize=(5, 5), tight_layout=True)
    # plt.subplot(1, 2, 1)
    # sns.boxplot(volume_list,  color='green', boxprops={'alpha': 0.5}, flierprops={'marker': 'o', 'markersize': 5})
    # sns.stripplot(volume_list, color='k', alpha=1, jitter=True, size=5)
    # plt.xlabel('Tumor size')
    # plt.ylabel(r'Tumor size [cm$^3$]')
    # plt.grid(axis='y', linestyle=':')

    # plt.subplot(1, 2, 2)
    # sns.boxplot(depht_list,  color='green', boxprops={'alpha': 0.5}, flierprops={'marker': 'o', 'markersize': 5})
    # sns.stripplot(depht_list, color='k', alpha=0.6, jitter=True, size=3)
    # plt.xlabel('tumor surface')
    # plt.ylabel(r'tumor surface [cm$^2$]')
    # plt.grid(axis='y', linestyle=':')

    plt.show()


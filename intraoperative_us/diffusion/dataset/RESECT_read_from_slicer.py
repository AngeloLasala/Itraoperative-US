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
from PIL import Image

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
        dataset_dict[subject] = {'volume':f'{subject}.nii', 'tumor':None, 'sulci':None, 'falx':None}
       
        # chek if exitst the item that start with f'{item}-tumor'
        if os.path.exists(os.path.join(dataset_path, f'{subject}-tumor.nii.gz.seg.nrrd')):
            dataset_dict[subject]['tumor'] = f'{subject}-tumor.nii.gz.seg.nrrd'
        else:
            # chek if exitst the item that start with f'{item}-tumor.nii.sef.nrrd'
            if os.path.exists(os.path.join(dataset_path, f'{subject}-tumor.nii.seg.nrrd')):
                dataset_dict[subject]['tumor'] = f'{subject}-tumor.nii.seg.nrrd'
            else:
                # print(f'No tumor for subject {subject}')
                pass
        
        # chek if exitst the item that start with f'{item}-sulci'
        if os.path.exists(os.path.join(dataset_path, f'{subject}-sulci.nii.gz.seg.nrrd')):
            dataset_dict[subject]['sulci'] = f'{subject}-sulci.nii.gz.seg.nrrd'
        else:
            # chek if exitst the item that start with f'{item}-sulci.nii.sef.nrrd'
            if os.path.exists(os.path.join(dataset_path, f'{subject}-sulci.nii.seg.nrrd')):
                dataset_dict[subject]['sulci'] = f'{subject}-sulci.nii.seg.nrrd'
            else:
                # print(f'No sulci for subject {subject}')   
                pass

        # chek if exitst the item that start with f'{item}-falx'
        if os.path.exists(os.path.join(dataset_path, f'{subject}-falx.nii.gz.seg.nrrd')):
            dataset_dict[subject]['falx'] = f'{subject}-falx.nii.gz.seg.nrrd'
        else:
            # chek if exitst the item that start with f'{item}-falx.nii.sef.nrrd'
            if os.path.exists(os.path.join(dataset_path, f'{subject}-falx.nii.seg.nrrd')):
                dataset_dict[subject]['falx'] = f'{subject}-falx.nii.seg.nrrd'
            else:
                pass
                # print(f'No falx for subject {subject}')

    if save_dict:
        # save dict ad json
        with open(os.path.join(dataset_path, 'dataset_dict.json'), 'w') as json_file:
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
        logging.info(f'volume shape: {volume_array.shape}')
    else:
        logging.info('No volume for this subject')

    if value['tumor'] is not None :
        tumor = os.path.join(dataset_path, value['tumor'])
        tumor = sitk.ReadImage(tumor)
        tumor_size, spacing = compute_volume_mm3(tumor)
        # covert the order of the spacinf in 2, 1, 0
        # spacing = np.array(spacing)[::-1]
        logging.info(f'tumor size: {tumor_size/1000.0:.4f} [cm^3]')
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
        sulci_array = np.zeros_like(volume_array)
        logging.info('No sulci for this subject')

    if value['falx'] is not None : 
        falx = os.path.join(dataset_path, value['falx'])
        falx = sitk.ReadImage(falx)
        falx_array = sitk.GetArrayFromImage(falx)

    else:
        falx_array = np.zeros_like(volume_array)
        logging.info('No falx for this subject')

    # in tumor, find the slice with the maximun number of 1 pixel
    max_tumor = 0
    for i in range(tumor_array.shape[0]):
        if np.sum(tumor_array[i, :, :]) > max_tumor:
            max_tumor = np.sum(tumor_array[i, :, :])
            slice_tumor = i
    max_tumor_surface = np.sum(max_tumor) * tumor.GetSpacing()[1] * volume.GetSpacing()[2]
    logging.info(f'tumor size: {max_tumor_surface/100.0:.4f} [cm^2] in slice {slice_tumor}')
    logging.info('======================================')

    if show_plot:
        ## plot the volume with the tumor in the max_tumor slice 
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), num=f'{subject} max tumor slice: {slice_tumor}') 
        ax.imshow(volume_array[slice_tumor, :, :], cmap="gray")
        ax.imshow(tumor_array[slice_tumor, :, :], alpha=0.2, cmap="jet")  # Overlay
        ax.axis('off')

    return volume_array, tumor_array, sulci_array, falx_array, tumor_size, spacing, slice_tumor

def slicing_volume(volume_array, tumor_array, sulci_array, falx_array, spacing, slice_tumor,
                   subject, slice_spacing=1,
                   dataset_path=None, save_dataset=False):
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
    dataset_path :
        Path of the dataset, default None
    save_dataset :
        Save the dataset, default False
    """

    ## create folders structure for the dataset to develop DL model
    if save_dataset:
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        os.makedirs(os.path.join(dataset_path, 'dataset'), exist_ok=True)

        subject_path = os.path.join(dataset_path, 'dataset', subject)
        os.makedirs(os.path.join(subject_path, 'volume'), exist_ok=True)
        os.makedirs(os.path.join(subject_path, 'tumor'), exist_ok=True)
        os.makedirs(os.path.join(subject_path, 'sulci'), exist_ok=True)
        os.makedirs(os.path.join(subject_path, 'falx'), exist_ok=True)
    
    ## surface of the max tumor slice
    max_tumor_surface = np.sum(tumor_array[slice_tumor, :, :]) * spacing[1] * spacing[2]
    slicing_window = tumor_array.shape[0] * spacing[0] // 4

    int_slicing = int(slice_spacing/spacing[0]) #int of slice
    int_window = int(slicing_window/spacing[0]) #int of window)
    tumor_surface_list = []
    print(subject)
    for slice_i in np.arange(slice_tumor - (int_window//2), slice_tumor + (int_window//2) + 1 ,int_slicing):
        logging.info(f'tumor surface: {np.sum(tumor_array[slice_i, :, :]) * spacing[1] * spacing[2]:.4f}')
        tumor_surface_list.append(np.sum(tumor_array[slice_i, :, :]) * spacing[1] * spacing[2])

        ## save the dataset
        if save_dataset:
            vol = volume_array[slice_i, :, :]
            tum = tumor_array[slice_i, :, :]*255
            sul = sulci_array[slice_i, :, :]*255
            fal = falx_array[slice_i, :, :]*255

            # convert un PIL image
            vol = Image.fromarray(vol).convert('L')
            tum = Image.fromarray(tum).convert('L')
            sul = Image.fromarray(sul).convert('L')
            fal = Image.fromarray(fal).convert('L')

            # save the image, only image with tumor
            if np.sum(tum) > 0.0: 
                vol.save(os.path.join(subject_path, 'volume', f'{subject}_{slice_i}.png'))
                tum.save(os.path.join(subject_path, 'tumor', f'{subject}_{slice_i}.png'))
                sul.save(os.path.join(subject_path, 'sulci', f'{subject}_{slice_i}.png'))
                fal.save(os.path.join(subject_path, 'falx', f'{subject}_{slice_i}.png'))

            
        # plt.figure(figsize=(15, 15), num=f'{slice_i} slice', tight_layout=True)
        # plt.subplot(1, 2, 1)
        # plt.imshow(volume_array[slice_i, :, :], cmap='gray')
        # plt.imshow(tumor_array[slice_i, :, :], alpha=0.2, cmap='jet')
        # plt.axis('off')

        # plt.subplot(1, 2, 2)
        # plt.imshow(volume_array[slice_i, :, :], cmap='gray')
        # plt.axis('off')
        # plt.show()
    return tumor_surface_list, aspect_ratio

  

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read the dataset of the RESECT iUS dataset')
    parser.add_argument('--dataset_path', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/RESECT_iUS_dataset", help='Path to the dataset')
    parser.add_argument('--log', type=str, default='debug', help='Logging level')
    args = parser.parse_args()

    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])


    # subject with volume
    subjects_with_volume = check_subject_volume(args.dataset_path)
    logging.info(f'Number of subjects with volume: {len(subjects_with_volume)}')

    # Create the dictionary with the subject and the label
    dataset_dict = subject_label_dict(args.dataset_path)

    # Read the volume and the label of a SINGLE subject
    subject = subjects_with_volume[np.random.randint(0, len(subjects_with_volume))]
    read_volume_label_subject(args.dataset_path, dataset_dict, subject, show_plot=True)
    
    # Read the entire dataset
    volume_list, depht_list, aspect_ratio = [], [], []
    for subject in tqdm.tqdm(subjects_with_volume):
        volume, tumor, sulci, flax, tumor_size, spacing, slice_tumor = read_volume_label_subject(args.dataset_path, dataset_dict, subject, show_plot=False)
        max_tumor, aspect_ratio = slicing_volume(volume, tumor, sulci, flax, spacing, slice_tumor, subject,
                                                 slice_spacing=1.0,
                                                 dataset_path=args.dataset_path, save_dataset=True)
        aspect_ratio.append(volume.shape[1]/volume.shape[2])
        depht_list.append(max_tumor)
        volume_list.append(tumor_size/1000.0)
        # print(f'Subject: {subject}, tumor size: {tumor_size/1000.0:.4f} [cm^3]')

    ## get a single list of item from the list of list of dehpt_list
    depht_list = [item/100.0 for sublist in depht_list for item in sublist]
    print(f'Number of 2D slice: {len(depht_list)}')

    ## plot the histogram of the tumor size
    plt.figure('Tumor size', figsize=(5, 5), tight_layout=True)
    plt.subplot(1, 2, 1)
    sns.boxplot(volume_list,  color='green', boxprops={'alpha': 0.5}, flierprops={'marker': 'o', 'markersize': 5})
    sns.stripplot(volume_list, color='k', alpha=1, jitter=True, size=5)
    plt.xlabel('Tumor size')
    plt.ylabel(r'Tumor size [cm$^3$]')
    plt.grid(axis='y', linestyle=':')

    plt.subplot(1, 2, 2)
    sns.boxplot(depht_list,  color='green', boxprops={'alpha': 0.5}, flierprops={'marker': 'o', 'markersize': 5})
    sns.stripplot(depht_list, color='k', alpha=0.6, jitter=True, size=3)
    plt.xlabel('tumor surface')
    plt.ylabel(r'tumor surface [cm$^2$]')
    plt.grid(axis='y', linestyle=':')

    plt.figure('aspect ratio', figsize=(5, 5), tight_layout=True)
    print(aspect_ratio)
    sns.boxplot(aspect_ratio,  color='green', boxprops={'alpha': 0.5}, flierprops={'marker': 'o', 'markersize': 5})
    sns.stripplot(aspect_ratio, color='k', alpha=0.6, jitter=True, size=3)
    plt.xlabel('aspect ratio')
    plt.ylabel('aspect ratio')
    plt.grid(axis='y', linestyle=':')
    
    plt.show()


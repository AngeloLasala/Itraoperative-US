"""
Create forlde with de-identified real and generated images
"""
import numpy as np
import os
import cv2
import argparse
import re
import random
import json

def extract_case_number(s):
    """
    Extract the case number from a string.

    Parameters
    ----------
    s : str
        String from which to extract the case number.

    Returns
    -------
    int or None
        The extracted case number as an integer, or None if no case number is found.
    """
    match = re.search(r'Case(\d+)', s)
    if match:
        return int(match.group(1))
    return None

def set_reproducibility(seed):
    """
    Set the random seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)

def read_real_images(real_path, output_path, num_images, gt_dict=None, existing_id=None):
    """
    Read real images

    Parameters
    ----------
    real_path : str
        Path to the directory containing real images.
    num_images : int
        Number of images to select from the real category.
    
    Returns
    -------
    gt_real : dict
        Dictionary containing the real images and their labels.
    """
    set_reproducibility(42)  

    # gt dictionary
    if gt_dict is None:
        gt_dict = {}
    else:
        gt_dict = gt_dict.copy()  

    if existing_id is None:
        existing_id = []
    else:
        existing_id = existing_id.copy()

    # random selection of subject
    list_of_subject = [extract_case_number(f) for f in os.listdir(real_path) if extract_case_number(f) is not None]
    selected_subjects = random.sample(list_of_subject, num_images)
    
    for i in selected_subjects:
        case_path = os.path.join(real_path, f"Case{i}-US-before")
        images_path = os.path.join(case_path, "volume")
        selected_slice = random.choice(os.listdir(images_path))     # random slice

        random_id = random.randint(0, 100)
        while random_id in existing_id:
            random_id = random.randint(0, 100)
        existing_id.append(random_id)                            # random ID for de-identification

        # Read the image
        image_path = os.path.join(images_path, selected_slice)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        
        # Save the image with de-identification
        image_output_path = os.path.join(output_path, 'images', f"image_{random_id}.png")
        cv2.imwrite(image_output_path, image)
        
        # save ground truth label   
        gt_dict[f"image_{random_id}"] = [0, selected_slice] 

    
    return gt_dict, existing_id



def read_generated_images(gen_path, output_path, num_images, gt_dict=None, existing_id=None):
    """
    Read generated images

    Parameters
    ----------
    gen_path : str
        Path to the directory containing generated images.
    num_images : int
        Number of images to select from the generated category.
    
    Returns
    -------
    gt_gen : dict
        Dictionary containing the generated images and their labels.
    """
    set_reproducibility(42)  

    # gt dictionary
    if gt_dict is None:
        gt_dict = {}
    else:
        gt_dict = gt_dict.copy()

    if existing_id is None:
        existing_id = []
    else:
        existing_id = existing_id.copy()

    # random selection of images without replacement    
    selected_images = random.sample(os.listdir(gen_path), num_images)

    for selected_slice in selected_images:
        random_id = random.randint(0, 100)
        while random_id in existing_id:
            random_id = random.randint(0, 100)
        existing_id.append(random_id)

        image_path = os.path.join(gen_path, selected_slice)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        image_output_path = os.path.join(output_path, 'images', f"image_{random_id}.png")
        cv2.imwrite(image_output_path, image)
        gt_dict[f"image_{random_id}"] = [1, selected_slice]

    return gt_dict, existing_id

def main(args):
    """
    Create folder with de-identified real and generated images and the groud truth label
    real = 0, generated = 1

    Parameters
    ----------
    args : argparse.Namespace
        Arguments containing paths for real and generated images, and output path.
        - real_path : str
            Path to the directory containing real images.
        - gen_path : str
            Path to the directory containing generated images.
        - output_path : str
            Path to the directory where the output images will be saved.
        - num_images : int
            Number of images to select from each category. Default = 15.
    """
    ## create output folder
    images_path = os.path.join(args.output_path, "images")
    dict_path = os.path.join(args.output_path, "dict")
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(dict_path, exist_ok=True)

    ## Real images
    gt_dict, existing_id = read_real_images(args.real_path, args.output_path, args.num_images, gt_dict = {}, existing_id=None)

    ## Generated images
    gt_dict, existing_id = read_generated_images(args.gen_path, args.output_path, args.num_images, gt_dict= gt_dict, existing_id=existing_id)

    ## Save ground truth dictionary 
    gt_dict_path = os.path.join(dict_path, "gt_dict.json")
    with open(gt_dict_path, 'w') as f:
        json.dump(gt_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select images for medical evaluation.")
    parser.add_argument("--real_path", type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/RESECT_iUS_dataset/dataset",
                                                 help="Directory containing real images.")
    parser.add_argument("--gen_path", type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model/ius/VAE_finetuning/split_1/cond_ldm_finetuning/w_3.0/ddpm/samples_ep_6000/ius",
                                                help="Directory containing generated images.")
    parser.add_argument("--output_path", type=str, default="form_images", help="output path.")
    parser.add_argument("--num_images", type=int, default=15, help="Number of images to select from each category.")
    args = parser.parse_args()

    main(args)
    
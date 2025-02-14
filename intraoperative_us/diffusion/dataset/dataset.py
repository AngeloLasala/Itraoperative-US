import glob
import os

import torchvision
from PIL import Image
from tqdm import tqdm
import json
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import math
import matplotlib.pyplot as plt


class IntraoperativeUS():
    """
    IntraoperativeUS dataset class for developing DL model for neurosurgical US images

    Current version: 14/02/2025
    ---------------------------
    The data come from the 'RESECT_iUS_dataset/dataset'. The tree data is the follow
    ../RESECT_iUS_dataset/dataset
    ├── Case1-US-before
    │   ├── volume
    │   |   ├── Case1-US-before_126.npy    
    │   │   ├── Case1-US-before_130.npy
    │   ├── tumor
    │   |   ├── Case1-US-before_126.npy
    │   │   ├── Case1-US-before_130.npy
    │   ├── sulci
    │   |   ├── Case1-US-before_126.npy
    │   │   ├── Case1-US-before_130.npy
    │   ├── fulci
    │   |   ├── Case1-US-before_126.npy
    │   │   ├── Case1-US-before_130.npy
    ├── Case2-US-before
    ├── Case3-US-before
    └── Case4-US-before
    """
    def __init__(self, size, dataset_path, im_channels, split,
                 splitting_seed, train_percentage, val_percentage, test_percentage,
                 condition_config=None, data_augmentation=None):

        self.dataset_path = dataset_path

        # condition and data augmentation parameters
        self.condition_types = [] if condition_config is None else condition_config['condition_types']
        self.data_augmentation = data_augmentation if data_augmentation is not None else None
        
        #img parameters
        self.size = size
        self.im_channels = im_channels

        #splitting parameters   
        self.splitting_seed = splitting_seed
        self.train_percentage = train_percentage
        self.val_percentage = val_percentage
        self.test_percentage = test_percentage

        ## get the splitting of the dataset
        self.split = split
        self.splitting_dict = self.get_data_splittings()
        self.subjects_files = self.splitting_dict[self.split]

        ## image and label list
        self.image_list, self.label_list = self.get_image_label_dict()['image'], self.get_image_label_dict()['tumor']


    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, index):
        im, label, subject = self.get_image_label(index)

        cond_inputs = {}    ## add to this dict the condition inputs
        if len(self.condition_types) > 0:  # check if there is at least one condition in ['class', 'text', 'image']

            ################ IMAGE CONDITION ###################################
            if 'image' in self.condition_types:
                if self.data_augmentation_image is not None:
                    im_tensor, label = self.data_augmentation_image(im, label)
                else:
                    im_tensor, label_tensor = self.trasform(im, label)
                cond_inputs['image'] = label_tensor
            #####################################################################
            return im_tensor, cond_inputs   

        else: # no condition
            if self.data_augmentation is not None:
                im_tensor = self.data_augmentation(im)
            else:
                resize = transforms.Resize(size=self.size)
                image = resize(im)
                image = transforms.functional.to_tensor(image)
                im_tensor = (2 * image) - 1
                
            return im_tensor

        # return im_tensor, label_tensor


    def get_image_label(self, index):
        """
        Return the image and label given the index
        """
        # read image and label with PIL
        im = Image.open(self.image_list[index])
        label = Image.open(self.label_list[index])
        subject = self.image_list[index].split('/')[-1].split('.')[0]
        return im, label, subject

    def get_data_splittings(self):
        """
        Given the path of the dataset return the list of patient for train, valindation and test
        """
        np.random.seed(self.splitting_seed)
        
        subjects = os.listdir(self.dataset_path)
        np.random.shuffle(subjects)

        n_test = math.floor(len(subjects) * self.test_percentage) 
        n_val = math.floor(len(subjects) * self.val_percentage)
        n_train = math.floor(len(subjects) * self.train_percentage) + 1  ## floor(18.4) = 18 so i add 1

        train = subjects[:n_train]
        val = subjects[n_train:n_train+n_val]
        test = subjects[n_train+n_val:]
        splitting_dict = {'train': train, 'val': val, 'test': test}

        return splitting_dict

    def get_image_label_dict(self):
        """
        From the list of patient in self.subjects_files return the list of image and tumor

        ** TO DO: extended for the sulci and falx **
        """
        image_label_dict = {'image': [], 'tumor': []}
        for subject in self.subjects_files:
            subject_path = os.path.join(self.dataset_path, subject)
            for item in os.listdir(os.path.join(self.dataset_path, subject, 'volume')):
                image_label_dict['image'].append(os.path.join(subject_path, 'volume', item))
                image_label_dict['tumor'].append(os.path.join(subject_path, 'tumor', item))
                
        ## check the leng of the list
        assert len(image_label_dict['image']) == len(image_label_dict['tumor']), 'The number of images and labels are different'

        return image_label_dict

    def data_augmentation(self, image, label=None):
        """
        Set of trasformation to apply to image.
        """
        ## Resize
        resize = transforms.Resize(size=self.size)
        image = resize(image)
        if label is not None: label = resize(label)
        
        ## random rotation to image and label
        if torch.rand(1) > 0.5:
            angle = np.random.randint(-30, 30)
            image = transforms.functional.rotate(image, angle)
            if label is not None: label = transforms.functional.rotate(label, angle)

        ## random translation to image and label in each direction
        if torch.rand(1) > 0.5:
            translate = transforms.RandomAffine.get_params(degrees=(0.,0.), 
                                                        translate=(0.10, 0.10),
                                                        scale_ranges=(1.0,1.0),
                                                        shears=(0.,0.), 
                                                        img_size=self.size)
            image = transforms.functional.affine(image, *translate)
            if label is not None: label = transforms.functional.affine(label, *translate)

        ## random horizontal flip
        if torch.rand(1) > 0.5:
            image = transforms.functional.hflip(image)
            if label is not None: label = transforms.functional.hflip(label)

        ## random vertical flip
        if torch.rand(1) > 0.5:
            image = transforms.functional.vflip(image)
            if label is not None: label = transforms.functional.vflip(label)
            
        ## random brightness and contrast
        if torch.rand(1) > 0.5:
            image = transforms.ColorJitter(brightness=0.5, contrast=0.5)(image)

        
        ## random gamma correction
        if torch.rand(1) > 0.5:
            gamma = np.random.uniform(0.5, 1.5)
            image = transforms.functional.adjust_gamma(image, gamma)
      
        image = transforms.functional.to_tensor(image)
        if label is not None: label = transforms.functional.to_tensor(label)
        image = (2 * image) - 1  

        if label is not None:
            return image, label
        else:
            return image

    def trasform(self, image, label):
        """
        Simple trasformaztion of the label and image. Resize and normalize the image and resize the label
        """
        ## Resize
        resize = transforms.Resize(size=self.size)
        image = resize(image)
        label = resize(label)

        ## convert to tensor and normalize
        label = torch.tensor(label)
        image = transforms.functional.to_tensor(image)
        image = (2 * image) - 1    
        return image, label


if __name__ == '__main__':
    import yaml
    
    conf = '/home/angelo/Documenti/Itraoperative-US/intraoperative_us/diffusion/conf/conf.yaml'
    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config['dataset_params'])
    dataset_config = config['dataset_params']

    dataset = IntraoperativeUS(size= [dataset_config['im_size_h'], dataset_config['im_size_w']],
                               dataset_path= dataset_config['dataset_path'],
                               im_channels= dataset_config['im_channels'], 
                               split='train',
                               splitting_seed=dataset_config['splitting_seed'],
                               train_percentage=dataset_config['train_percentage'],
                               val_percentage=dataset_config['val_percentage'],
                               test_percentage=dataset_config['test_percentage'])
                            
    dataset[0]
   


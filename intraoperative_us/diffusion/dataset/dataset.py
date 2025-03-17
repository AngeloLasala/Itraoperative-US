import glob
import os
import yaml


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
                 splitting_json=None,
                 condition_config=None, data_augmentation=False,
                 rgb=False):

        self.dataset_path = dataset_path

        # condition and data augmentation parameters
        self.condition_types = [] if condition_config is None else condition_config['condition_types']
        self.data_augmentation = data_augmentation
        
        #img parameters
        self.size = size
        self.im_channels = im_channels
        self.rgb = rgb

        #splitting parameters   
        self.splitting_seed = splitting_seed
        self.train_percentage = train_percentage
        self.val_percentage = val_percentage
        self.test_percentage = test_percentage

        ## get the splitting of the dataset
        self.split = split
        if splitting_json is None:
            self.splitting_dict = self.get_data_splittings()
        else:
            with open(os.path.join(os.path.dirname(self.dataset_path), splitting_json), 'r') as file:
                self.splitting_dict = json.load(file)
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
                if self.data_augmentation:
                    im_tensor, label_tensor = self.augmentation(im, label)
                else:
                    im_tensor, label_tensor = self.trasform(im, label)
                cond_inputs['image'] = label_tensor
            #####################################################################
            return im_tensor, cond_inputs   

        else: # no condition
            if self.data_augmentation:
                im_tensor = self.augmentation(im)
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
        if self.rgb:
            im = im.convert('RGB')

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
                
        return image_label_dict

    def augmentation(self, image, label=None):
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
        label = transforms.functional.to_tensor(label)
        image = transforms.functional.to_tensor(image)
        image = (2 * image) - 1    
        return image, label

class IntraoperativeUS_mask():
    """
    IntraoperativeUS dataset class for developing DL model for neurosurgical US images

    Load only mask for generating the mask for the tumor
    """
    def __init__(self, size, dataset_path, im_channels, split,
                 splitting_seed, train_percentage, val_percentage, test_percentage,
                 splitting_json=None,
                 data_augmentation=False,
                 condition_config=None):

        self.dataset_path = dataset_path

        # condition and data augmentation parameters
        self.data_augmentation = data_augmentation
        
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
        if splitting_json is None:
            self.splitting_dict = self.get_data_splittings()
        else:
            with open(os.path.join(os.path.dirname(self.dataset_path), splitting_json), 'r') as file:
                self.splitting_dict = json.load(file)
        self.subjects_files = self.splitting_dict[self.split]

        ## image and label list
        self.label_list = self.get_label_list()

    def __len__(self):
        return len(self.label_list)


    def __getitem__(self, index):
        label = self.get_label(index)

            
        if self.data_augmentation:
            label_tensor = self.augmentation(label)
        else:
            label_tensor = self.trasform(label)
        
        return label_tensor

       

    def get_label(self, index):
        """
        Return the image and label given the index
        """
        # read image and label with PIL
        label = Image.open(self.label_list[index])
        return label

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

    def get_label_list(self):
        """
        From the list of patient in self.subjects_files return the list of image and tumor

        ** TO DO: extended for the sulci and falx **
        """
        tumor_list = []
        for subject in self.subjects_files:
            subject_path = os.path.join(self.dataset_path, subject)
            for item in os.listdir(os.path.join(self.dataset_path, subject, 'tumor')):
                tumor_list.append(os.path.join(subject_path, 'tumor', item))
                
        return tumor_list

    def augmentation(self, label):
        """
        Set of trasformation to apply to image.
        """
        ## Resize
        resize = transforms.Resize(size=self.size)
        label = resize(label)
        
        ## random rotation to image and label
        if torch.rand(1) > 0.5:
            angle = np.random.randint(-30, 30)
            label = transforms.functional.rotate(label, angle)

        ## random translation to image and label in each direction
        if torch.rand(1) > 0.5:
            translate = transforms.RandomAffine.get_params(degrees=(0.,0.), 
                                                        translate=(0.10, 0.10),
                                                        scale_ranges=(1.0,1.0),
                                                        shears=(0.,0.), 
                                                        img_size=self.size)
            label = transforms.functional.affine(label, *translate)

        ## random horizontal flip
        if torch.rand(1) > 0.5:
            label = transforms.functional.hflip(label)

        ## random vertical flip
        if torch.rand(1) > 0.5:
            label = transforms.functional.vflip(label)
     
        label = transforms.functional.to_tensor(label)
        
        return label

    def trasform(self, label):
        """
        Simple trasformaztion of the label and image. Resize and normalize the image and resize the label
        """
        ## Resize
        resize = transforms.Resize(size=self.size)
        label = resize(label)
        
        ## convert to tensor and normalize
        label = transforms.functional.to_tensor(label)  
        return label

class GeneratedMaskDataset(torch.utils.data.Dataset):
    """
    Dataset of generated mask image loaded from the path
    """
    def __init__(self, par_dir, size, input_channels):
        self.par_dir = par_dir
        self.size = size
        self.input_channels = input_channels

        self.data_dir_label = par_dir
        self.files = [os.path.join(self.data_dir_label, f'x0_{i}.png') for i in range(len(os.listdir(self.data_dir_label)))]

    def __len__(self):
        return len(os.listdir(self.data_dir_label))

    def __getitem__(self, idx):
        image_path = self.files[idx]

        # read the image wiht PIL
        image = Image.open(image_path)
        resize = transforms.Resize(size=self.size)
        image = resize(image)
        if self.input_channels == 1: image = image.convert('L')
        image = transforms.functional.to_tensor(image)
        image = (image > 0.5).float()
        return image

class GenerateDataset(torch.utils.data.Dataset):
    """
    Dataset of generated image loaded from the path
    """
    def __init__(self, par_dir, trial, experiment, guide_w, scheduler, epoch,  size, input_channels,
                 mask=False):
        self.par_dir = par_dir
        self.trial = trial
        self.experiment = experiment
        self.guide_w = guide_w
        self.scheduler = scheduler
        self.epoch = epoch
        self.size = size
        self.input_channels = input_channels
        self.mask = mask

        self.data_ius= self.get_eco_path()
        self.files_data = [os.path.join(self.data_ius, f'x0_{i}.png') for i in range(len(os.listdir(self.data_ius)))]

    def __len__(self):
        return len(self.files_data)

    def __getitem__(self, idx):
        image_path = self.files_data[idx]

        # read the image wiht PIL
        image = Image.open(image_path)
        resize = transforms.Resize(size=self.size)
        image = resize(image)
        if self.input_channels == 1: image = image.convert('L')
        image = transforms.functional.to_tensor(image)
        image = (2 * image) - 1 

        if self.mask:
            mask_numb = image_path.split('/')[-1].split('.')[0].split('_')[1]
            mask_path = os.path.join(self.get_mask_images(), f'mask_{mask_numb}.png')
            print(mask_path.split('/')[-1], image_path.split('/')[-1])
            mask = Image.open(mask_path)
            mask = resize(mask)
            mask = mask.convert('L')
            mask = transforms.functional.to_tensor(mask)

            return image, mask

        else: 
            return image

    def get_eco_path(self):
        """
        retrive the path 'eco' from current directory
        """
        print(self.scheduler)
        data_ius = os.path.join(self.par_dir, self.trial, self.experiment, f'w_{self.guide_w}', self.scheduler, f'samples_ep_{self.epoch}','ius')
        return data_ius

    def get_mask_images(self):
        """
        Return mask and generated images
        """
        data_mask = self.get_eco_path().split('/')[0:-1] + ['masks']
        data_mask = os.path.join('/', *data_mask)
    
        return data_mask


if __name__ == '__main__':
    current_directory = os.path.dirname(__file__)
    par_dir = os.path.dirname(current_directory)
    conf = os.path.join(par_dir, 'conf', f'conf.yaml')

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
                               splitting_json=dataset_config['splitting_json'],
                               split='val',
                               splitting_seed=dataset_config['splitting_seed'],
                               train_percentage=dataset_config['train_percentage'],
                               val_percentage=dataset_config['val_percentage'],
                               test_percentage=dataset_config['test_percentage'],
                               condition_config=config['ldm_params']['condition_config'],
                               data_augmentation=False)

    print(dataset.splitting_dict)                            
    im, lab = dataset[10]
    print(im, lab)

    # convert in numpy and plot the image
    im = im.numpy().transpose(1,2,0)
    lab = lab['image'].numpy().transpose(1,2,0)
    plt.imshow(im, cmap='gray')
    plt.imshow(lab, cmap='jet', alpha=0.5)
    plt.show()


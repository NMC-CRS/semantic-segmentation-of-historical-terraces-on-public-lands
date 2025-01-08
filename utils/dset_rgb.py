#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Claudine Gravel-Miguel

This script contains the code required to create the PyTorch dataset

"""

import torch
from torch.utils.data import Dataset

import numpy as np

from skimage.io import imread  # this is from the scikit-learn library

from torchvision import transforms

# define the device from the start
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This is a torchvision transform 
tfms = transforms.Compose([
    transforms.ToTensor()
])

# This formats the images as PyTorch dataset
class SegData(Dataset):
    
    def __init__(self,
                 inputs_dim1: list,
                 inputs_dim2: list,
                 inputs_dim3: list,
                 targets: list,
                 backbone = None,
                 transform = None
                 ):
        self.inputs_dim1 = inputs_dim1
        self.inputs_dim2 = inputs_dim2
        self.inputs_dim3 = inputs_dim3
        self.targets = targets
        self.backbone = backbone
        self.transform = transform

    ''' 
    __len__ is required for PyTorch Dataset. Must retun a single, contsnat value after initialization.
    '''
    
    def __len__(self):
        return len(self.inputs_dim1)
    
    '''
    __getitem__ is also required. Takes an index and returns a tuble with sample data to be used for training or validation
    '''
    
    def __getitem__(self,
                    index: int):
        # Select the sample
        input_dim1_ID = self.inputs_dim1[index]
        input_dim2_ID = self.inputs_dim2[index]
        input_dim3_ID = self.inputs_dim3[index]
        target_ID = self.targets[index]

        # Load input and target as numpy arrays
        image_dim1, image_dim2, image_dim3, mask = imread(input_dim1_ID), imread(input_dim2_ID), imread(input_dim3_ID), imread(target_ID)
        
        # Combine the 3 images into one
        image = np.stack((image_dim1, image_dim2, image_dim3), 2)
        
        # Mask preprocessing (binarize values)
        mask[mask > 0] = 1.
        # Keep only one dimension from the mask (same code as for tensors)
        mask = mask.astype(np.int16) 
        
        # Transformations
        # Feeding both image and mask to transform makes sure that the pair (image-mask) is transformed the same way
        if self.transform is not None:
            transformed = self.transform(image = image, mask = mask) 
            image, mask = transformed["image"], transformed["mask"]
        
        if self.transform is not None:
            if self.backbone[0] == "V":
                # Sets the image type to floats16 (just for VGG)
                image = np.float16(image)

        return image, mask

    # Function called by dataloader
    def collate_fn(self, batch):
        ims, masks = list(zip(*batch))
        ims = torch.cat([tfms(im.copy())[None] for im in ims]).float().to(device)
        ce_masks = torch.cat([torch.Tensor(mask[None]) for mask in masks]).long().to(device)
        return ims, ce_masks

# THE END
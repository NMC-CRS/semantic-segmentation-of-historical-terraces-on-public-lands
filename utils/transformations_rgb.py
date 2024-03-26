#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Claudine Gravel-Miguel

This script defines the transformations done on images, using the albumentation package

"""

import albumentations as A
    
def train_augmentation(im_size):
    """
    Defines the transform that will be apply to the training data during the data loading workflow
    
        Parameters:
            im_size (int): The size of the width/height of training tiles, to resize to them after transformation

        Returns:
            A.Compose (Albumentation function): Function that runs the defined transformations

    """
    
    train_transform = [ 
        A.OneOf([
            A.Blur(p = .25),
            A.ShiftScaleRotate(shift_limit = 0.1, scale_limit = 0.2, rotate_limit = 30, p = .50),
            ], p = 0.9),
        A.HorizontalFlip(p = .50),
        A.VerticalFlip(p = .50),
        A.Rotate(limit = 90, p = .50),
        A.Resize(im_size, im_size), # makes sure that all tiles are of the correct format
    ]
    
    return A.Compose(train_transform)

# THE END

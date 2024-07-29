#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Claudine Gravel-Miguel

This script defines the transformations done on images, using the albumentation package

"""

import albumentations as A
    
def train_augmentation(im_size):
    '''
    Performs transformations on uploaded images. Each transformations has a certain probability of happening.
    The transformations are done one after the other, on the array resulting from the previous transformation (if transformed)

    Parameters
    ----------
    im_size : int
        Size of the input (height or width).

    Returns
    -------
    Albumentation Compose workflow
        The workflow that actually runs the images through the transformation when they are loaded to the model.

    '''
    
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

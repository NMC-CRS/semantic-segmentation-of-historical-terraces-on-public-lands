#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:18:23 2024

@author: Claudine Gravel-Miguel
@Description: This script separates the provided tiles dataset into training/validation/testing datasets. 
If the separation is random, it uses the sklearn train_test_split function to set 80% into training, and 10% in validation and testing, respectively.
If the separtion is geographic, it uses the train_bound to separate the datasets.

"""

from sklearn.model_selection import train_test_split

def separate_dataset(filenames, separation_random, train_bounds):
    
    """
    Separates the list from filenames into training and validation/testing datasets.
    If separation_random is set to True, the validation and testing datasets are different and each has 10% of the original datasets
    If separation_random is set to False, the training dataset is tiles within the train_bounds, and the rest is both validation and testing datasets (same dataset)
    
        Parameters:
            filenames (list): List of tile names
            separation_random (boolean): If the tiles are separated randomly (80-10-10) or not
            train_bounds (list): xmin, ymin, xmax, and ymax around the tiles that will be used for training
        Returns:
            inputs_train (list): List of image tile names that are used for training
            inputs_val (list): List of image tile names that are used for validation
            inputs_test (list): List of image tile names that are used for testing
            targets_train (list): List of mask tile names that are used for training (identical to inputs_train)
            targets_val (list): List of mask tile names that are used for validation (identical to inputs_val)
            targets_test (list):  List of mask tile names that are used for testing (identical to inputs_test)

    """
    
    if separation_random:

        # Set the random seed to make sure that we separate the paired input-targets similarly.
        random_seed = 42
        
        # Split dataset into training, validation, and testing set
        train_size = 0.8  # 80:10:10 split
        
        # Randomly put the list of file names into their respective lists *80% in training and 20% in validation).
        inputs_train, inputs_val, targets_train, targets_val = train_test_split(
            filenames,
            filenames,
            random_state = random_seed,
            train_size = train_size,
            shuffle = True)
        
        # Further separate the validation dataset in two (50% validation and 5% testing)
        inputs_val, inputs_test, targets_val, targets_test = train_test_split(
            inputs_val,
            targets_val,
            random_state = random_seed,
            train_size = 0.5,
            shuffle = True)

    else:
        # Create empty lists that will take the appropriate filenames
        inputs_train = []
        inputs_val = []
        inputs_test = []
        
        # Iterate through all filenames (tiles)
        for file in filenames:
            
            # Get the coordinates of each file (lower left anchor) from their name
            file_cleaned = file.removeprefix("Tile").removesuffix(".tif")
            file_split = file_cleaned.split("_")
            y_orig = float(file_split[1])
            x_orig = float(file_split[0])
            
            # Add the filename to its appropriate list based on its coordinates, compared to the train_bounds provided
            # Every tile within the train_bounds goes into the training dataset, everything else goes in both validation and testing datasets
            if x_orig >= train_bounds[0] and y_orig >= train_bounds[1] and x_orig < train_bounds[2] and y_orig < train_bounds[3]:
                inputs_train.append(file)
            else:
                inputs_val.append(file)
                inputs_test.append(file)
        
        # Because the entities separated are filenames, they are the same for inputs and targets, so we can simply copy the lists.
        targets_train = inputs_train
        targets_val = inputs_val
        targets_test = inputs_test
        
    return inputs_train, inputs_val, inputs_test, targets_train, targets_val, targets_test

# THE END
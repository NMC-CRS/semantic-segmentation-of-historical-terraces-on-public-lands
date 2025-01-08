# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:44:42 2023

@author: Claudine Gravel-Miguel

This script formats the training tiles, separates them to training/validation/testing, and trains the UNet model.
The main() function at the end calls all other functions (and other imported functions) in the required order and with the required parameters

"""

# SETUP

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch_snippets import Report
from torch.optim.lr_scheduler import ReduceLROnPlateau # Addition to adjust lr

import pandas as pd

import os

from datetime import datetime
import time

import numpy as np

import random

import calendar
    
# Importing my own functions
import calc_mask_perc as cm
import loss_functions as lf
import dset_rgb
import transformations_rgb as A
import separate_datasets as sepdata
import set_lr_parameters as setlr
import clean_datasets as clean
    
def import_backbone(backbone):
    
    '''
    This imports the appropriate backbone structure as the variable UNet. 
    Each structure is in a different python script, so this finds the right file to import.

    Parameters
    ----------
    backbone : str
        The name of the backbone to use.

    Returns
    -------
    UNet : Pytorch model
        The UNet model structure.

    '''

    if backbone == "VGG16":
        from unet_backbones.unet_vgg16 import UNet
    elif backbone == "VGG19":
        from unet_backbones.unet_vgg19 import UNet
    elif backbone == "ResNet18":
        from unet_backbones.unet_resnet18 import UNet
        
    return UNet

def train_model(model, n_epochs, loss_fun, lr_variable, trn_dl, val_dl, criterion, new_optimizer, writer):
    '''
    This iterates through the epochs to:
        * Run the training dataset (image and targets) through the model in batches, and computes metrics on the model predictions.
        * Use the difference between the predictions and the actual targets to adjust the weights of the model nodes.
        * Run the validation dataset (image and targets) through the model in batches and computes metrics here as well.

    Parameters
    ----------
    model : PyTorch model
        The model structure being trained.
    n_epochs : int
        Number of epochs to train the model. The model sees all the training images in each epoch..
    loss_fun : str
        The name of the loss function that is used to optimize the model training.
    lr_variable : bool
        If the learning rate is updating when val_loss stagnates (True) or kept constant at 0.001 (False).
    trn_dl : Pytorch Dataloader
        The workflow through which the training images and their targets go through when passed to the model.
    val_dl : Pytorch Dataloader
        The workflow through which the validation images and their targets go through when passed to the model.
    criterion : function
        A function that defined which function to use based on the loss_function chosen. This calculates the loss..
    new_optimizer : PyTorch optimizer
        Optimizer of the parameters that are learning, which gets updated during training.
    writer : bool
        If True, the code creates a Tensorboard file and log metrics to it..

    Returns
    -------
    None.

    '''
    
    # Set up the lr scheduler to update if the val_loss stagnates for more than 2 epochs.
    if lr_variable:
        scheduler = ReduceLROnPlateau(new_optimizer, 'min', patience = 2)
    
    # Record the time before training starts to calculate how long training took.
    start_time = datetime.now()
    
    # To follow up progress
    log = Report(n_epochs)
    
    # Iterate through for each epoch
    for epoch in range(n_epochs):
        
        # Set the model on training mode
        model.train()
        N_batch = len(trn_dl) # number of training batches
        current_epoch = epoch + 1 # for clearer code below
        
        # Set up variables to get average metrics of all batches in each epoch
        total_train_loss = 0
        total_train_recall = 0
        total_train_precision = 0
        total_train_f1 = 0
        total_train_acc = 0
        total_train_mcc = 0
        
        total_val_loss = 0
        total_val_recall = 0
        total_val_precision = 0
        total_val_f1 = 0
        total_val_acc = 0
        total_val_mcc = 0
        
        # Iterate through each training batch
        for batch, data in enumerate(trn_dl):
            
            # Get the data in the batch and train the model on it
            ims, ce_masks = data
            _masks = model(ims)
            
            # Reset the optimizer
            new_optimizer.zero_grad()
            
            # Compute the metrics
            loss, acc, recall, precision, f1, mcc = criterion(_masks, ce_masks, loss_fun)
            
            # Add up the batches' metrics
            total_train_loss += loss.item()
            total_train_recall += recall.item()
            total_train_precision += precision.item()
            total_train_f1 += f1.item()
            total_train_acc += acc.item()
            total_train_mcc += mcc.item()
    
            # Go backward through the model and update the weights
            loss.backward()
            new_optimizer.step()
            
            # Still useful to follow progress
            log.record(epoch+(batch + 1)/N_batch, trn_loss=loss.item(), end='\r')
            
        # Metrics for outputs (no need to update at every batch)
        tr_loss = total_train_loss/N_batch
        tr_acc = total_train_acc/N_batch
        tr_recall = total_train_recall/N_batch
        tr_precision = total_train_precision/N_batch
        tr_f1 = total_train_f1/N_batch
        tr_mcc = total_train_mcc/N_batch
        
        # If save_weights, write metrics to Tensorboard file
        if writer is not None:
            writer.add_scalar("Loss/train", tr_loss, current_epoch)
            writer.add_scalar("Recall/train", tr_recall, current_epoch)
            writer.add_scalar("Precision/train", tr_precision, current_epoch)
            writer.add_scalar("F1/train", tr_f1, current_epoch)
            writer.add_scalar("Accuracy/train", tr_acc, current_epoch)
            writer.add_scalar("MCC/train", tr_mcc, current_epoch)
    
        # Set the model on Evaluation mode
        model.eval()
        N_batch = len(val_dl) # Reset the number of batches (different dataset than training)
        with torch.no_grad(): # Save memory
            # Iterate through each validation batch
            for batch, data in enumerate(val_dl):
                
                # Get the data and run it through the trained model
                ims, masks = data
                _masks = model(ims)
                
                # Compute metrics
                loss, acc, recall, precision, f1, mcc = criterion(_masks, masks, loss_fun)
                
                # To follow progress (print values to Console as they get updated)
                log.record(epoch+(batch + 1)/N_batch, val_loss=loss.item(), end='\r')
                
                # Add up the batches' metrics
                total_val_loss += loss.item()
                total_val_recall += recall.item()
                total_val_precision += precision.item()
                total_val_f1 += f1.item()
                total_val_acc += acc.item()
                total_val_mcc += mcc.item()
            
        # For outputs (no need to update at every batch)
        val_loss = total_val_loss/N_batch
        val_acc = total_val_acc/N_batch
        val_recall = total_val_recall/N_batch
        val_precision = total_val_precision/N_batch
        val_f1 = total_val_f1/N_batch
        val_mcc = total_val_mcc/N_batch
        
        # If save_weights, write metrics to Tensorboard file
        if writer is not None:
            writer.add_scalar("Loss/val", val_loss, current_epoch)
            writer.add_scalar("Recall/val", val_recall, current_epoch)
            writer.add_scalar("Precision/val", val_precision, current_epoch)
            writer.add_scalar("F1/val", val_f1, current_epoch)
            writer.add_scalar("Accuracy/val", val_acc, current_epoch)
            writer.add_scalar("MCC/val", val_mcc, current_epoch)
        
        # Set up variable to print the lr to Console
        curr_lr = new_optimizer.param_groups[0]['lr']
        
        # Calculate the time since training began
        time_since_start = datetime.now() - start_time
        
        # Print metrics, lr, and time on the Console if it's the first, last epoch or each 10 epochs in between
        if (current_epoch) == 1 or (current_epoch)%10 == 0 or current_epoch == n_epochs:
            log.report_avgs(current_epoch) # Those are averages, not the final value...
        
            print(f'LR: {curr_lr}, time: {time_since_start}')
            print(f'TRAIN loss: {tr_loss:.3f}, recall: {tr_recall:.3f}, precision: {tr_precision:.3f}, F1: {tr_f1:.3f}, MCC: {tr_mcc:.3f}, accuracy: {round(tr_acc, 3)}')
            print(f'VAL loss: {val_loss:.3f}, recall: {val_recall:.3f}, precision: {val_precision:.3f}, F1: {val_f1:.3f}, MCC: {val_mcc:.3f}, accuracy: {round(val_acc, 3)}\n')
        
        # Change the learning rate if val_loss plateaus
        if lr_variable:
            scheduler.step(val_loss)

    # Clears the writer's memory
    if writer is not None:
        writer.flush()
    
    return

def test_model(model, train_dir_dim1, train_dir_dim2, train_dir_dim3, mask_dir, inputs_test, backbone, batch_size):
    '''
    This iterates through the testing dataset (image and targets). It runs them through the trained model in batches, and computes metrics on the model predictions.

    Parameters
    ----------
    model : PyTorch model
        The model structure being trained.
    train_dir_dim1 : str
        Path to the tiles of the first visualization.
    train_dir_dim2 : str
        Path to the tiles of the second visualization.
    train_dir_dim3 : str
        Path to the tiles of the third visualization.
    mask_dir : str
        Path to the mask tiles.
    inputs_test : list
        List of the filenames of the testing dataset.
    backbone : str
        The name of the backbone to use.
    batch_size : int
        Number of images that will be uploaded to the model at the same time.

    Returns
    -------
    None.

    '''
    
    # Attach the names of each testing tile to its appropriate path
    inputs_test_dim1 = [f'{train_dir_dim1}/{item}' for item in inputs_test] 
    inputs_test_dim2 = [f'{train_dir_dim2}/{item}' for item in inputs_test] 
    inputs_test_dim3 = [f'{train_dir_dim3}/{item}' for item in inputs_test] 
    targets_test = [f'{mask_dir}/{item}' for item in inputs_test] 

    # Workflow to format and upload the testing data 
    tst_ds = dset_rgb.SegData(inputs_test_dim1, inputs_test_dim2, inputs_test_dim3, targets_test, backbone)
    tst_dl = DataLoader(tst_ds, batch_size=batch_size, shuffle=True, collate_fn=tst_ds.collate_fn)
    
    # Set up empty list variables that will be populated in iteration.
    list_recall = []
    list_precision = []
    list_f1 = []
    list_mcc = []
    
    # Set the model on Evaluation mode
    model.eval()
    with torch.no_grad(): # Save memory
        # Iterate through each testing batch
        for batch, data in enumerate(tst_dl):
            
            # Get the data in the batch
            im, mask = data
            
            # Run the tiles through the model
            _mask = model(im)
    
            # Calculate metrics between predicted and actual
            recall, precision, f1, mcc = lf.compute_metrics(_mask, mask)
            
            # Get the metrics from device
            recall = recall.detach().cpu()
            precision = precision.detach().cpu()
            f1 = f1.detach().cpu()
            mcc = mcc.detach().cpu()
            
            # Append the metrics to the lists
            list_recall.append(recall)
            list_precision.append(precision)
            list_f1.append(f1)
            list_mcc.append(mcc)

    # Calculate the mean of the lists
    mean_recall = np.mean(list_recall)
    mean_precision = np.mean(list_precision)
    mean_f1 = np.mean(list_f1)
    mean_mcc = np.mean(list_mcc)

    # Print the results to Console
    print(f'TEST recall: {mean_recall:.3f}, precision: {mean_precision:.3f}, F1: {mean_f1:.3f}, MCC: {mean_mcc:.3f}')
    
    return 

def main(backbone, vis1, vis2, vis3, im_size, buffer_size, data_path, mask_folder_name, threshold, batch_size, separation_random, train_bounds, n_epochs, loss_fun, log_metrics, save_weights, output_path, lr_variable, remove_overlap):
    
    '''
    Calls all the functions to load the model structure and its backbone, format the different datasets, train the model, evaluate it, and test it.
    This function is called by the 02_one_script_to_rule_them_all.py script.

    Parameters
    ----------
    backbone : str
        Name of the backbone used for training.
    vis1 : str
        Name of the first visualization. This is to find the correct folder to upload.
    vis2 : str
        Name of the second visualization. This is to find the correct folder to upload.
    vis3 : str
        Name of the third visualization. This is to find the correct folder to upload.
    im_size : int
        Size of the training tiles' height (or width).
    buffer_size : int
        Buffer size (in meters) around the annotated object. This is to find the correct folder to upload.
    data_path : str
        Path to the CNN_input folder that holds inputs folders as well as the mask subfolder identified in mask_folder_name (see below).
    mask_folder_name : str
        Name of the folder that holds the masks folders. This is to find the correct folder to upload.
    threshold : int
        For pre-processing. This removes any mask images that have less than the threshold's number of positive's values.
    batch_size : int
        Number of images that will be uploaded to the model at the same time.
    separation_random : bool
        If set to True, the training/validation/testing datasets are separated randomly (80-10-10). If False, they are separated geographically based on train_bounds (see below).
    train_bounds : list
        xmin, ymin, xmax, and ymax around the tiles that will be used for training.
    n_epochs : int
        Number of epochs to train the model. The model sees all the training images in each epoch.
    loss_fun : str
        Name of the loss function to optimize (e.g., dice, iou, focal).
    log_metrics : bool
        If True, the code creates a Tensorboard file and log metrics to it.
    save_weights : bool
        If True, the model saves the trained weights to a .pt file.
    output_path : str
        Path to the CNN_output folder in which the predicted tiles, compiled tiff, compiled shapefiles will be saved.
    lr_variable : bool
        If the learning rate is updating when val_loss stagnates (True) or kept constant at 0.001 (False).
    remove_overlap : bool
        If we clean the datasets to remove any overlap between them

    Returns
    -------
    filename : str
        Name that is used for weights file, the folder that holds predicted tiles, the compiled geotiff, and the compiled shapefile.
    inputs_test : list
        List of the testing dataset to use to compute object-by-object metrics when weights are saved (done by another script).

    '''
    
    # Define the device from the start
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sets seeds for reproducibility and comparability
    seed = 3
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Define paths to each visualization dataset
    train_dir_dim1 = os.path.join(data_path, f'Input_{vis1}_{im_size}')
    train_dir_dim2 = os.path.join(data_path, f'Input_{vis2}_{im_size}')
    train_dir_dim3 = os.path.join(data_path, f'Input_{vis3}_{im_size}')
    mask_dir = os.path.join(data_path, f'{mask_folder_name}/Target_{buffer_size}m_{im_size}')
    
    # REFINE THE DATASET OF TILES

    # Create a table with the size of annotations in each mask tile
    tiles_table = cm.log_mask_tiles(mask_dir)
    
    # Get the list of tiles with big enough annotations from the table created above
    filtered_df = tiles_table[tiles_table['min_nonzero'] > threshold]
    filenames = filtered_df['filename'].values.tolist()

    # Assign tiles to training or validation/testing datasets
    inputs_train, inputs_val, inputs_test = sepdata.separate_dataset(filenames, separation_random, train_bounds)

    if remove_overlap:
        ## Clean the datasets
        # Calculate the min_distance based on the resolution and im_size
        min_distance = clean.calculate_min_distance(f'{train_dir_dim1}/{inputs_train[0]}')
        
        # Use that min_distance to clean overlapping tiles
        inputs_train, inputs_val = clean.clean_overlapping_tiles(inputs_train, inputs_val, min_distance)
        inputs_train, inputs_test = clean.clean_overlapping_tiles(inputs_train, inputs_test, min_distance)
        inputs_val, inputs_test = clean.clean_overlapping_tiles(inputs_val, inputs_test, min_distance)

    # Calculate the number of cleaned tiles used.
    n_tiles_used = len(inputs_train) + len(inputs_val) + len(inputs_test)

    print(f"\nUsing {n_tiles_used} tiles with objects in them.")
    print(f"Out of the {n_tiles_used} tiles, {len(inputs_train)}({round(len(inputs_train)/n_tiles_used, 2)}) are for training, {len(inputs_val)}({round(len(inputs_val)/n_tiles_used,2)}) are for validation, and {len(inputs_test)}({round(len(inputs_test)/n_tiles_used,2)}) for testing.")

    # Add the paths to the filenames in each category
    # TRAINING
    inputs_train_dim1 = [f'{train_dir_dim1}/{item}' for item in inputs_train] 
    inputs_train_dim2 = [f'{train_dir_dim2}/{item}' for item in inputs_train] 
    inputs_train_dim3 = [f'{train_dir_dim3}/{item}' for item in inputs_train] 
    targets_train = [f'{mask_dir}/{item}' for item in inputs_train] 

    # VALIDATION
    inputs_val_dim1 = [f'{train_dir_dim1}/{item}' for item in inputs_val] 
    inputs_val_dim2 = [f'{train_dir_dim2}/{item}' for item in inputs_val] 
    inputs_val_dim3 = [f'{train_dir_dim3}/{item}' for item in inputs_val] 
    targets_val = [f'{mask_dir}/{item}' for item in inputs_val] 
    
    # CREATE THE DATALOADER

    # Workflow to create the PyTorch dataset (these are still numpy arrays)
    trn_ds = dset_rgb.SegData(inputs_train_dim1, inputs_train_dim2, inputs_train_dim3, targets_train, backbone = backbone, transform = A.train_augmentation(im_size))
    val_ds = dset_rgb.SegData(inputs_val_dim1, inputs_val_dim2, inputs_val_dim3, targets_val, backbone = backbone)

    # Workflow to load the images in batches
    trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True, collate_fn=trn_ds.collate_fn)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=val_ds.collate_fn)

    # CREATE THE MODEL STRUCTURE
    
    # Import the model backbone and pretrained weights
    UNet = import_backbone(backbone)
    
    # Move model to device
    model = UNet().to(device)
    
    # Set the LR parameters
    pretrained_optimizer, new_optimizer = setlr.setup_parameter_learning_rate("UNet", model)
    
    # Get the number of output channels/classes and print it to Console
    num_classes = model.out.out_channels
    print("Number of classes:", num_classes)

    # Set up the loss function as a separate variable
    criterion = lf.UnetLoss # this computes focal loss and accuracy

    if lr_variable:
        lr_type = "lrVariable"
    else:
        lr_type = "lrStable"
    
    # DEFINE THE FILENAME

    # Define the timestamp for the filename
    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)

    # Define the filename from parameter values
    filename = f"UNet_{backbone}_{n_epochs}ep_{buffer_size}m_{loss_fun}_{batch_size}bs_{lr_type}_{vis1}_{vis2}_{vis3}_{threshold}Thresh_{im_size}_{time_stamp}"
    print(f'\nThe weight file name of this training is: {filename}\n')

    # Define the writer if necessary
    if log_metrics:
        writer = SummaryWriter(log_dir = f"{output_path}/Tensorboard_files/{filename}")
    else:
        writer = None

    '''
    TRAIN AND VALIDATE MODEL
    '''
    
    train_model(model = model, 
                n_epochs = n_epochs, 
                loss_fun = loss_fun, 
                lr_variable = lr_variable, 
                trn_dl = trn_dl,
                val_dl = val_dl, 
                criterion = criterion, 
                new_optimizer = new_optimizer,
                writer = writer)

    if log_metrics:
        writer.close()

    '''
    TEST MODEL
    '''

    # Evaluate on testing dataset
    test_model(model, train_dir_dim1, train_dir_dim2, train_dir_dim3, mask_dir, inputs_test, backbone, batch_size)

    # Get the list of test tiles' paths for the polygons.
    inputs_test_dim1 = [f'{train_dir_dim1}/{item}' for item in inputs_test]
    # Create a polygon that represents each dataset and write to disk
    print("Create training polygon")
    gdf_train_polygons = clean.create_poly_from_tiles(inputs_train_dim1, "Training")
    gdf_val_polygons = clean.create_poly_from_tiles(inputs_val_dim1, "Validation")
    gdf_test_polygons = clean.create_poly_from_tiles(inputs_test_dim1, "Testing")

    # Join the geodatabases of dataset areas and export to disk
    joined_gdf = pd.concat([gdf_train_polygons, gdf_val_polygons, gdf_test_polygons])
    joined_gdf.to_file(f"{output_path}/Model_predictions/{filename}_areas.gpkg", driver="GPKG")

    if save_weights:
        # To save the model's trained weights
        torch.save(model.state_dict(), f"{output_path}/Model_weights/{filename}.pt")
    
    # Return the filename (holds lots of metadata) and the inputs_test list, which can be used to create a map of predictions on test data
    return filename, inputs_test

# PRACTICE CODE

# filename, inputs_test = main(backbone = "VGG16",
#      vis1 = "Slope",
#      vis2 = "PosOp",
#      vis3 = "TRI", 
#      im_size = 256,
#      buffer_size = 20,
#      data_path = '''REPLACE WITH PATH TO CNN_input FOLDER''',
#      mask_folder_name = "Terrace_masks",
#      threshold = 1000,
#      batch_size = 8,
#      separation_random = False,
#      train_bounds = [0, 1, 0, 1],
#      n_epochs = 2,
#      loss_fun = "iou",
#      log_metrics = True,
#      save_weights = True,
#      output_path = '''REPLACE WITH PATH TO CNN_output FOLDER''',
#      lr_variable = False, 
#      remove_overlap = True)

# THE END
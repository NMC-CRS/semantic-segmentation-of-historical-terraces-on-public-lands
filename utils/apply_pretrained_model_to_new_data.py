#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:18:42 2023

@author: Claudine Gravel-Miguel

This script holds most of the functions that are used to: 
1. Upload a pretrained model
2. Run new tiles (or the testing dataset) through that pretrained model, 
2. Merge the tiles created into one raster saved as a geotiff, vectorize the raster, and save that vector as a shapefile
3. Calculate the metrics per objects by comparing the raster prediction with the annotated prediction (as shp) if applicable

Its two main functions (main_with_metrics and main_without_metrics) were kept separate because they have different variable requirements.

"""

# load the packages required
import rasterio
from rasterio.merge import merge

import torch
import os
from skimage.io import imread
import numpy as np
from torchvision import transforms

# Import helper functions already created
from unet_main_script import import_backbone
from separate_datasets import separate_dataset
import calculate_metrics

# Define which device to use based on the computer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(image_dim1, image_dim2, image_dim3):
    '''
    Combine the 3 dimensions provided into a 3D array, transform into a tensor, and format its dimensions appropriately for the model

    Parameters
    ----------
    image_dim1 : numpy array
        The first visualization tile imported as numpy array.
    image_dim2 : numpy array
        The second visualization tile imported as numpy array.
    image_dim3 : numpy array
        The third visualization tile imported as numpy array.

    Returns
    -------
    im : Pytorch tensor
        Tensor of dimensions [B, C, H, W] where B stands for the batch size.

    '''
    
    
    # This is a torchvision transform to transform the numpy array into a Pytorch tensor
    tfms = transforms.Compose([
        transforms.ToTensor()
    ])    

    # format the images to fit what we trained the model on.
    im = np.stack((image_dim1, image_dim2, image_dim3), 2)
    im = tfms(im)
    
    # Add the first dimension (batch size)
    im = im.unsqueeze(0) 
    
    # Return the pytorch tensor
    return im

def postprocess_UNet(mask):
        
    '''
    Transform the tile predicted by the model into a 1 dimension float32 numpy array sent to CPU (for possible visualization in Python and elsewhere)

    Parameters
    ----------
    mask : Pytorch tensor
        The tensor of objects prediction created by the model.

    Returns
    -------
    mask : numpy array
        Array of dimension [H, W] and float32 type.

    '''
    
    # Remove the one-hot encoding
    _max, mask = torch.max(mask, dim=1)

    # Transform to numpy array
    mask = mask.detach().cpu().numpy()
    
    # Remove the batch dimension
    mask = mask.squeeze()
    
    # Change the type to float32, so that those tiles can be read in QGIS or other programs
    mask = mask.astype("float32") 
    
    return(mask)


def predict_on_new_tiles(model_structure, model, filename, tilenames, inputs_test_dim1, inputs_test_dim2, inputs_test_dim3, output_path):
    
    '''
    Iterate through all the files in the filenames list. For each filename, import the three visualizations as numpy arrays.
    Call functions to format the 3 bands of visualization into a 3D Pytorch
    Run that 3D tensor through the trained model, which creates a prediction tensor
    Call functions to format that prediction tensor into a 1D binary numy array.
    Extract the CRS of one of the input visualization map and assign that CRS profile to the new predicted array.
    Write the georeferenced prediction to a geotiff file.

    Parameters
    ----------
    model_structure : str
        Name of the model structure.
    model : Pytorch model
        Pretrained model structure with loaded weights.
    filename : str
        Name of the weights file.
    tilenames : list
        List of filenames of tiles to run through the model.
    inputs_test_dim1 : str
        Name of the first visualization type.
    inputs_test_dim2 : str
        Name of the second visualization type.
    inputs_test_dim3 : str
        Name of the third visualization type.
    output_path : str
        Path to the CNN_output/Model_predictions folder that will hold the geotiffs created.

    Returns
    -------
    None.

    '''
    
    # Check if the output folder exists and create a new one if it doesn't
    isExist = os.path.exists(output_path)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(output_path)
       print("Created a new folder for the predictions")
    
    # Print statement to follow progress
    print(f"\nRunning {len(tilenames)} tiles through the model to create predictions. This may take a few minutes.\n")
    
    # Iterate through the list of filenames provided
    for file in tilenames:
        if file.endswith(".tif"):
            
            # Open the tiles with same name different band of same tile)
            image_dim1 = imread(os.path.join(inputs_test_dim1, file))
            image_dim2 = imread(os.path.join(inputs_test_dim2, file))
            image_dim3 = imread(os.path.join(inputs_test_dim3, file))
            
            # Preprocess the image to fit the trained format
            im = preprocess(image_dim1, image_dim2, image_dim3)
        
            # Load the image to the device where the model is loaded
            im = im.to(torch.device(device))
            
            with torch.no_grad():
                # Create the prediction of the mask using the trained model
                mask = model(im)
            
            # Prostprocess the mask (transform into a 1-dim numpy array)
            if model_structure == "UNet":
                mask = postprocess_UNet(mask)
        
            # georeference the predicted mask
            with rasterio.open(os.path.join(inputs_test_dim1, file)) as src:
                # Extract the profile of one of the vizualization tiles used as source
                profile = src.profile
                profile.update(driver = 'GTiff', dtype = mask.dtype)
                # Set nodata as 0
                profile.update(nodata = 0) 

                # Apply the profile extracted to the new image and write it to a new file.
                with rasterio.open(os.path.join(output_path, file), "w", **profile) as output_file:
                    output_file.write(mask, indexes = 1)
            
    return

def merge_tiles(pred_tiles_path, output_file):
    
    '''
    Merge all the predicted geotiffs into one big geotiff

    Parameters
    ----------
    pred_tiles_path : str
        Path to the predicted geotiff tiles.
    output_file : str
        Name of the big geotiff to create with the merged tiles.

    Returns
    -------
    None.

    '''
    
    # Get a list of all the raster files ending with ".tif" in the input folder
    file_list = [f for f in os.listdir(pred_tiles_path) if f.endswith('.tif')]
    
    # Create an empty list that will take on all merged tiles
    raster_to_mosaic = []
    
    # Loop through the raster files, open them, and append them to the list
    for p in file_list:
        raster = rasterio.open(os.path.join(pred_tiles_path, p))
        raster_to_mosaic.append(raster)
    
    # Merge them
    mosaic, output = merge(raster_to_mosaic)
    
    # Update the metadata of the merged raster
    out_meta = raster.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": output,
    })
    
    # Write the merged raster to the output file
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    # Print statement to follow progress
    print("Merged raster file created: {}".format(output_file))
    
    return
    
def import_and_load_model_weights(model_structure, device, backbone, weights_path, filename):
    
    '''
    Import the model structure and its weights
    Load them and send them to the appropriate device

    Parameters
    ----------
    model_structure : str
        Name of the model structure (e.g., "UNet", "MaskRCNN", "FasterRCNN").
    device : str
        Name of the device ("cuda" or "cpu") depending on the computer.
    backbone : str
        Name of the backbone used to create the trained weights to import.
    weights_path : str
        Path to the folder that holds the weights files or to the .pt file that holds the model weights directly.
    filename : str
        Name of the weights file (if not already provided in weights_path). Can be None if it's included in weight_path.

    Returns
    -------
    model : Pytorch model
        Pytorch model structure with loaded weights and already sent to the proper device..

    '''
    
    # If a filename is provided (when called by main_with_metrics), use it to define the weights_path
    if filename != None:
        weights_path = f'{weights_path}/{filename}.pt'
    
    if model_structure == "UNet":
        # Import the UNet structure
        UNet = import_backbone(backbone)
        model = UNet() 

    # Load the trained weights to the appropriate device and apply to the model
    if device == "cuda":
        checkpoint = torch.load(weights_path)
    else:
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    # Send the model to the device (CUDA if on PC)
    model.to(device)
    
    # Return the loaded model sent to the device
    return model
 
def main_with_metrics(filename, cnn_output_path, data_path, separation_random, train_bounds, inputs_test, path_to_shp, threshold):
    
    '''
    Call most of the functions above to:
        Load the model and its weights, and send to the appropriate device
        Set the model to eval mode
        Iterate through the testing dataset and run them through the model to create predictions, which are exported as geotiff tiles
        Merge all predicted tiles into one big raster saved as a geotiff, vectorize it and save it as a shapefile
        Use the provided shapefile of annotated objects to calculate the object-per-object metrics using the provided threshold

    Parameters
    ----------
    filename : str
        Name of pretrained file (weights) that holds metadata.
    cnn_output_path : str
        Path to the CNN_output folder.
    data_path : str
        Path to the CNN_input folder that holds the visuzliation tiles' folders.
    separation_random : bool
        How the training/validation/testing datasets were separated. True for random, False for geographical.
    train_bounds : list
        Geographical bounds of the training dataset if separation_random is set to False.
    inputs_test : list
        List of the tile names that were set as testing dataset when running the model training (to keep the same).
    path_to_shp : str
        Path to the shapefile that holds the actual annotated objects to compare against the predictions.
    threshold : int
        All predicted polygons with area < that value will be deleted before calculating object-per-object metrics.

    Returns
    -------
    None.

    '''
    
    # Define some local variables
    # The path where the predicted tiles will be saved using the information we have here
    pred_tiles_path = f'{cnn_output_path}/Model_predictions/{filename}'
    # The path and name of the merged raster to create
    path_to_ras =  f'{pred_tiles_path}.tif'
    
    # Parse the filename to get the visualization maps names and backbone
    filename_list = filename.split("_")
    model_structure = filename_list[0]
    backbone = filename_list[1]
    vis1 = filename_list[7]
    vis2 = filename_list[8]
    vis3 = filename_list[9]
    im_size = filename_list[11]
    
    # Import and load the model and weights, and send to correct device
    model = import_and_load_model_weights(model_structure, device, backbone, f'{cnn_output_path}/Model_weights', filename)

    # Set the model to eval to avoid training it again.
    model.eval()
    
    # Add paths to datasets so the model knows where to look
    train_dir_dim1 = os.path.join(data_path, f'Input_{vis1}_{im_size}')
    train_dir_dim2 = os.path.join(data_path, f'Input_{vis2}_{im_size}')
    train_dir_dim3 = os.path.join(data_path, f'Input_{vis3}_{im_size}')

    # Run each test tile through the model and create predictions, which are saved as geotiffs
    predict_on_new_tiles(model_structure, model, filename, inputs_test, train_dir_dim1, train_dir_dim2, train_dir_dim3, pred_tiles_path)
    
    # Merge the predicted tiles to create one raster
    merge_tiles(pred_tiles_path, path_to_ras)
    
    # Calculate the metrics on the new predicted raster
    calculate_metrics.compute_metrics(path_to_shp, path_to_ras, threshold)
    
    return

def main_without_metrics(path_to_weights_file, data_path):
    
    '''
    Call most of the functions above to:
        Load the model and its weights, and send to the appropriate device
        Set the model to eval mode
        Iterate through the provided dataset and run them through the model to create predictions, which are exported as geotiff tiles
        Merge all predicted tiles into one big raster saved as a geotiff, vectorize it and save it as a shapefile

    Parameters
    ----------
    path_to_weights_file : str
        Path to the weights file that will be used (pretrained model).
    data_path : str
        Path to the CNN_input folder that holds the visuzliation tiles' folders.

    Returns
    -------
    path_to_ras : str
        Path to the raster created (if applicable). Used in the 03_predict_on_new_data.py script.

    '''
    
    # Define some local variables using the information provided
    # Get the filename from the path to weights
    path_sections = path_to_weights_file.split("/")
    weights_file = path_sections[-1]
    filename = weights_file.replace(".pt", "")
    # print(filename)
    
    # Parse the filename to get the visualization maps and backbone
    filename_list = filename.split("_")
    model_structure = filename_list[0]
    backbone = filename_list[1]
    vis1 = filename_list[7]
    vis2 = filename_list[8]
    vis3 = filename_list[9]
    im_size = filename_list[11]

    # Define the paths to the folder where we save cnn outputs and the name of the raster created
    cnn_output_path = f'{data_path}/CNN_output'
    output_path = f'{cnn_output_path}/Model_predictions/{filename}'
    path_to_ras = f'{output_path}.tif'

    # Import and load the model and weights, and send to correct device
    model = import_and_load_model_weights(model_structure, device, backbone, path_to_weights_file, filename = None)

    # Set the model to eval to avoid training it again.
    model.eval()

    # Get a list of all the tiles that will be run through the model
    list_tiles = os.listdir(f'{data_path}/CNN_input/Input_{vis1}_{im_size}')

    # Add paths to datasets so the model knows where to look
    train_dir_dim1 = f'{data_path}/CNN_input/Input_{vis1}_{im_size}'
    train_dir_dim2 = f'{data_path}/CNN_input/Input_{vis2}_{im_size}'
    train_dir_dim3 = f'{data_path}/CNN_input/Input_{vis3}_{im_size}'

    # Iterate through the tiles, run them through the model and predict the presence of objects
    predict_on_new_tiles(model_structure, model, filename, list_tiles, train_dir_dim1, train_dir_dim2, train_dir_dim3, output_path)

    # Merge the predicted tiles to create one raster
    merge_tiles(output_path, path_to_ras)

    # Vectorize the predicted raster and save as shapefile
    calculate_metrics.vectorize(path_to_ras)
    
    return path_to_ras

# THE END

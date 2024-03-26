#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Claudine Gravel-Miguel put this document together, but most of the functions were written by Katherine Peck and simplified by CGM
@date: 10 Jan 2024

@description: 
    This script contains functions to separate a raster into tiles that correspond to the squares of a provided shapefile grid

"""

# Reading and writing shapefiles, geodataframes
import geopandas as gpd

import rasterio
import rasterio.mask

import numpy as np

import math

import os

from skimage.io import imread

def format_grid(path_to_shp):
    """
    This imports and formats a shapefile grid so it can be used by later code

        Parameters:
            path_to_shp (string): Path to the hsapefile grid created in QGIS or Python
            
        Returns:
            grid (DataFrame): geopanda DataFrame with bottom left coordinates of each polygon

    """
    
    # Import the grid shapefile
    grid = gpd.read_file(path_to_shp)

    # Get the x and y anchors in named columns
    grid['x'] = grid['geometry'].bounds['minx']
    grid['y'] = grid['geometry'].bounds['miny']
    
    return(grid)
    

def scale_floats(array):
    """
    Inputs for the ML process need to be in floats between 0-1
    This takes a numpy array and returns a 0-1 scaled version

        Parameters:
            array (NumPy array): numpy array representing the raster image post-SLRM
            
        Returns:
            scaled (NumPy array): numpy array of floats with scaled values

    """
    
    scaled = (array-np.min(array))/(np.max(array)-np.min(array))
    return scaled


def tile_from_grid(bigraster, path_to_grid, outfolder, tile_size, is_mask, drop_nan):
    """
    Extract rasters by the mask of the grid and save as GeoTiffs

        Parameters:
            bigraster (str): filepath of raster to be tiled
            path_to_grid (str): filepath to shapefile grid used to define the tiles
            outfolder (str): filepath for output folder
            tile_size (int): desired tile height and width
            is_mask (boolean): If the raster is a mask of annotated objects (True) or not (False)
            drop_nan (boolean): If the user wants to drop any tile that has NA values or keep them (but replace NA with 0s)
        
        Returns:
            Nothing
    """
    
    # Check if the outfolder already exists, and create a new one if it does not.
    isExist = os.path.exists(outfolder)
    if isExist:
        print("The folder for the tiles already exists. New tiles will be added to it.")
        
    if not isExist:
        # Create a new folder because it does not exist
        os.makedirs(outfolder)
        print("Created a new folder for the tiles")
    
    # Import and format the grid to fit requirements of code below
    grid = format_grid(path_to_grid)
    
    # Create a counter to catch resolution errors
    counter = 0
    
    # Load the bigraster (only once)
    with rasterio.open(bigraster) as src:
        
        # Iterate through each square in the provided grid
        for index, row in grid.iterrows():
            out_image, out_transform = rasterio.mask.mask(src, [row['geometry']], crop = True)
            out_meta = src.meta
            
            # If the created tile is not square (was at the edge of the raster map), it is ignored
            if out_image.shape[1] != tile_size or out_image.shape[2] != tile_size:
                # Update the counter to print an error message later on.
                counter += 1
                #print(f"Tile_{str(math.floor(row['x']))}_{str(math.floor(row['y']))} goes beyond the raster and has shape {out_image.shape}. Therefore, it will be ignored.")
                pass
            
            else:
                # Height and width are the same dimensions as the desired tile size
                out_meta.update({"driver": "GTiff",
                             "height": tile_size,
                             "width": tile_size,
                             "transform": out_transform})
                
                # Ignores tiles with NAs if drop_nan is set to True
                if drop_nan == True and np.count_nonzero(out_image < -9000) > 0:
                    print(f"Dropping Tile{int(row['x'])}_{int(row['y'])} because it has NAs")
                    pass
                
                else:
                    # Filename should reference the lower left point of the tile
                    filename = outfolder + "/Tile" + str(math.floor(row['x'])) + "_" + str(math.floor(row['y'])) + ".tif"
                    
                    # -999999 or -9999 are the NoData value for these rasters - this transforms NAs into 0s before rescaling or cutting
                    out_image[out_image < -9000] = 0
                    
                    # Scale float values between 0 - 1 if the raster is NOT a mask
                    if is_mask == True:
                        pass
                    else:
                        out_image = scale_floats(out_image)
                                            
                    # Export the tile as a geotiff
                    with rasterio.open(filename, "w", **out_meta) as dest:
                        dest.write(out_image)
    
    # If all tiles were ignored, it means there is a problem in the parameter values provided or the resolution of the map.
    if counter == len(grid):
        raise Exception("The size of the tiles would be different from the value provided in tile_size, so they were all ignored. Check your large_raster's resolution, the tile_size value you provided, and how those fit with the size of the grid squares")

    return
        
def check_tile_size(outfolder, tile_size):
    """
    Compare the size of one created tile (in pixels) with the tile_size provided to make sure they are the same.
    This is a saveguard to prevent errors created when working with resolutions other then 1m/pixel.

        Parameters:
            outfolder (str): filepath of the folder where the tiles were saved
            tile_size (int): desired tile height and width
        
        Returns:
            Nothing, but it may print a warning statement if there is a discrepancy
    """
    
    # Create a list of the tiles' names
    filelist = os.listdir(outfolder)

    # Load the first tile (this is done only on one tile because they would already show a problem) and calculate its size
    m = imread(f'{outfolder}/{filelist[1]}')
    m_size = m.shape[1]
    
    # Print the size of the created tile
    print(f'The tiles created are {m_size}x{m_size}')
    
    # If the size of the tile is the same as a the provided tile_size, nothing happens because everything is OK.
    if m_size == tile_size:
        pass
    # If the size is different, it prints a warning statement
    else:
        print('WARNING: The size of the tiles you created differ from the tile_size entered above. Double-check your paths and values entered.')

# THE END

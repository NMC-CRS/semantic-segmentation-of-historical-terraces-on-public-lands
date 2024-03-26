#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 2023

@author: Katherine Peck

This script holds functions that vectorize the predicted presence pixels, 
do some post-processing cleaning by removing polygons that are smaller than a provided threshold,
and calculate the standard metrics (TP, TN, FP, FN, recall, precision, F1 score) by comparing the cleaned predictions to a provided shapefile of annotated features.

"""

# Import necessary modules

import geopandas as gpd

import rasterio

from shapely.geometry import shape

import numpy as np

def vectorize(rasterpath):
    """
    Given a binary raster, return a geodataframe of polygons for every pixel with a value of 1

        Parameters:
            rasterpath (str): binary raster filepath

        Returns:
            features_gdf (GeoDataFrame): geopandas gdf of polygons    
    """
    # Open raster 
    with rasterio.open(rasterpath) as src:
        data = np.float32(src.read())
        data[np.isnan(data)] = src.nodata
        transform = src.transform
        
    # Create feature mask 
    mask = data > 0
    
    # Get shapes and values based on the mask
    # Creates GeoJSON-like object
    shapeDict = rasterio.features.shapes(data, mask = mask, transform = transform)
    feats = []
    geoms = []
    
    # Append shapes to empty lists
    for key, value in shapeDict:
        feats.append(value)
        geoms.append(shape(key))
    crs = src.crs
    
    # Create new geodatarame from lists with original raster CRS
    features_gdf = gpd.GeoDataFrame({'feats': feats, 'geometry': geoms}, crs = crs)
    
    # Test exporting the result as a shapefile as well
    shp_filepath = rasterpath.replace(".tif", ".shp")
    
    # Add area
    features_for_shp = features_gdf
    features_for_shp["area"] = features_for_shp.area
    features_for_shp.to_file(shp_filepath, driver='ESRI Shapefile')
    
    # Return geodataframe
    return features_gdf

# Define the main helper function, which also calls vectorize() (defined above)
def compute_metrics(path_to_shp, path_to_ras, threshold):
    """
    Given a shapefile of actual features, a raster of predicted features,
    and an integer threshold for feature size, prints the recall, precision, and F1 values.

        Parameters:
            path_to_shp (str): filepath for shapefile of actual features
            path_to_ras (str): filepath for raster of detected features
            threshold (int): integer for max size of features to be removed from the predicted features
    """

    # Read in shapefile of predicted features
    actual = gpd.read_file(path_to_shp)
    pred_poly = vectorize(path_to_ras)

    # Filter out the smaller polygons using input area threshold
    pred_poly = pred_poly[pred_poly.area > threshold]

    # CALCULATE FN (false negative), TP (true positive), and FP (false positive)
    TP = 0
    FN = 0
    for actual_geom in actual.geometry:
        #If a shape in the predicted file intersects a shape in the actual file, it's a true positive
        #Otherwise, it's a true negative
        overlap = any(actual_geom.intersects(pred_geom) for pred_geom in pred_poly.geometry)
        if overlap:
            TP += 1
        else:
            FN += 1

    print("TP: " + str(TP))
    print("FN: " + str(FN))

    FP = 0
    
    # For all shapes in the gdf of predicted polygons
    for pred_geom in pred_poly.geometry:
        # if a polygon does not intersect a feature in the actual shapefile, it is a false positive
        overlap = any(pred_geom.intersects(actual_geom) for actual_geom in actual.geometry)
        if not overlap:
            FP += 1

    print("FP: " + str(FP))

    # CALCULATE AND PRINT METRICS
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F1 = (2 * recall * precision) / (recall + precision)

    print("Recall:", round(recall, 3))
    print("Precision:", round(precision, 3))
    print("F1:", round(F1, 3))

# Define an alternate helper function, which saves metrics to a list
def save_metrics(path_to_shp, path_to_ras, threshold):
    """
    Given a shapefile of actual features, a raster of predicted features,
    and an integer threshold for feature size, returns the recall, precision, and F1 values.

        Parameters:
            path_to_shp (str): filepath for shapefile of actual features
            path_to_ras (str): filepath for raster of detected features
            threshold (int): integer for max size of features to be removed from the predicted features

        Returns:
            metrics (list): list of metrics in the order [recall, precision, f1]
    """
    
    # Read in shapefile of predicted features
    actual = gpd.read_file(path_to_shp)
    print("Read the Shapefile")
    
    pred_poly = vectorize(path_to_ras)
    print("Vectorized the raster")

    # Filter out the smaller polygons
    # These features are in UTM, so area calculation should be in m^2 
    # at the moment, I don't think we need to use pint to convert, but it depends on the inputs from the previous step
    pred_poly = pred_poly[pred_poly.area > threshold]
    print("Filtered the polygons")

    # CALCULATE FN (false negative), TP (true positive), and FP (false positive)
    TP = 0
    FN = 0
    for actual_geom in actual.geometry:
        # If a shape in the predicted file intersects a shape in the actual file, it's a true positive
        # Otherwise, it's a true negative
        overlap = any(actual_geom.intersects(pred_geom) for pred_geom in pred_poly.geometry)
        if overlap:
            TP += 1
        else:
            FN += 1

    print(f"TP: {TP}, and FN: {FN}")
    
    FP = 0
    # For all shapes in the gdf of predicted polygons
    for pred_geom in pred_poly.geometry:
        #if a polygon does not intersect a feature in the actual shapefile, it is a false positive
        overlap = any(pred_geom.intersects(actual_geom) for actual_geom in actual.geometry)
        if not overlap:
            FP += 1

    print(f"FP: {FP}")
    
    # CALCULATE AND PRINT METRICS
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F1 = (2 * recall * precision) / (recall + precision)

    metrics = [recall, precision, F1]
    return metrics

# THE END

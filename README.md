# Semantic segmentation of historial terraces on public lands
## Description
This repository holds the code we used for our study of historical cotton terraces within the Piedmont National Wildlife Refuge, Georgia, USA

The scripts allows training a UNet that can take 3 different pre-trained backbones (VGG16, VGG19, and ResNet18)

## Installation
The scripts included here require Python 3.10 maximum. It also uses the following packages: `pandas`, `rasterio`, `shapely`, `geopandas`, `matplotlib`, `scikit-learn`, `scikit-image`, `tensorboard`, `torch_snippets`, `albumentations`, `torchmetrics`, `pycocotools`, and `tifffile`.

## File organization
These scripts require this specific file organization of input and output folders, as it uses hard-coded relative paths that follow this structure:
```bash
├── CNN_data
│   ├── CNN_input
│   │   ├── Grids
│   │   ├── Input_[vis name]_[tile size]
│   │   ├── [Object name mask subfolder]
│   │   │   ├── Target_[buffer size]_[tile_size]
│   ├── CNN_output
│   │   ├── Model_predictions
│   │   ├── Model_weights
│   ├── Visualizations_and_annotations
│   │   ├── Rasters
│   │   ├── Shapefiles
```

where:\
`[Object name mask subfolder]` can be anything that relates to the object to detect (e.g., *Terrace_masks*),\
`[vis name]` is the name of the visualization type, **without spaces** (e.g., *SLRM20m*),\
`[tile size]` is the height/width of the tiles (e.g., *256*), and\
`[buffer_size]` is the buffer around the annotated object, **without spaces** (e.g., *10m*)

## Workflow
The scripts included in the utils folder cover the 3 main steps of the workflow we used, which are described in more details below:
1. Create training tiles
2. Train a model
3. Apply a trained model to new data

### Create training tiles
To create our training dataset, we first created different visualization maps from our LiDAR-derived DTM. Those maps covered the whole area of the Piedmont National Wildlife Refuge (PNWR). We also annotated the presence of terraces in QGIS, and transformed those annotations into a raster map where 0 represents the background and 1 represents buffers around terraces.

We used the `create_overlapping_grid.py` script to create two overlapping grids using one of the visualization raster as a basemap to get the grids' extent. We created a 256x256 and a 512x512 grid. Both were saved in the **Grids** folder in **CNN_input**.

We then used the `tile_raster_from_grid.py` script to tile each of our visualization maps. The resulting tiles were separated into input folders with names that reflected the visualization. For example, 256x256 pixel tiles created from the SLRM map using 20m moving window were placed into a folder called **Input_SLRM20m_256** within the **CNN_input** folder, whereas the 512x512 tiles created from the slope map were saved into a folder called **Input_Slope_512**.

Similarly, we tiled the annotation mask using the same grids. The resulting tiles were placed in target folders with names that reflected their size and the buffer size around the objects. For example, the 256x256 tiles from the map with 20m buffers around terraces were saved in a folder called **Target_20m_256** within the **Terrace_masks** subfolder of **CNN_input**.

## Support
For support, contact one of the authors of this repository (see below) or open an issue.

## Authors and acknowledgment
**Authors:**

Claudine Gravel-Miguel, PhD\
Grant Snitker, PhD\
Katherine Peck, MA

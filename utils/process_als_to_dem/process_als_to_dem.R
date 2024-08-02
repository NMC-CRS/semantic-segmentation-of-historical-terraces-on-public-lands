
# Cultural Resource Sciences 
# New Mexico Consortium, Los Alamos, NM
# Author(s): Grant Snitker from workflow by Wade Ross
# Updated July, 2024

# Tasks: 
# 1. Generate als-derived Digital Terrain Models (DTMs) for archaeological feature detection inputs

# Note: make sure that R project and R script are organized in the following folder structure:
# ├── process_als_to_dem.rproj
# ├── process_als_to_dem.R
# ├── laz_data    #folder containing LAZ files to be processed into DEMs
# ├── laz_ground  #folder to hold output LAZ files from ground-classification if that step is used in the workflow
# ├── dem_output  #folder to hold output DEMs

rm(list = ls()) #clear workspace if needed

# 1. install packages and libs w/ pacman 
if (!require("pacman")) install.packages("pacman"); library(pacman)
p_load(lidR, here, tidyverse, sf, raster, sfheaders, magrittr, future, MBA, gstat, fields, remotes)

# 1.1 Set wd (specific to user ) # Only if not using the data in the project
# setwd('')

# 2. Check status of ALS tiles from vendor
ctg <- readLAScatalog("./laz_data")
las_check(ctg)

# 3. Set options for lidR LAZ file import
# opt_filter(ctg) <- opt_filter(ctg) <- "-keep_first"
opt_select(ctg) <- "xyzc"
opt_chunk_buffer(ctg) <- 40
plan(strategy = multisession, workers = 8L) # parallelization with future
#nbrOfWorkers() # evaluate number of cores that can be used 

plot(ctg) # examine the extent of the tiles

# 4. Classify ground points for tiles if needed
# Not needed for pre-classified tiles from vendor
# Be sure to delete any previous outputs in the LAZ_ground folder, or the code will give you an error that the file already exists
# opt_output_files(ctg) <- here("laz_ground/{ORIGINALFILENAME}_pmf") # add ground classification algorithm here to file name
# ctg.gnd = classify_ground(ctg, algorithm = pmf(ws = 5, th = 3)) #Progressive morphological filter
# ctg.gnd = classify_ground(ctg, mcc(1.5,0.3)) # Multiscale curvature classification
# ctg.gnd = classify_ground(ctg, algorithm = csf()) # Cloth simulation
 
# 5. run terrain function for tiles ------------------------
dem <- rasterize_terrain(ctg, res = 1, algorithm = tin()) # use 'ctg.gnd' as input if ground classification step is included

# 6. Finish up and write raster results 
# reset the futures plan to sequential processing 
future::plan("sequential")

# write raster outputs 
#dem output
writeRaster(dem, "./dem_output/dem_pmf_05m_tin.tif", filetype="GTiff", datatype="FLT4S", overwrite=TRUE) 




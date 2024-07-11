# Soil Moisture Data Gap Filling

Overview

This repository contains code for filling gaps in soil moisture data using various techniques.
The code is designed to handle missing values in soil moisture time series data and provide filled data points based on signal decomposition and residual prediction using Gaussian Process ML model.

## Requirements
- Countries vector boundary layer
- Crop Mask (500m spatial resolution) with values from 0-100 representing the percentage of pixel with crops of interest.
- Soil Moisture raster data at 25 km spatial resolution and dekad (10 days) temporal resolution coming from C3S: https://cds.climate.copernicus.eu/portfolio/dataset/satellite-soil-moisture

## Usage

Once the paths for input data are setup:
Define target like: 
  country_target = "Senegal"
and run: python preprocessing.py

## Output

Numpy arrays and statistics regarding pixels for crop mask areas for target country:

Folder **raster_tmp/pixels_sm_senegal/**

  - senegal_gap_filled_pixel_stack.npy
  - senegal_original_pixel_stack.npy
  - senegal_logs_gap_filling.csv
  - senegal_pixel_stat_table.csv

Folder **raster_tmp/clipped_sm_senegal/** with rasters gap filled at 500 m spatial resolution clipped to Target Country borders

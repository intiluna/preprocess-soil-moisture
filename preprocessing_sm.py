from pathlib import Path
import os
import rasterio
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from rasterio.plot import show
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from pyproj import CRS
import utiles as ut
import time
import subprocess
import numpy as np
#gap fill
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from statsmodels.tsa.seasonal import seasonal_decompose, STL
#stats
from datetime import datetime


print(os.getcwd())

print("Starting preprocessing...step 1")

ut.test_function("checking utiles")

country_target = "Senegal"
country_target_lower = country_target.lower()

# Paths
path_raster_sm = Path.cwd()/("../data/sm4maria")
path_raster_crop_mask = Path.cwd()/("../data/crop_mask")
path_vector_countries = Path.cwd()/("../data/gaul0_asap")

crop_mask_file = path_raster_crop_mask/("asap_mask_crop_v04.tif")

# Paths for temporal files
path_raster_temp = Path.cwd()/("raster_tmp")

skip_if_exist = True  # Set to True if you want to skip processing if file exist

# Check Paths
try:

    if not crop_mask_file.exists():
        raise FileNotFoundError(f"The crop mask file '{crop_mask_file}' does not exist.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)


# Soil Moisture Raster Files
sm_files = list(path_raster_sm.glob("*.tif"))
sm_files_sorted = sorted(sm_files, key=lambda x: int(x.stem.split('_')[1]))
sm_files_names = [file.stem for file in sm_files_sorted]
print(sm_files_names[:5])


# Read Crop Mask (v4)
crop_mask = rxr.open_rasterio(crop_mask_file)

# Read vector
countries = gpd.read_file(path_vector_countries/("gaul0_asap.shp"))
country_x = countries.loc[countries['adm0_name'] == country_target]
print(country_x.head())

# 1.0 Crop and clip crop_mask based on country_target

clipped_mask_path = path_raster_temp/(f"crop_mask_{country_target_lower}_clipped.tif")
crop_mask_cropped_path = path_raster_temp/(f"crop_mask_{country_target_lower}_cropped.tif")

if not skip_if_exist or not (path_raster_temp / f"crop_mask_{country_target_lower}_cropped.tif").exists():
    print("Start crop and clip...")
    crop_mask_cropped = crop_mask.rio.clip_box(*country_x.total_bounds)
    crop_mask_cropped.rio.to_raster(crop_mask_cropped_path)
    print("done cropped")
if not skip_if_exist or not (path_raster_temp / f"crop_mask_{country_target_lower}_clipped.tif").exists():
    crop_mask_clipped = crop_mask_cropped.rio.clip(country_x['geometry'])
    crop_mask_clipped.rio.to_raster(clipped_mask_path)
    print("Finish crop and clip...")

# binary crop mask
binary_mask_path = path_raster_temp/(f"crop_mask_{country_target_lower}_clipped_binary.tif")

if not skip_if_exist or not (binary_mask_path).exists():
    crop_mask_clipped_reclassified = (crop_mask_clipped > 0).astype('float32') # it was uint8
    crop_mask_clipped_reclassified.rio.to_raster(binary_mask_path)
    print("done binary transformation")
print("checkpoint after binary transformation...")

#2.0 Crop SM to crop mask binary extent
        
# create folder
cropped_sm_folder = (path_raster_temp / f"cropped_sm_{country_target_lower}")

if cropped_sm_folder.exists() and cropped_sm_folder.is_dir():
    print("cropped sm folder does exists and we skip processing")
else:
    cropped_sm_folder.mkdir(parents=True, exist_ok=True)
    print(f"The folder '{cropped_sm_folder}' has been created.")

    mask_binary = rxr.open_rasterio(binary_mask_path)
    print(f"mask_binary dimensions: {mask_binary.sizes}")

    # Require sm to be cropped (not clipped otherwise it wont cover the whole area)
    for file in sm_files_sorted:

        out_resample_crop = (cropped_sm_folder / f"cropped_{file.name}")
        file_sm = rxr.open_rasterio(file)
        print(file_sm.sizes)
        sm_crop = file_sm.rio.clip_box( minx= mask_binary.x.min().item(),
                                        miny= mask_binary.y.min().item(),
                                        maxx= mask_binary.x.max().item(),
                                        maxy= mask_binary.y.max().item()
                                       )
        sm_crop.rio.to_raster(out_resample_crop)
        print(f"done crop process for {file.name}")



# extract pixels for each soil moisture dekad

raster_path_list = list(cropped_sm_folder.glob("*.tif"))
pixels_sm_folder = (path_raster_temp / f"pixels_sm_{country_target_lower}")

# # sort raster path list
# ## Define a custom key function to extract the date from the filename
get_date_from_path = lambda path: int(path.stem[-8:])

# # Sort the list of Path objects based on the date
raster_path_list_sorted = sorted(raster_path_list, key=get_date_from_path)

stack_path = pixels_sm_folder/(f"{country_target_lower}_original_pixel_stack.npy")

if pixels_sm_folder.exists() and pixels_sm_folder.is_dir():
    print("pixels sm folder does  exists and we skip processing")
else:
    pixels_sm_folder.mkdir(parents=True, exist_ok=True)
    print(f"The folder '{pixels_sm_folder}' has been created.")

    
    stack = ut.extract_all_pixels(raster_path_list_sorted,stack_path) 
    
print("Done extracting pixels")    

# v4 gap fill at pixel level ---------------------------------------------------------

# gap fill at pixel level ----------------------------------------------------------

stack_filled_path = pixels_sm_folder/(f"{country_target_lower}_gap_filled_pixel_stack.npy")
if stack_filled_path.exists():
    print("Gap filled array exists and we skip processing")
else:
    print("Gap filling processing starting............")

    pixel_data = np.load(stack_path)
    pixel_data[pixel_data == -9999.0] = np.nan

    px = pixel_data.shape[1]
    py = pixel_data.shape[2]

    print(f"pixel_data shape:{pixel_data.shape}")

    tsf = np.full(pixel_data.shape, np.nan)
    logs_gap_filling=[]

    gp_process_start = time.time()


    for x in range(px):
        
        for y in range(py):
        #for y in range(3):#
            print(f"pixel_x:{x},pixel_y:{y}")
            
            time_serie = pixel_data[:, x, y]
            na_perc_start = ut.calculate_nan_percentage(time_serie)
            print(f"Start_Na%: {na_perc_start}")

            if na_perc_start > 75:
                print(f"Skipping decomposition for pixel:{(x,y)} with more than 75% raw missing values.")
                continue
            
            else:
                tidy_dataset = ut.get_data_v2(time_serie, start_date="1978-11-01", freq="10D", fulldate_start="1978-11-01", fulldate_end="2040-01-01", fillmethod="median")
                na_perc_end = ut.calculate_nan_percentage(tidy_dataset['y_hat_01'])
                print(f"End_Na%: {na_perc_end}")
                   
                # get decomposition

                ts_decomposition, decomposed_dataset = ut.decadal_decomposition_v2(tidy_dataset, period=365//10, seasonal=41,trend=61, improved="initial")
                print(f"Decomposition done for pixel:{(x,y)} done")
                        
                # GP estimates
                kv3 = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
                n_optimizer = 5

                gapfilled_dataset, kernel = ut.gapfilling_gp_v2(dataset=decomposed_dataset, n_restarts_optimizer=n_optimizer,kernel=kv3)
                filled_02 = gapfilled_dataset["y_hat_02"]
                
                                
                #replace values in tsf (time series filled)
                tsf[:, x, y] = filled_02.values.reshape(-1)

                logs_gap_filling.append({'x': x, 'y': y, 'na_perc_start': round(na_perc_start, 2), 'na_perc_end': round(na_perc_end, 2)})
                print(f"Gapfilled done for pixel:{(x,y)} done")
            
    # save no-gaps pixels array
    stack_filled_path = pixels_sm_folder/(f"{country_target_lower}_gap_filled_pixel_stack.npy")
    pixel_data[pixel_data == np.nan] = -9999.0
    np.save(stack_filled_path, tsf)
            
    # logs_gap_filling as a CSV file
    df_logs_gap_filling = pd.DataFrame(logs_gap_filling)
    df_logs_gap_filling.to_csv(pixels_sm_folder/(f"{country_target_lower}_logs_gap_filling.csv"), index=False)


    gp_process_end = time.time()
    total_time = gp_process_end - gp_process_start

    print(f"Total GP gap fill process took:{total_time}")




# Create raster tif files using gap filled pixel stack----------------------------------------------------------

# Paths
gap_filled_stack_path = pixels_sm_folder/(f"{country_target_lower}_gap_filled_pixel_stack.npy")
    
gap_filled_sm_folder = path_raster_temp/(f"gp_croppped_sm_{country_target_lower}")
if gap_filled_sm_folder.exists() and gap_filled_sm_folder.is_dir():
    print("gap filled raster folder exists and we skip processing")
else:
    gap_filled_sm_folder.mkdir(parents=True, exist_ok=True)
    print(f"Gap filled raster folder {gap_filled_sm_folder} has been created")

    # Load the gap filled pixel stack
    pixel_data = np.load(gap_filled_stack_path)

    start_process = time.time()
        
    # Open each original raster file and replace the array with the corresponding slice from pixel_data
    for i, original_raster_path in enumerate(raster_path_list_sorted, start=0):
        # Load the replacement data for the current raster
        replacement_data = pixel_data[i, :, :]  # Indexing starts from 0 in arrays, but from 1 in raster names

        # Open the original raster file for reading
        with rasterio.open(original_raster_path) as src:
            
            # Create the output raster file
            filename = original_raster_path.name
            #print(filename)
            output_path = gap_filled_sm_folder/(f'gp_{filename}')

            # Write the replacement data to the new raster file
            with rasterio.open(output_path, 'w', **src.profile) as dst:
                dst.write(replacement_data, 1)  # Writing the replacement data to the first band
    end_process = time.time()
    total_time = end_process - start_process

    print(f"Total write gap fill tif process took:{total_time}") 


# Resample to 500 m ----------------------------------------------------------
resample_sm_folder = (path_raster_temp / f"resample_gp_sm_{country_target_lower}")

gp_raster_path_list = list(gap_filled_sm_folder.glob("*.tif"))
gp_raster_path_list_sorted = sorted(gp_raster_path_list, key=get_date_from_path)

if resample_sm_folder.exists() and resample_sm_folder.is_dir():
    print("resample sm folder does  exists")
else:
    resample_sm_folder.mkdir(parents=True, exist_ok=True)
    print(f"The folder '{resample_sm_folder}' has been created.")

    start_resample_process = time.time()

    # Require sm to be cropped (not clipped otherwise it wont cover the whole area)
    for file in gp_raster_path_list_sorted:
        out_resample = (resample_sm_folder / f"resampled_{file.name}")
        ut.align_and_resample_raster(file, binary_mask_path, out_resample)
        print(f"done resample for {file.name}")

    end_resample_process = time.time()
    total_time = end_resample_process - start_resample_process
    print(f"Total resample process took:{total_time}")

# Clip resample gap filled SM raster files ----------------------------------------------------------
    
clipped_sm_folder=(path_raster_temp / f"clipped_sm_{country_target_lower}")   
sm_files_resampled = list(resample_sm_folder.glob("*.tif"))

if clipped_sm_folder.exists() and clipped_sm_folder.is_dir():
    print("resample-crop sm folder does  exists and we skip processing")
else:
    clipped_sm_folder.mkdir(parents=True, exist_ok=True)
    print(f"The folder '{clipped_sm_folder}' has been created.")

    mask_binary = rxr.open_rasterio(binary_mask_path)
    print(f"mask_binary dimensions: {mask_binary.sizes}")

    start_clip_process = time.time()

    
    for file in sm_files_resampled:
        print(f"Processing:{file.name}")
        out_resample_crop = (clipped_sm_folder / f"cropped_{file.name}")
        file_sm = rxr.open_rasterio(file)
        #print(file_sm.sizes)
        sm_crop = file_sm.rio.clip_box( minx= mask_binary.x.min().item(),
                                        miny= mask_binary.y.min().item(),
                                        maxx= mask_binary.x.max().item(),
                                        maxy= mask_binary.y.max().item()
                                       )
        sm_crop.rio.to_raster(out_resample_crop)
        print(f"done resample for {file.name}")

    end_clip_process = time.time()
    total_time = end_clip_process - start_clip_process
    print(f"Total clip process took:{total_time}")


# Create table Stats -----------------------------------------------------------------------------

#output path
pixel_stat_table_path = pixels_sm_folder/(f"{country_target_lower}_pixel_stat_table.csv")
#input
clipped_raster_path_list = list(clipped_sm_folder.glob("*.tif"))
clipped_raster_path_list_sorted = sorted(clipped_raster_path_list, key=get_date_from_path)

if pixel_stat_table_path.exists():
    print("Pixel stats does  exists and we skip processing")
else:
    print("Pixel stats starting")
    
    start_pixel_stat_process = time.time()

    #input-mask
    #clipped_mask_path
    crop_mask = rxr.open_rasterio(clipped_mask_path)

    # get mask
    mask = crop_mask[0, :, :] > 0

    # Get the indices where the mask is True
    y_indices, x_indices = np.where(mask)

    # Get coordinates of pixels masked
    latitudes = crop_mask.y[y_indices]
    longitudes = crop_mask.x[x_indices]

    # Transform xarray to numpy array
    crop_mask_arr = crop_mask.values

    # Mask mask and get values
    crop_mask_values = crop_mask_arr[0, mask]

    # Initialize an empty list to store DataFrames
    dfs = []

    # Loop through the raster files
    for i,raster_file in enumerate(clipped_raster_path_list_sorted):
        # Assuming raster_file.stem[-8:] is a string in the format YYYYMMDD
        date_str = raster_file.stem[-8:]
        date = datetime.strptime(date_str, '%Y%m%d').date()
        print(date)

        # Mask sm and get values
        sm = rxr.open_rasterio(raster_file)
        sm_arr = sm.values
        sm_values = sm_arr[0, mask]

        # Stack the arrays column-wise
        stacked_data = np.column_stack((x_indices, y_indices, latitudes.values, longitudes.values, sm_values, crop_mask_values))

        # Convert the stacked data to a DataFrame
        df = pd.DataFrame(stacked_data, columns=['x_index', 'y_index', 'Latitude', 'Longitude', 'SM_Value', 'Mask_Value'])

        # Add the "Country" column
        df.insert(0, 'Country', country_target)

        # Add the "Date" column
        df.insert(1, 'Date', date)

        # Replace -9999.0 values with NaN
        df.replace(-9999.0, np.nan, inplace=True)

        # Append the current DataFrame to the list of DataFrames
        summary_df = ut.calculate_summary(df)
        
        dfs.append(summary_df)
        print(f"Extracted pixel data for file:{i}")
    
    # Concatenate all DataFrames in the list along the rows
    df_combined = pd.concat(dfs, ignore_index=True)

    # reorder columns
    new_order = ['Country', 'Date', 'ave_sm', 'weighted_ave_sm', 'na_percent', 'weighted_na_percent','weighted_available_percent']
    df_combined = df_combined.reindex(columns=new_order)

    df_combined['Date'] = pd.to_datetime(df_combined['Date'])

    # Display the combined DataFrame
    print(df_combined.shape)
    print(df_combined.head())
    print(df_combined.tail())

    # Save the combined DataFrame as a CSV file  
    df_combined.to_csv(pixel_stat_table_path, index=False)

    end_pixel_stat_process = time.time()
    total_time = end_pixel_stat_process - start_pixel_stat_process
    print(f"Total pixel stat process took:{total_time}")

# Cleaning --------------------------------------------------------------------------------------------
# Leave clipped sm_folder and pixel sm folder


# Example usage
#delete_folders_list = [cropped_sm_folder, resample_sm_folder, gap_filled_sm_folder]
#ut.delete_folders(delete_folders_list)    
    


# Wait for 60 seconds
#time.sleep(60)

# Shut down the computer
#subprocess.run(['sudo', 'shutdown', '-h', 'now'])



    






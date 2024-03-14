from pathlib import Path
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
from statsmodels.tsa.seasonal import seasonal_decompose


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
if not skip_if_exist or not (path_raster_temp / f"crop_mask_{country_target}_cropped.tif").exists():
    print("Start crop and clip...")
    crop_mask_cropped = crop_mask.rio.clip_box(*country_x.total_bounds)
    crop_mask_cropped.rio.to_raster(path_raster_temp/(f"crop_mask_{country_target}_cropped.tif"))
    print("done cropped")
if not skip_if_exist or not (path_raster_temp / f"crop_mask_{country_target}_clipped.tif").exists():
    crop_mask_clipped = crop_mask_cropped.rio.clip(country_x['geometry'])
    crop_mask_clipped.rio.to_raster(path_raster_temp/(f"crop_mask_{country_target}_clipped.tif"))
    print("Finish crop and clip...")

# binary crop mask
binary_mask_path = path_raster_temp/(f"crop_mask_{country_target}_clipped_binary.tif")

if not skip_if_exist or not (path_raster_temp / f"crop_mask_{country_target}_clipped_binary.tif").exists():
    crop_mask_clipped_reclassified = (crop_mask_clipped > 0).astype('uint8')
    crop_mask_clipped_reclassified.rio.to_raster(binary_mask_path)
    print("done binary transformation")
print("checkpoint after binary transformation...")

# 2.0 pre process: SM -Crop to CM-binary
## create folfer if not exist
# cropped_sm_folder = (path_raster_temp / f"cropped_sm_{country_target_lower}")

# if cropped_sm_folder.exists() and cropped_sm_folder.is_dir():
#     print("cropped sm folder does exists and we skip processing")
# else:
#     cropped_sm_folder.mkdir(parents=True, exist_ok=True)
#     print(f"The folder '{cropped_sm_folder}' has been created.")

#     # Require sm to be cropped (not clipped otherwise it wont cover the whole area
#     for file in sm_files_sorted:
    
#         #out_resample = (resample_sm_folder / f"{country_target}_{file.name}")
#         start_time = time.strftime("%Y-%m-%d %H:%M:%S")
#         print(f"starting crop for SM: {file.name} at {start_time}")

        
#         sm_file = rxr.open_rasterio(file)
#         sm_cropped = sm_file.rio.clip_box(*country_x.total_bounds)
#         #sm_clipped = sm_cropped.rio.clip(country_x['geometry'])
#         sm_cropped.rio.to_raster(cropped_sm_folder/(f"{country_target_lower}_cropped_{file.name}"))
            
        
#         print(f"done crop for {file.name}")


# # resample sm to match resolution of crop mask
# sm_files_cropped = list(cropped_sm_folder.glob("*.tif"))
# #print(sm_files_clipped[:5])
# crop_mask_cropped_path = path_raster_temp/(f"crop_mask_{country_target}_cropped.tif")

# resample_sm_folder = (path_raster_temp / f"resample_sm_{country_target_lower}")

# if resample_sm_folder.exists() and resample_sm_folder.is_dir():
#     print("resample sm folder does  exists")
# else:
#     resample_sm_folder.mkdir(parents=True, exist_ok=True)
#     print(f"The folder '{resample_sm_folder}' has been created.")

#     # Require sm to be cropped (not clipped otherwise it wont cover the whole area)
#     for file in sm_files_cropped:
#         out_resample = (resample_sm_folder / f"resampled_{file.name}")
#         start_time = time.strftime("%Y-%m-%d %H:%M:%S")
#         print(f"starting resample for SM: {file.name} at {start_time}")
#         print(f"input file:{file}")
#         print(f"output file:{out_resample}")
#         ut.align_and_resample_raster(file, crop_mask_cropped_path, out_resample,target_resolution=(0.004464285715000,0.004464285715000))

#         #ut.reproj_match(infile = file, match= crop_mask,outfile = out_resample)    
#         print(f"done resample for {file.name}")


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

    # Require sm to be cropped (not clipped otherwise it wont cover the whole area
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
        print(f"done resample for {file.name}")



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
    
    

# # gap fill at pixel level ----------------------------------------------------------

stack_filled_path = pixels_sm_folder/(f"{country_target_lower}_gap_filled_pixel_stack.npy")
if stack_filled_path.exists():
    print("gap filled array exists and we skip processing")
else:
    print("gap filled array starting")

    
    pixel_data = np.load(stack_path)
    pixel_data[pixel_data == -9999.0] = np.nan

    px = pixel_data.shape[1]
    py = pixel_data.shape[2]

    #tsf = pixel_data[:].copy()
    tsf=np.ones(pixel_data.shape)
    pixel_no_gap_filled=[]

    gp_process_start = time.time()


    for x in range(px):
        
        for y in range(py):
        #for y in range(3):#
            print(f"pixel_x:{x},pixel_y:{y}")
            
            time_serie = pixel_data[:, x, y]
            tidy_dataset = ut.get_data(time_serie)

            #print(tidy_dataset.head())
            
            if tidy_dataset['y_hat_01'].isnull().any():
                print(f"Skipping decomposition for pixel:{(x,y)} with missing values.")
                pixel_no_gap_filled.append((x,y))
                continue  # Skip to the next iteration
            # get decomposition
            ts_decomposition, decomposed_dataset = ut.decadal_decomposition(tidy_dataset, period=365//10)
        
            # GP estimates
            kernel1 = RBF(1.0, (1e-2, 1e2)) # seed the kernel
            #kernel1 = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) # seed the kernel    
            #kernel1 = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-1) + ExpSineSquared(1.0, 5.0, (1e-2, 1e2))
            
            gapfilled_dataset, kernel = ut.gapfilling_gp(
                                                    dataset=decomposed_dataset,
                                                    n_restarts_optimizer=0,
                                                    kernel=kernel1
                                                    )
        
            #original = gapfilled_dataset["y"]
            #filled_01 = gapfilled_dataset["y_hat_01"]
            filled_02 = gapfilled_dataset["y_hat_02"] 
            #flag = gapfilled_dataset["flag"]
            
            #replace values in tsf (time series filled)
            #tsf[:, x, y] = filled_02.values.reshape(-1, 1)
            tsf[:, x, y] = filled_02.values.reshape(-1)



    # save no-gaps pixels array
    stack_filled_path = pixels_sm_folder/(f"{country_target_lower}_gap_filled_pixel_stack.npy")
    pixel_data[pixel_data == 1] = -9999.0
    np.save(stack_filled_path, tsf)

        # Save pixel_no_gap_filled as a CSV file
    df_pixel_no_gap_filled = pd.DataFrame({'PixelNoGapFilled': pixel_no_gap_filled})
    #df_pixel_no_gap_filled.to_csv('pixel_no_gap_filled.csv', index=False)
    df_pixel_no_gap_filled.to_csv(pixels_sm_folder/(f"{country_target_lower}_pixel_gapfill.csv"), index=False)


    gp_process_end = time.time()
    total_time = gp_process_end - gp_process_start

    print(f"Total GP gap fill process took:{total_time}")

# Wait for 60 seconds
#time.sleep(60)

# Shut down the computer
#subprocess.run(['sudo', 'shutdown', '-h', 'now'])



    






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

# Crop and clip crop_mask based on country_target
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

# pre process for resample soil moisture
## create folfer if not exist
#resample_sm_folder = (path_raster_temp / f"resample_sm_{country_target_lower}")
cropped_sm_folder = (path_raster_temp / f"cropped_sm_{country_target_lower}")



if cropped_sm_folder.exists() and cropped_sm_folder.is_dir():
    print("cropped sm folder does exists and we skip processing")
else:
    cropped_sm_folder.mkdir(parents=True, exist_ok=True)
    print(f"The folder '{cropped_sm_folder}' has been created.")

    # Require sm to be cropped (not clipped otherwise it wont cover the whole area
    for file in sm_files_sorted:
    
        #out_resample = (resample_sm_folder / f"{country_target}_{file.name}")
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"starting crop for SM: {file.name} at {start_time}")

        
        sm_file = rxr.open_rasterio(file)
        sm_cropped = sm_file.rio.clip_box(*country_x.total_bounds)
        #sm_clipped = sm_cropped.rio.clip(country_x['geometry'])
        sm_cropped.rio.to_raster(cropped_sm_folder/(f"{country_target_lower}_cropped_{file.name}"))
            
        
        print(f"done crop for {file.name}")

## add condition so resample process is not run if folder exists

# resample sm to match resolution of crop mask
sm_files_cropped = list(cropped_sm_folder.glob("*.tif"))
#print(sm_files_clipped[:5])
crop_mask_cropped_path = path_raster_temp/(f"crop_mask_{country_target}_cropped.tif")

resample_sm_folder = (path_raster_temp / f"resample_sm_{country_target_lower}")

if resample_sm_folder.exists() and resample_sm_folder.is_dir():
    print("resample sm folder does  exists")
else:
    resample_sm_folder.mkdir(parents=True, exist_ok=True)
    print(f"The folder '{resample_sm_folder}' has been created.")

    # Require sm to be cropped (not clipped otherwise it wont cover the whole area)
    for file in sm_files_cropped:
        out_resample = (resample_sm_folder / f"resampled_{file.name}")
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"starting resample for SM: {file.name} at {start_time}")
        print(f"input file:{file}")
        print(f"output file:{out_resample}")
        ut.align_and_resample_raster(file, crop_mask_cropped_path, out_resample,target_resolution=(0.004464285715000,0.004464285715000))

        #ut.reproj_match(infile = file, match= crop_mask,outfile = out_resample)    
        print(f"done resample for {file.name}")


# tune sm to crop mask binary extent
        
sm_files_resampled = list(resample_sm_folder.glob("*.tif"))

# create folder
resample_crop_sm_folder = (path_raster_temp / f"resample_crop_sm_{country_target_lower}")

if resample_crop_sm_folder.exists() and resample_crop_sm_folder.is_dir():
    print("resample-crop sm folder does  exists")
else:
    resample_crop_sm_folder.mkdir(parents=True, exist_ok=True)
    print(f"The folder '{resample_crop_sm_folder}' has been created.")

    mask_binary = rxr.open_rasterio(binary_mask_path)
    print(f"mask_binary dimensions: {mask_binary.sizes}")

    # get bounding box of crop mask clipped

    #cm_minx=mask_binary.x.min().item()
    #cm_miny=mask_binary.y.min().item()
    #cm_maxx=mask_binary.x.max().item()
    #cm_maxy=mask_binary.x.max().item()
    
    for file in sm_files_resampled:
        print(file)
        out_resample_crop = (resample_crop_sm_folder / f"cropped_{file.name}")
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

raster_path_list = list(resample_sm_folder.glob("*.tif"))
pixels_sm_folder = (path_raster_temp / f"pixels_sm_{country_target_lower}")

if pixels_sm_folder.exists() and pixels_sm_folder.is_dir():
    print("pixels sm folder does  exists")
else:
    pixels_sm_folder.mkdir(parents=True, exist_ok=True)
    print(f"The folder '{pixels_sm_folder}' has been created.")

    stack_path = pixels_sm_folder/(f"{country_target_lower}_original_pixel_stack.npy")
    stack = ut.extract_pixels_using_mask(binary_mask_path,raster_path_list,stack_path) # error because of different dimensions
    # probably works if I clip the crop mask binary layer to match the SM resample raster
    # IndexError: boolean index did not match indexed array along dimension 0; dimension is 1008 but corresponding boolean dimension is 982
    # check time series
    time_series = stack[:,:1].copy()
    time_series[time_series == -9.9990000e+03] = np.nan
    print(time_series[0:3])

# gap fill at pixel level


# replace NA with gap filled data


# save no-gaps raster files


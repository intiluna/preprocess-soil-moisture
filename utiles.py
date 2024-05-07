from rasterio.warp import reproject, Resampling, calculate_default_transform
import rasterio
from osgeo import gdal
import numpy as np
import time
import pandas as pd
import shutil
#gap fill
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from statsmodels.tsa.seasonal import seasonal_decompose, STL


def test_function(a):
    print(a)
 
def reproj_match(infile, match, outfile):
    """Reproject a file to match the shape and projection of existing raster. 
    
    Parameters
    ----------
    infile : (string) path to input file to reproject
    match : (string) path to raster with desired shape and projection 
    outfile : (string) path to output file tif
    """
    # open input
    with rasterio.open(infile) as src:
        src_transform = src.transform
        
        # open input to match
        with rasterio.open(match) as match:
            dst_crs = match.crs
            
            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,     # input CRS
                dst_crs,     # output CRS
                match.width,   # input width
                match.height,  # input height 
                *match.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
            )

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": 0})
        print("Coregistered to shape:", dst_height,dst_width,'\n Affine',dst_transform)
        # open output
        with rasterio.open(outfile, "w", **dst_kwargs) as dst:
            # iterate through bands and write using reproject function
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
                


def align_and_resample_raster(input_raster_path, reference_raster_path, output_path, resampling_method=gdal.GRA_NearestNeighbour):
    """
    Aligns and resamples an input raster to match the spatial characteristics of a reference raster.

    Parameters:
    - input_raster_path (str): Path to the input raster that needs alignment and resampling.
    - reference_raster_path (str): Path to the reference raster used for alignment and resampling.
    - output_path (str): Path to save the aligned and resampled raster.
    - resampling_method (int): Resampling method (GDAL constant). Default is gdal.GRA_NearestNeighbour but other methods are available like gdal.GRA_Bilinear.

    Returns:
    - None
    
    Example usage:
    input_raster_path = "input_raster.tif"
    reference_raster_path = "reference_raster.tif"
    output_path = "output_raster.tif"
    
    align_and_resample_raster(input_raster_path, reference_raster_path, output_path)
    
    """

    # Open the reference raster to get its spatial information
    reference_ds = gdal.Open(reference_raster_path)
    target_crs = reference_ds.GetProjectionRef()
    target_resolution = (reference_ds.GetGeoTransform()[1], reference_ds.GetGeoTransform()[5])
    print("Reference raster resolution (xRes, yRes):", target_resolution)

    # Perform the alignment and resampling
    gdal.Warp(output_path, input_raster_path, format="GTiff", dstSRS=target_crs,
              xRes=target_resolution[0], yRes=target_resolution[1],
              resampleAlg=resampling_method, options=['COMPRESS=DEFLATE'])

    # Close the datasets
    reference_ds = None

    
def extract_pixels_using_mask(binary_mask_path,raster_path_list,stack_path):
    with rasterio.open(binary_mask_path) as src:
        mask = src.read(1)

    initial_process_time = time.time()

    stack = None
    for tiff_path in raster_path_list:
        
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"starting extraction from for SM: {tiff_path} at {start_time}")
        
        with rasterio.open(tiff_path) as src:
            array = src.read(1)
            extract = array[mask==1]  # [3 4]
            if stack is None:
                stack = extract.copy()
            else:
                stack = np.vstack((stack, extract))
        print ("done file extraction")
    
    end_process_time = time.time()
    total_time = end_process_time - initial_process_time
    print(f"complete extraction lasted: {total_time}")
    np.save(stack_path, stack)

    return stack

def extract_all_pixels(raster_path_list,stack_path):
    initial_process_time = time.time()

    stack = None
    for i, tiff_path in enumerate(raster_path_list):
        
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"starting extraction from for SM: {tiff_path} at {start_time}")
        
        with rasterio.open(tiff_path) as src:
            array = src.read(1)
            #extract = array[mask==1]  # [3 4]
            if stack is None:
                stack = np.empty((len(raster_path_list), array.shape[0], array.shape[1]))
            stack[i, :, :] = array
        print ("done file extraction")
    
    end_process_time = time.time()
    total_time = end_process_time - initial_process_time
    print(f"complete extraction lasted: {total_time}")
    np.save(stack_path, stack)

    return stack

# functions for gap filling---------
# Apply gap fill as a function
def get_data(time_serie: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame()
    df["y"]=time_serie.ravel()
    X = np.arange(0, df.shape[0]).reshape(-1, 1)
    y = df["y"].values

    # Fill missing values with NaN
    #y[y == -9999] = np.nan

    # Simple linear interpolation to fill missing values
    df = pd.DataFrame({'X': X.ravel(), 'y': y, 'flag': np.isnan(y)})
    y_hat_01 = pd.Series(y).interpolate(method='linear').values
    df['y_hat_01'] = df['y'].interpolate(method='linear')

    return df

def get_data_v2(time_series: np.ndarray, start_date: str, freq: str, fulldate_start: str, fulldate_end: str, fillmethod: str) -> pd.DataFrame:
    # 1. Define the date range for scaling
    full_dates = pd.Series(pd.date_range(start=fulldate_start, end=fulldate_end))
    
    # 2. Define the time series based on start date, frequency, and length
    date_time_series = pd.date_range(start=start_date, freq=freq, periods=len(time_series))
    
    date_min = full_dates.min()
    date_max = full_dates.max()
    
    # Transform dates to numbers and scale them
    X_numeric = (date_time_series - date_min).days
    date_numeric_max = (date_max - date_min).days
    X_scaled = X_numeric / date_numeric_max

    X = np.arange(0, len(time_series))
    y = time_series.ravel()
    
    # Create DataFrame
    df = pd.DataFrame({'X': X.ravel(), 'y': y, 'flag': np.isnan(y)})
    df['X_date'] = date_time_series
    df['X_scaled'] = X_scaled
    
    # Manage missing values according to the fillmethod
    if fillmethod == "interpolate":
        df['y_hat_01'] = df['y'].interpolate(method='linear')
    elif fillmethod == "median":
        y_median = np.nanmedian(y)
        df['y_hat_01'] = df['y'].fillna(y_median)
    else:
        raise ValueError("fillmethod is not valid.Use 'interpolate' o 'median'.")
    
    return df

def decadal_decomposition(dataset: pd.DataFrame, period: int=365//10) -> pd.DataFrame:
    ts_decomposition = seasonal_decompose(
        dataset['y_hat_01'],
        model='additive',
        period=period,
        extrapolate_trend='freq'
    )

    # Time series decomposition    
    ts_residual = dataset['y_hat_01'] - ts_decomposition.seasonal - ts_decomposition.trend
    dataset["residual"] = ts_residual
    dataset["trend"] = ts_decomposition.trend
    dataset["seasonal"] = ts_decomposition.seasonal

    return ts_decomposition, dataset

def decadal_decomposition_v2(dataset: pd.DataFrame, period: int=365//10, seasonal=31, trend=51, improved="all") -> pd.DataFrame:
    # Decomposition using STL
    stl_result = STL(dataset['y_hat_01'], period=period, seasonal=seasonal, trend=trend).fit()

    # Add the components to the dataset
    dataset["residual"] = stl_result.resid
    dataset["trend"] = stl_result.trend
    dataset["seasonal"] = stl_result.seasonal
    dataset["seasonal_improved"] = stl_result.seasonal.copy()  

    # Calculate seasonal average
    seasonal_avg = stl_result.seasonal.groupby(stl_result.seasonal.index % period).mean()

    # Determine first NAN (flag=False)
    first_false_index = dataset.index[~dataset['flag']].min() if (~dataset['flag']).any() else len(dataset)

    if improved == "all":
        # Replace all gaps with the seasonal average
        nan_indices = dataset.index[dataset['flag']]
    elif improved == "initial":
        # Replace only the initial gaps with the seasonal average
        nan_indices = dataset.index[dataset['flag'] & (dataset.index < first_false_index)]
    else:
        raise ValueError("Argument 'improved' must be 'all' or 'initial'")

    for idx in nan_indices:
        # Calculate the seasonal index corresponding to the seasonal cycle
        seasonal_index = idx % period
        # Assign the average seasonal value to the missing value
        dataset.at[idx, 'seasonal_improved'] = seasonal_avg[seasonal_index]

    return stl_result, dataset

def gapfilling_gp(
    dataset: pd.DataFrame,
    n_restarts_optimizer: int=0,
    kernel: RBF = C(1.0, (1e-3, 1e3))
) -> pd.DataFrame:
    # Get the missing values
    missing_indices = np.where(dataset["flag"])[0]
    
    # Obtain the residuals without the missing values
    X = np.delete(dataset["X"].values, missing_indices).reshape(-1, 1)
    y = np.delete(dataset["residual"].values, missing_indices)
    ynorm = (y - np.mean(y)) / np.std(y) # to make the GP more stable


    # Fit the GP
    gp = GaussianProcessRegressor(
        kernel=kernel,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=42
    )
    gp.fit(X, ynorm)

    # Predict missing values
    y_pred, sigma = gp.predict(
        dataset["X"][missing_indices].values.reshape(-1, 1),
        return_std=True
    )

    # Transform the predicted values back to the original scale
    y_pred = y_pred * np.std(y) + np.mean(y)
    sigma = sigma * np.std(y)

    # Transform the residuals to real values
    y_pred = (
        y_pred + 
        dataset["seasonal"].values[missing_indices] +
        dataset["trend"].values[missing_indices]
    )
    
    # Copy the original dataset and fill the missing values
    dataset["y_hat_02"] = dataset["y_hat_01"].copy()
    
    # Cast y_pred to the expected dtype
    y_pred_casted = y_pred.astype(np.float32)
    
    dataset.loc[missing_indices, "y_hat_02"] = y_pred_casted
    

    
    dataset["sigma"] = 0.0000
    #dataset["sigma"][missing_indices] = sigma
    dataset.loc[missing_indices, "sigma"] = sigma

    return dataset, gp.kernel_
# end of gap filling----------------

# -----------------------------------
def gapfilling_gp_v2(
    dataset: pd.DataFrame,
    n_restarts_optimizer: int=5,
    kernel: RBF = C(1.0, (1e-3, 1e3))
) -> pd.DataFrame:
    # Get the missing values
    missing_indices = np.where(dataset["flag"])[0] # flag==True
    
    # Obtain the residuals without the missing values
    X = np.delete(dataset["X"].values, missing_indices).reshape(-1, 1)
    y = np.delete(dataset["residual"].values, missing_indices)
    ynorm = (y - np.mean(y)) / np.std(y) # to make the GP more stable
    

    # Fit the GP
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=n_restarts_optimizer,optimizer="fmin_l_bfgs_b", random_state=42)
    
    gp.fit(X, ynorm)

    # Predict missing values
    X_pred = dataset["X_scaled"][missing_indices].values.reshape(-1, 1) # X donde estan gaps
    y_pred, sigma = gp.predict(X_pred,return_std=True)

    # Transform the predicted values back to the original scale
    y_pred = y_pred * np.std(y) + np.mean(y)
    sigma = sigma * np.std(y)

    # Store predicted residuals
    dataset["residual_predicted"] = dataset["residual"].copy()
    dataset.loc[missing_indices, "residual_predicted"] = y_pred

    # Transform the residuals to real values
    y_pred = (
        y_pred + 
        dataset["seasonal_improved"].values[missing_indices] +
        dataset["trend"].values[missing_indices]
    )
    
    # Copy the original dataset and fill the missing values
    dataset["y_hat_02"] = dataset["y_hat_01"].copy()
    
    # Cast y_pred to the expected dtype
    y_pred_casted = y_pred.astype(np.float32)
    
    dataset.loc[missing_indices, "y_hat_02"] = y_pred_casted
    
    dataset["sigma"] = 0.0000
    #dataset["sigma"][missing_indices] = sigma
    dataset.loc[missing_indices, "sigma"] = sigma

    return dataset, gp.kernel_
# end of gap filling----------------

def calculate_nan_percentage(arr):
    nan_percentage = (np.isnan(arr).sum() / len(arr)) * 100
    return nan_percentage


# functions for calculate stats table---------

# Function to calculate summary statistics
def calculate_summary(df):
    # Calculate average SM_Value
    ave_sm = df['SM_Value'].mean()

    # Calculate percentage of NaN values in SM_Value
    na_percent = df['SM_Value'].isna().mean() * 100

    # Calculate weighted average SM_Value
    weighted_ave_sm = (df['SM_Value'] * df['Mask_Value']).sum() / df['Mask_Value'].sum()

    # Calculate weighted percentage of NaN values in SM_Value
    weighted_na_percent = (df['SM_Value'].isna() * df['Mask_Value']).sum() / df['Mask_Value'].sum() * 100

    # Get available weighted sm data
    weighted_available_percent = 100 - weighted_na_percent

    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'Country': [df['Country'].iloc[0]],
        'Date': [df['Date'].iloc[0]],
        'ave_sm': [ave_sm],
        'na_percent': [na_percent],
        'weighted_ave_sm': [weighted_ave_sm],
        'weighted_na_percent': [weighted_na_percent],
        'weighted_available_percent': [weighted_available_percent]
        
    })

    return summary_df

def delete_folders(paths):
    for path in paths:
        # Check if the path exists
        if path.exists():
            # Use shutil.rmtree() to delete the folder and its contents recursively
            shutil.rmtree(path)
            print(f"Folder '{path}' and its contents have been deleted successfully.")
        else:
            print(f"The folder '{path}' does not exist.")
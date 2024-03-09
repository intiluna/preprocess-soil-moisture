from rasterio.warp import reproject, Resampling, calculate_default_transform
import rasterio

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
                
from osgeo import gdal

def align_and_resample_raster(input_raster_path, reference_raster_path, output_path, target_resolution=(0.01, 0.01), resampling_method=gdal.GRA_Bilinear):
    """
    Aligns and resamples an input raster to match the spatial characteristics of a reference raster.

    Parameters:
    - input_raster_path (str): Path to the input raster that needs alignment and resampling.
    - reference_raster_path (str): Path to the reference raster used for alignment and resampling.
    - output_path (str): Path to save the aligned and resampled raster.
    - target_resolution (tuple): Target spatial resolution in the form (xRes, yRes). Default is (0.01, 0.01).
    - resampling_method (int): Resampling method (GDAL constant). Default is gdal.GRA_Bilinear.

    Returns:
    - None
    """

    # Open the reference raster to get its spatial information
    reference_ds = gdal.Open(reference_raster_path)
    target_crs = reference_ds.GetProjectionRef()

    # Perform the alignment and resampling
    gdal.Warp(output_path, input_raster_path, format="GTiff", dstSRS=target_crs,
              xRes=target_resolution[0], yRes=target_resolution[1],
              resampleAlg=resampling_method, options=['COMPRESS=DEFLATE'])

    # Close the datasets
    reference_ds = None
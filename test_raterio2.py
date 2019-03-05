import pprint
from collections import OrderedDict
import rasterio
import fiona
from rasterio import features

image_in = r'C:\Users\dpilon\Desktop\DeepLearning\sherbrooke_zone1.tif'
image_out = r'C:\Users\dpilon\Desktop\DeepLearning\sherbrooke_out.tif'
burned = r'C:\Users\dpilon\Desktop\DeepLearning\sherbrooke_burned.tif'
shape_in = r'C:\Users\dpilon\Desktop\DeepLearning\batiment_test1.shp'

with rasterio.open(image_in) as src:
    band1 = src.read(1)
    band1[:,:] = 127
    with rasterio.open(image_out, 'w',
                       driver='GTiff',
                       dtype=rasterio.uint8,
                       count=1,
                       width=src.width,
                       height=src.height,
                       nodata=0,
                       transform=src.transform,
                       crs=src.crs  ) as dest:
        dest.write(band1,indexes=1)


with fiona.open (shape_in, 'r', driver='ESRI Shapefile') as c:

    with rasterio.open(image_out, 'r') as src:
        image = features.rasterize(
            ( (a['geometry'],255) for a in c),
            out_shape=src.shape,
            transform=src.transform)

        with rasterio.open(
                burned, 'w',
                driver = 'GTiff',
                dtype = rasterio.uint8,
                count = 1,
                width = src.width,
                height = src.height,
                transform=src.transform,
                crs=src.crs) as dst:
            dst.write(image, indexes=1)








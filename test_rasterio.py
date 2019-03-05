import pprint
from collections import OrderedDict
import rasterio
import fiona
from rasterio import features, open

schema = landmarks = { 'geometry': 'Polygon',
                              'properties': OrderedDict([('classe', 'int')] )
                            }

feature2 =  { 'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[(70.0, 46.0),
                                     (70.0, 50.0),
                                     (74.0, 50.0),
                                     (74.0, 46.0),
                                     (70.0, 46.0)]]
                          },
              'properties': OrderedDict([  ('classe', '1') ])
             }

feature1 =  { 'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[(20.0, 6.0),
                                     (20.0, 10.0),
                                     (24.0, 10.0),
                                     (24.0, 6.0),
                                     (20.0, 6.0)]]
                          },
              'properties': OrderedDict([  ('classe', '1') ])
             }

with fiona.open('test_logo.shp', 'w', driver='ESRI Shapefile',
                schema=landmarks) as c:
    c.write(feature1)
    c.write(feature2)
c.close

with fiona.open ('test_logo.shp', 'r', driver='ESRI Shapefile') as c:

    with rasterio.open('13547682814_f2e459f7a5_o.png') as src:

        image = features.rasterize(
            ( (a['geometry'],255) for a in c),
            out_shape=src.shape,
            transform=src.transform)

        with rasterio.open(
                'rasterized-results.tif', 'w',
                driver = 'GTiff',
                dtype = rasterio.uint8,
                count = 1,
                width = src.width,
                height = src.height,
                transform=src.transform,
                crs=src.crs) as dst:
            dst.write(image, indexes=1)






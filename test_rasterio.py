import pprint
from collections import OrderedDict
import rasterio
import fiona
from rasterio import features, open

schema = landmarks = { 'geometry': 'Polygon',
                              'properties': OrderedDict([('classe', 'int')] )
                            }

feature1 =  { 'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[(70.0, 6.0),
                                     (70.0, 10.0),
                                     (74.0, 10.0),
                                     (74.0, 6.0),
                                     (70.0, 6.0)]]
                          },
              'properties': OrderedDict([  ('classe', '1') ])
             }

with fiona.open('test_logo.shp', 'w', driver='ESRI Shapefile',
                schema=landmarks) as c:
    c.write(feature1)

with fiona.open ('test_logo.shp', 'r', driver='ESRI Shapefile') as c:
    for feature in c:
        feat = feature['geometry']

with rasterio.open('13547682814_f2e459f7a5_o.png') as src:
    blue = src.read(3)

    mask = blue != 255
    shapes = features.shapes(blue, mask=mask)



#for feat in shapes:
#    print (feat[1])
#pprint.pprint(next(shapes))


    image = features.rasterize(
        ( (a,255) for a in (feat,)),
        out_shape=src.shape,
        transform=src.transform)

#    image = features.rasterize(
#        ((g, 255) for g, v in shapes),
#        out_shape=src.shape,
#        transform=src.transform)

    with rasterio.open(
            'rasterized-results.tif', 'w',
            driver = 'GTiff',
            dtype = rasterio.uint8,
            count = 1,
            width = src.width,
            height = src.height) as dst:
        dst.write(image, indexes=1)






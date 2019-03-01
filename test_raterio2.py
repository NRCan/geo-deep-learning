import pprint
import rasterio
from rasterio import features

with rasterio.open('13547682814_f2e459f7a5_o.png') as src:
    blue = src.read(3)

    mask = blue != 255
    shapes = features.shapes(blue, mask=mask)

    a = ((g, 255) for g, v in shapes)
    for b in a:
        print (b)
    image = features.rasterize(
                ((g, 255) for g, v in shapes),
                 out_shape=src.shape)


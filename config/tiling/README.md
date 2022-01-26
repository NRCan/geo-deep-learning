# Tiling

The tiling process consists of cutting up the imagery and ground truth to tiles of a certain size. This prepares the 
dataset for training.

## Tile size

Size of an individual tile. For example, a raster of 1024 x 1024 pixels will output 4 tiles if tile_size is 512. The 
value for this parameter should remain relatively stable as varying tile sizes has little impact of the performance of 
model. Tiling is mostly aimed at making is possible to fill a batch with at least 4 tile pairs of different AOIs without
busting a machine's memory while training. Defaults to 512.

## Resampling factor

A resampling of the imagery can be done to artificially modify the resolution (aka ground sampling distance). Typically,
upsampling (or upscaling) is used for building footprint segmentation since a higher resolution can help provide 
[a finer segmentation](https://github.com/SpaceNetChallenge/SpaceNet7_Multi-Temporal_Solutions/blob/master/1-lxastro0/code/README.md?plain=1#L25), 
possibly even separating very close buildings. 

For information about the "resampling factor" parameter, see [rasterio's resampling documentation]
(https://rasterio.readthedocs.io/en/latest/topics/resampling.html)

## Minimum annotated percent

Discards tile pairs (imagery & ground truth) if the non-background area (e.g. area covered with classes of interest) 
on a given ground truth tile is lower than this minimum. Defaults to 0 (keep all tiles). This parameter is a data 
balancing tool for undersampling. It is easy to implement and use, but may not be the perfect solution for all data
balancing problems. For more information on pros and cons of undersampling, oversampling and other class
balancing strategies, see [*Buda & Al., 2018*](https://www.sciencedirect.com/science/article/pii/S0893608018302107?casa_token=1gtjUgWc6pUAAAAA:SUDHxtgD8SPDrsM4wR93mH6ZYW57Mr-BYX2nBwxTuT8DsUlWJcvpAV1vgdACQgY78IbiZuCrPgb_) 
and [*Longadge & Dongre, 2013*](https://arxiv.org/pdf/1305.1707).

## Minimum raster tile size

Discards tile pairs if the raster tile is smaller than this minimum size (in bytes). In effect, this is used to discard
tiles pairs for which the imagery tile contains only no data. Defaults to 0.
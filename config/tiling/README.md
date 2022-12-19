# Tiling

The tiling process consists of cutting up the imagery and ground truth to patches of a certain size. This prepares the 
dataset for training.

## Tiling data directory

This directory defines where output patches will be written

## Train val percent

This parameter defines the proportion of patches to be redirected to the validation set.
`{'trn':0.7, 'val':0.3}` means around 30% of patches will belong to validation.

## Patch size

Size of an individual patch. For example, a raster of 1024 x 1024 pixels will output 4 patchs if patch_size is 512. The 
value for this parameter should remain relatively stable as varying patch sizes has little impact of the performance of 
model. Tiling is mostly aimed at making it possible to fill a batch with at least 4 patch pairs of different AOIs without
busting a machine's memory while training. Defaults to 512.

## Minimum annotated percent

Discards patch pairs (imagery & ground truth) if the non-background area (e.g. area covered with classes of interest) 
on a given ground truth patch is lower than this minimum. Defaults to 0 (keep all patchs). This parameter is a data 
balancing tool for undersampling. It is easy to implement and use, but may not be the perfect solution for all data
balancing problems. For more information on pros and cons of undersampling, oversampling and other class
balancing strategies, see [*Buda & Al., 2018*](https://www.sciencedirect.com/science/article/pii/S0893608018302107?casa_token=1gtjUgWc6pUAAAAA:SUDHxtgD8SPDrsM4wR93mH6ZYW57Mr-BYX2nBwxTuT8DsUlWJcvpAV1vgdACQgY78IbiZuCrPgb_) 
and [*Longadge & Dongre, 2013*](https://arxiv.org/pdf/1305.1707).

## Continuous values

If True, the tiling script will ensure all pixels values in the rasterized ground truth have continuous values starting 
at 1 (0 being background). 

In most cases, this parameter has no impact as values may already be continuous. However, it becomes useful to set 
`continuous_values == True` when filtering polygons from a ground truth file using an attribute field and 
attribute values (see config/dataset/README.md). For example, filtering values [2,4] from a given attribute field will 
create ground truth rasterized patches with these same discontinuous values, unless `continuous_values == True`. 

If you choose to set `continuous_values == False`, errors may occur in metrics calculation, training and outputted 
values at inference. We strongly recommend keeping the default `True` value.

## Save preview labels

If True, a .png copy of rasterized ground truth patches will be written for quick visualization. A colormap is used to 
map actual values in .geotiff ground truth patch (usually very close to 0, thus hard to visualize with a system viewer),
However, the conversion from .geotiff to .png discard the georeferencing information. If one wishes to locate a 
particular patch, it is recommended to open the .geotiff version of ground truth patch in a GIS software like QGIS.  

## Multiprocessing

If True, the tiling script uses Python's multiprocessing capability to process each AOI in parallel. This greatly 
accelerates the tiling process. For testing or debugging purposes or for small dataset, we'd recommend keeping the 
default `multiprocessing == False`.

## Write destination raster

If True, the destination raster (built as [VRT](../../dataset/README.md#under-the-hood)) will be written in the AOI's 
root directory (see AOI's class docstrings).

Defaults to `write_dest_raster=False`<sup>1</sup> <sup>2</sup>

1. When bands requested don't require a VRT to be created, no destination raster is written even if 
`write_dest_raster=True` since the destination raster would be identical to the source raster.
2. if a VRT is required, but `write_dest_raster=False`, no destination raster is written to disk.

> This feature is currently implemented mostly for debugging and demoing purposes. 

## Write mode

Defines behavior in case patches already exist in destination folder for a particular dataset.

Modes:
"raise_exists" (default): tiling will raise error if patches already exist. 
"append": tiling will skip AOIs for which all patches already exist 

> This feature applies to 1st step of tiling only, does not apply to 2nd 
step (filtering, sorting among trn/val and burning vector ground truth patches).
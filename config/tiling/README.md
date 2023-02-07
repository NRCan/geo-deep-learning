# Tiling

The tiling process consists of cutting up the imagery and ground truth to patches of a certain size. This prepares the 
dataset for training.

## Tiling data directory (str)

This directory defines where output patches will be written

## Train val percent (dict with str keys and float values)

This parameter defines the proportion of patches to be redirected to the validation set.
`{'trn':0.7, 'val':0.3}` means around 30% of patches will belong to validation.

## Patch size (int)

Size of an individual patch. For example, a raster of 1024 x 1024 pixels will output 4 patchs if patch_size is 512. The 
value for this parameter should remain relatively stable as varying patch sizes has little impact of the performance of 
model. Tiling is mostly aimed at making it possible to fill a batch with at least 4 patch pairs of different AOIs without
busting a machine's memory while training. Defaults to 512.

## Minimum annotated percent (int)

Discards patch pairs (imagery & ground truth) if the non-background area (e.g. area covered with classes of interest) 
on a given ground truth patch is lower than this minimum. Defaults to 0 (keep all patchs). This parameter is a data 
balancing tool for undersampling. It is easy to implement and use, but may not be the perfect solution for all data
balancing problems. For more information on pros and cons of undersampling, oversampling and other class
balancing strategies, see [*Buda & Al., 2018*](https://www.sciencedirect.com/science/article/pii/S0893608018302107?casa_token=1gtjUgWc6pUAAAAA:SUDHxtgD8SPDrsM4wR93mH6ZYW57Mr-BYX2nBwxTuT8DsUlWJcvpAV1vgdACQgY78IbiZuCrPgb_) 
and [*Longadge & Dongre, 2013*](https://arxiv.org/pdf/1305.1707).

## Continuous values (bool)

If True, the tiling script will ensure all pixels values in the rasterized ground truth have continuous values starting 
at 1 (0 being background). 

In most cases, this parameter has no impact as values may already be continuous. However, it becomes useful to set 
`continuous_values == True` when filtering polygons from a ground truth file using an attribute field and 
attribute values (see config/dataset/README.md). For example, filtering values [2,4] from a given attribute field will 
create ground truth rasterized patches with these same discontinuous values, unless `continuous_values == True`. 

If you choose to set `continuous_values == False`, errors may occur in metrics calculation, training and outputted 
values at inference. We strongly recommend keeping the default `True` value.

## Save preview labels (bool)

If True, a .png copy of rasterized ground truth patches will be written for quick visualization. A colormap is used to 
map actual values in .geotiff ground truth patch (usually very close to 0, thus hard to visualize with a system viewer),
However, the conversion from .geotiff to .png discard the georeferencing information. If one wishes to locate a 
particular patch, it is recommended to open the .geotiff version of ground truth patch in a GIS software like QGIS.  

## Multiprocessing (bool)

If True, the tiling script uses Python's multiprocessing capability to process each AOI in parallel. This greatly 
accelerates the tiling process. For testing or debugging purposes or for small dataset, we'd recommend keeping the 
default `multiprocessing == False`.

## CLAHE clip limit (int)

Our teams empirical tests have shown that, in most satellite imagery with right skewed histogram 
(ex.: most of Worldview imagery), histogram equalization with the [CLAHE algorithm](https://srv2.freepaper.me/n/OH5z7hgkxfyC4zO_hufz5Q/PDF/03/0347192c1b9db2b2f55a5d329a5e4a53.pdf)
improves the performance of models and subsequent quality of extractions. 

After having compared [Kornia's](https://kornia.readthedocs.io/en/v0.5.0/enhance.html?highlight=clahe#kornia.enhance.equalize_clahe)
and [Scikit-image's](https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_adapthist) 
implementation [on 3 RGB images of varying sizes](https://github.com/NRCan/geo-deep-learning/issues/359), the 
geo-deep-learning team has favored Kornia's CLAHE.

> Kornia expects "clip_limit" as a float with default value at 40. Because sk-image's implementation expects this 
> parameter to be between 0 and 1, geo-deep-learning forces user to input an integer as clip_limi. This is meant to 
> reduce the potential confusion with sk-image's expected value.

## Write destination raster (bool)

If True, the destination raster (built as [VRT](../../dataset/README.md#under-the-hood)) will be written in the AOI's 
root directory (see AOI's class docstrings).

Defaults to `write_dest_raster=False`<sup>1</sup> <sup>2</sup>

1. When bands requested don't require a VRT to be created, no destination raster is written even if 
`write_dest_raster=True` since the destination raster would be identical to the source raster.
2. if a VRT is required, but `write_dest_raster=False`, no destination raster is written to disk.

> This feature is currently implemented mostly for debugging and demoing purposes. 

## Write mode (str)

Defines behavior in case patches already exist in destination folder for a particular dataset.

Modes:
"raise_exists" (default): tiling will raise error if patches already exist. 
"append": tiling will skip AOIs for which all patches already exist 

> This feature applies to 1st step of tiling only, does not apply to 2nd 
step (filtering, sorting among trn/val and burning vector ground truth patches).
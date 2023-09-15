# Input data

> Also see [dataset's configuration documentation](../config/dataset/README.md) for specific information about input data 
parameters.

The sampling and inference steps requires a csv referencing input data. An example of input csv can be found in 
[tests](../tests/tiling/tiling_segmentation_binary_ci.csv). Each row of this csv is considered, in geo-deep-learning 
terms, to be an [AOI](https://torchgeo.readthedocs.io/en/latest/user/glossary.html#term-area-of-interest-AOI).

| raster path               | vector ground truth path | dataset split | aoi id (optional) |
|---------------------------|--------------------------|---------------|-------------------|
| my_dir/my_geoimagery1.tif | my_dir/my_geogt1.gpkg    | trn           | Ontario-1         |
| my_dir/my_geoimagery2.tif | my_dir/my_geogt2.gpkg    | tst           | NewBrunswick-23   |
| ...                       | ...                      | ...           | ...               |

> If left blank, the aoi id will be derived from the raster path and bands requested.

The path to a custom csv must be entered in the 
[dataset configuration](../config/dataset/test_ci_segmentation_binary.yaml). See the 
[configuration documentation](../config/README.md) for more information.
Also check the [suggested folder structure](https://github.com/NRCan/geo-deep-learning#folder-structure).

## Dataset splits
Split in csv should be either "trn", "tst" or "inference". The validation split is automatically created during tiling. 
Its proportion is set by the [tiling config](../config/tiling/README.md#Train-val-percent)

## Raster and vector file compatibility
Rasters to be used must be in a format compatible with 
[rasterio](https://rasterio.readthedocs.io/en/latest/quickstart.html?highlight=supported%20raster%20format#opening-a-dataset-in-reading-mode)/
[GDAL](https://gdal.org/drivers/raster/index.html) (ex.: GeoTiff). Similarly, labels (aka annotations) for each image 
must be stored as polygons in a [Geopandas compatible vector file](Rasters to be used must be in a format compatible 
with [rasterio](https://rasterio.readthedocs.io/en/latest/quickstart.html?highlight=supported%20raster%20format#opening-a-dataset-in-reading-mode)/[GDAL](https://gdal.org/drivers/raster/index.html) 
(ex.: GeoTiff). Similarly, labels (aka annotations) for each image must be stored as polygons in a 
[Geopandas compatible vector file](https://geopandas.org/en/stable/docs/user_guide/io.html#reading-spatial-data) 
(ex.: GeoPackage).

## Single-band vs multi-band imagery

Remote sensing is known to deal with raster files from a wide variety of formats and flavors. To provide as much 
flexibility as possible with variable input formats for raster data, geo-deep-learning supports:
1. Multi-band raster files, to be used as is (all bands needed, all bands is expected order)
2. Multi-band raster files with more bands than needed (e.g. Actual is "BGRN", needed is "BGR")
3. Multi-band raster files with bands in different order than needed (e.g. Actual is "BGR", needed is "RGB")
4. Single-band raster files, identified with a common string pattern (see details below)
5. Single-band raster files, identified as assets in a stac item (see details below)

To support these variable inputs, geo-deep-learning expects the first column of an input csv to be in the 
following formats:

### Use case #1: Multi-band raster files, used as is

| raster path                     | ... |
|---------------------------------|-----|
| my_dir/my_multiband_geofile.tif | ... |

### Use cases #2 and #3: Multi-band raster files with more bands or different order than needed 

Same as above, but the expected order or subset of bands must be set in the 
["bands" parameter of dataset config](../config/dataset/README.md#bands). For example:

```
dataset:
    [...]
    bands: [3, 2, 1]
```

Here, if the original multi-band raster had BGR bands, geo-deep-learning will reorder these bands to RGB order. 
Also, band selection and reordering can be performed simultaneously. In this example, if the source raster contained 
BGRN bands, the "N" band would be removed and the resulting raster would still be as RGB.

> [Following the GDAL convention, bands are indexed from 
> 1.](https://rasterio.readthedocs.io/en/latest/quickstart.html#reading-raster-data)

> It is the user's responsibility to know which bands correspond to each index. When dealing with multi-band source 
> imagery, Geo-Deep-Learning doesn't "know" which bands are present in the file and in which order. 

The `bands` parameter is set in the [dataset config](../config/dataset/test_ci_segmentation_multiclass.yaml).

### Use case #4: Single-band raster files, identified with a common string pattern

A single line in the csv can refer to multiple single-band rasters, by using a string pattern.
The "band specific" string in the file name must be in a 
[hydra-like interpolation format](https://hydra.cc/docs/1.0/advanced/override_grammar/basic/#primitives), with `${...}` 
notation. Geo-deep-learning will locate a list of single-band rasters during execution using two informations:
1. the ["bands" parameter of dataset config](../config/dataset/README.md#bands)
2. the input csv's path, if it contains the `${dataset.bands}` string.

For example:

1. Bands in [dataset config](../config/dataset/test_ci_segmentation_binary.yaml):

```
dataset:
    [...]
    bands: [R, G, B]
```

2. [Input csv](../tests/tiling/tiling_segmentation_binary_ci.csv):

| raster path                                                | ... |
|------------------------------------------------------------|-----|
| my_dir/my_singleband_geofile_band_**${dataset.bands}**.tif | ... |

During execution, this would result in using, **in the same order as bands appear in dataset config**, the following 
files:
`my_dir/my_singleband_geofile_band_R.tif`
`my_dir/my_singleband_geofile_band_G.tif`
`my_dir/my_singleband_geofile_band_B.tif`

### Use case #5: Single-band raster files, identified as assets in a stac item

> Only Stac Items referencing **single-band assets** are supported currently. See 
> [our Worldview-2 example](https://datacube-stage.services.geo.ca/api/collections/spacenet-samples/items/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03)

> Also, the [STAC spec](https://github.com/radiantearth/stac-spec/blob/master/item-spec/item-spec.md) is young and 
> quickly evolving. There exists multiple formats for stac item. Only a very specific format of stac item is supported 
> by geo-deep-learning. If using stac items with geo-deep-learning, make sure they follow the structure if our 
> Worldview-2 example above. 

Bands must be selected by [common name](https://github.com/stac-extensions/eo/#common-band-names) in dataset config:

```
dataset:
    [...]
    bands: ["red", "green", "blue"]
```

> Order matters: `["red", "green", "blue"]` is not equal to `["blue", "green", "red"]` !

### Under the hood

For use cases 2 to 5, geo-deep-learning creates a [virtual raster](https://gdal.org/drivers/raster/vrt.html) to bring 
the variable source raster data to a common destination format without rewriting existing data to disk.

# Input data
The sampling and inference steps requires a csv referencing input data. An example of input csv can be found in [tests](tests/sampling/sampling_segmentation_binary_ci.csv).
Each row of this csv is considered, in geo-deep-learning terms, to be an [AOI](https://torchgeo.readthedocs.io/en/latest/user/glossary.html#term-area-of-interest-AOI).

| raster path               | vector ground truth path | dataset split | aoi id (optional) |
|---------------------------|--------------------------|---------------|-------------------|
| my_dir/my_geoimagery1.tif | my_dir/my_geogt1.gpkg    | trn           | Ontario-1         |
| my_dir/my_geoimagery2.tif | my_dir/my_geogt2.gpkg    | tst           | NewBrunswick-23   |
| ...                       | ...                      | ...           | ...               |

> The use of aoi id information will be implemented in a near future. It will serve, for example, to print a detailed report of tiling, training and evaluation, or for easier debugging.

The path to a custom csv must be entered in the [dataset configuration](https://github.com/NRCan/geo-deep-learning/blob/develop/config/dataset/test_ci_segmentation_binary.yaml#L9). See the [configuration documentation](config/README.md) for more information.
Also check the [suggested folder structure](https://github.com/NRCan/geo-deep-learning#folder-structure).

## Dataset splits
Split in csv should be either "trn", "tst" or "inference". The validation split is automatically created during tiling. It's proportion is set by the [dataset config](https://github.com/NRCan/geo-deep-learning/blob/develop/config/dataset/test_ci_segmentation_binary.yaml#L8). 

## Raster and vector file compatibility
Rasters to be used must be in a format compatible with [rasterio](https://rasterio.readthedocs.io/en/latest/quickstart.html?highlight=supported%20raster%20format#opening-a-dataset-in-reading-mode)/[GDAL](https://gdal.org/drivers/raster/index.html) (ex.: GeoTiff). Similarly, labels (aka annotations) for each image must be stored as polygons in a [Geopandas compatible vector file](Rasters to be used must be in a format compatible with [rasterio](https://rasterio.readthedocs.io/en/latest/quickstart.html?highlight=supported%20raster%20format#opening-a-dataset-in-reading-mode)/[GDAL](https://gdal.org/drivers/raster/index.html) (ex.: GeoTiff). Similarly, labels (aka annotations) for each image must be stored as polygons in a [Geopandas compatible vector file](https://geopandas.org/en/stable/docs/user_guide/io.html#reading-spatial-data) (ex.: GeoPackage).
) (ex.: GeoPackage).

## Single-band vs multi-band imagery

To support both single-band and multi-band imagery, the path in the first column of an input csv can be in **one of three formats**:

### 1. Path to a multi-band image file:
`my_dir/my_multiband_geofile.tif`

A particular order or subset of bands in multi-band file must be used by setting a list of specific indices:

#### Example:

`bands: [3, 2, 1]`

Here, if the original multi-band raster had BGR bands, geo-deep-learning will reorder these bands to RGB order. 

> [Following the GDAL convention, bands are indexed from 
> 1.](https://rasterio.readthedocs.io/en/latest/quickstart.html#reading-raster-data)

> It is the user's responsibility to know which bands correspond to each index. When dealing with multi-band source 
> imagery, Geo-Deep-Learning doesn't "know" which bands are present in the file and in which order. 

The `bands` parameter is set in the [dataset config](../config/dataset/test_ci_segmentation_multiclass.yaml).

### 2. Path to single-band image files, using only a common string
A path to a list of single-band rasters can be inserted in the csv, but only a the string common to all single-band files should be considered.
The "band specific" string in the file name must be in a [hydra-like interpolation format](https://hydra.cc/docs/1.0/advanced/override_grammar/basic/#primitives), with `${...}` notation. The interpolation string completed during execution by a dataset parameter with a list of desired band identifiers to help resolve the single-band filenames.

#### Example:

In [dataset config](../config/dataset/test_ci_segmentation_binary.yaml):

`bands: [R, G, B]`

In [input csv](../tests/tiling/tiling_segmentation_binary_ci.csv):

| raster path                                                | ground truth path | dataset split |
|------------------------------------------------------------|-------------------|---------------|
| my_dir/my_singleband_geofile_band_**${dataset.bands}**.tif | gt.gpkg           | trn           |

During execution, this would result in using, **in the same order as bands appear in dataset config**, the following files:
`my_dir/my_singleband_geofile_band_R.tif`
`my_dir/my_singleband_geofile_band_G.tif`
`my_dir/my_singleband_geofile_band_B.tif`

> To simplify the use of both single-band and multi-band rasters through a unique input pipeline, single-band files are artificially merged as a [virtual raster](https://gdal.org/drivers/raster/vrt.html).

### 3. Path to a Stac Item 
> Only Stac Items referencing **single-band assets** are supported currently. See [our Worldview-2 example](https://datacube-stage.services.geo.ca/api/collections/spacenet-samples/items/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03).

Bands must be selected by [common name](https://github.com/stac-extensions/eo/#common-band-names) in dataset config:
`bands: ["red", "green", "blue"]`

> Order matters: `["red", "green", "blue"]` is not equal to `["blue", "green", "red"]` !
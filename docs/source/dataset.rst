.. _dataset:

Dataset
+++++++

The dataset configuration defines the data (images, ground truth) and their parameters. The documentation
on the parameters used is explained in the :ref:`yaml <yamlparameters>` section.

The tiling and inference steps requires a csv referencing input data. An example of input csv for
massachusetts buildings dataset can be found in 
`tests <https://github.com/NRCan/geo-deep-learning/blob/develop/tests/tiling/tiling_segmentation_binary_ci.csv>`_. 
Each row of this csv is considered, in geo-deep-learning terms, to be an 
`AOI <https://torchgeo.readthedocs.io/en/latest/user/glossary.html#term-area-of-interest-AOI>`_.

.. list-table:: **Example CSV**
   :header-rows: 1

   * - raster path 
     - vector ground truth path
     - dataset split
     - aoi id (optional)
   * - my_dir/my_geoimagery1.tif
     - my_dir/my_geogt1.gpkg
     - trn
     - Ontario-1
   * - my_dir/my_geoimagery2.tif
     - my_dir/my_geogt2.gpkg
     - tst
     - NewBrunswick-23
   * - ...
     - ...
     - ...
     - ...

.. note::
    
     If left blank, the aoi id will be derived from the raster path and bands requested.

The path to a custom csv must be entered in the 
`dataset configuration <https://github.com/NRCan/geo-deep-learning/blob/develop/config/dataset/test_ci_segmentation_binary.yaml>`_.
See the :ref:`yaml <yamlparameters>` section for more information.

Dataset splits
--------------

Split in csv should be either "trn", "tst" or "inference". The tiling script outputs lists of 
patches for "trn", "val" and "tst" and these lists are used as is during training. 
Its proportion is set by the :ref:`tiling config <datatiling>`.  

AOI
---
An AOI is defined as an image (single imagery scene or mosaic), its content and metadata and the associated ground truth vector (optional).  

.. note::
    
     AOI without ground truth vector can only be used for inference purposes.


The AOI's implementation in the code is as follow:  

.. autoclass:: dataset.aoi.AOI
   :members:
   :special-members:

Raster and vector file compatibility
------------------------------------

Rasters to be used must be in a format compatible with 
`rasterio <https://rasterio.readthedocs.io/en/latest/quickstart.html?highlight=supported%20raster%20format#opening-a-dataset-in-reading-mode>`_ 
and `GDAL <https://gdal.org/drivers/raster/index.html>`_ (ex.: GeoTiff). Same for the labels 
(aka annotations) for each image must be stored as polygons in a `Geopandas <https://geopandas.org/en/stable/docs/user_guide/io.html>`_ 
compatible vector file.

.. _datasetsinglemultiband:

Single-band vs multi-band imagery
---------------------------------

Remote sensing is known to deal with raster files from a wide variety of formats and flavors. 
To provide as much 
flexibility as possible with variable input formats for raster data, geo-deep-learning supports:

#. :ref:`Multi-band raster files, used as is <datasetmultiband>` (all bands needed, all bands is in the expected order)
#. :ref:`Multi-band raster files with more bands or different order than needed <datasetmultibandmorebands>` (e.g. Actual is "BGRN", needed is "BGR" OR Actual is "BGR", needed is "RGB")
#. :ref:`Single-band raster files, identified with a common string pattern <datasetsingleband>` (see details below)
#. :ref:`Single-band raster files, identified as assets in a stac item <datasetstacitem>` (see details below)

To support these variable inputs, geo-deep-learning expects the first column of an input csv to be in the 
following formats.

.. _datasetmultiband:
Multi-band raster files, used as is
====================================

This is the default and basic use. 

.. list-table:: 
   :header-rows: 1

   * - raster path 
     - ...
   * - my_dir/my_multiband_geofile.tif
     - ...

.. _datasetmultibandmorebands:
Multi-band raster files with more bands or different order than needed 
======================================================================

Same as above, but the expected order or subset of bands must be set in the 
"*bands*" parameter of :ref:`dataset config <yamlparameters>`. For example:

.. code-block:: yaml

    dataset:
        [...]
        bands: [3, 2, 1]

Here, if the original multi-band raster had "*BGR*" bands, geo-deep-learning will reorder these bands
to "*RGB*" order. Also, band selection and reordering can be performed simultaneously. In this example,
if the source raster contained "*BGRN*" bands, the "*N*" band would be removed and the resulting 
raster would still be as "*RGB*".

The ``bands`` parameter is set in the 
`dataset config <https://github.com/NRCan/geo-deep-learning/blob/develop/config/dataset/test_ci_segmentation_multiclass.yaml>`_.

.. note:: 

    It's the user's responsibility to know which bands correspond to each index. When dealing with 
    multi-band source imagery, Geo-Deep-Learning doesn't "*know*" which bands are present in the file 
    and in which order. Good to know, Geo-Deep-Learning follows the GDAL convention, where bands are 
    indexed from 1 
    (`docs <https://rasterio.readthedocs.io/en/latest/quickstart.html#reading-raster-data>`_).

.. _datasetsingleband:
Single-band raster files, identified with a common string pattern
=================================================================

A single line in the csv can refer to multiple single-band rasters, by using a string pattern.
The "*band specific*" string in the file name must be in a 
`hydra-like interpolation format <https://hydra.cc/docs/1.0/advanced/override_grammar/basic/#primitives>`_,
with ``${...}`` notation. Geo-deep-learning will locate a list of single-band rasters during 
execution using two informations:
1. the "*bands*" parameter of :ref:`dataset config <yamlparameters>`.
2. the input csv's path, if it contains the ``${dataset.bands}`` string.

For example:

1. Bands in `dataset configuration <https://github.com/NRCan/geo-deep-learning/blob/develop/config/dataset/test_ci_segmentation_binary.yaml>`_.:

    .. code-block:: yaml

        dataset:
            [...]
            bands: [R, G, B]

2. `Input csv <https://github.com/NRCan/geo-deep-learning/blob/develop/tests/tiling/tiling_segmentation_binary_ci.csv>`_:

    .. list-table:: 
        :header-rows: 1

        * - raster path 
          - ...
        * - my_dir/my_singleband_geofile_band_${dataset.bands}.tif
          - ...

During execution, this would result in using, **in the same order as bands appear in dataset config**, 
the following files:

.. code-block:: shell

    my_dir/my_singleband_geofile_band_R.tif
    my_dir/my_singleband_geofile_band_G.tif
    my_dir/my_singleband_geofile_band_B.tif

.. _datasetstacitem:

Single-band raster files, identified as assets in a stac item
=============================================================

Only Stac Items referencing **single-band assets** are supported currently. See 
our `Worldview-2 example <https://datacube-stage.services.geo.ca/api/collections/spacenet-samples/items/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03>`_.
Also, the `STAC spec <https://github.com/radiantearth/stac-spec/blob/master/item-spec/item-spec.md>`_
is young and quickly evolving. There exists multiple formats for stac item. Only a very specific format 
of stac item is supported by geo-deep-learning. If using stac items with geo-deep-learning, make sure 
they follow the structure of our *Worldview-2* example above. 

Bands must be selected by `common name <https://github.com/stac-extensions/eo/#common-band-names>`_ 
in dataset config:

.. code-block:: yaml

    dataset:
        [...]
        bands: ["red", "green", "blue"]

.. note::
    Order matters, ``["red", "green", "blue"]`` is not equal to ``["blue", "green", "red"]`` !


Under the hood
--------------

For use cases *Multi-band raster files with more bands or different order than needed* to 
*Single-band raster files, identified as assets in a stac item*, geo-deep-learning creates 
a `virtual raster <https://gdal.org/drivers/raster/vrt.html>`_ to bring the variable source 
raster data to a common destination format without rewriting existing data to disk.

.. _yamlparameters:

Dataset configuration yaml file
-------------------------------

.. literalinclude:: ../../config/dataset/test_ci_segmentation_multiclass.yaml
   :language: yaml

- ``name`` (str)
    Name of the dataset.
- ``raw_data_csv`` (str)
    Path to csv referencing source data for tiling or training.
- ``raw_data_dir`` (str)
    The directory where online data may be downloaded.
- ``download_data`` (bool)
    If True, all source data as url will be downloaded to raw data directory. 
    The local copy will be used afterwards.
- ``bands`` (list)
    Bands to be selected during the tiling process. Order matters 
    (ie ``["red", "green", "blue"]`` is not equal to ``["blue", "green", "red"]``). See
    :ref:`single or multi bands imagery <datasetsinglemultiband>` section for more 
    information on input data and band selection and ordering.
- ``attribute_field`` (str)
    Name of the attribute from the ground truth data.
- ``attribute_values`` (list)
    Filter only relevant classes from the ground truth data by listing the value associated to 
    the class you desire.
    For example, if a ground truth GeoPackage contains polygons belonging to 4 classes of 
    interests (forests, water bodies, roads, buildings), a user can filter out all non-building 
    polygon by choosing an attribute field and value that separate building polygons from 
    others. Though attribute values may not be continuous numbers starting at 1 
    (0 being background), Geo-deep-learning ensures all values during training are continuous
    and, therefore, match values from predictions.
- ``class_name`` (str)
    Will follow soon.
- ``classes_dict`` (dict)
    Dictionary containing the name of the class and the value associated to them.
- ``class_weights`` (dict)
    Dictionary containing the class value and the percentage that must be include 
    in the tiling dataset. The class_weights is used as a balancing strategy 
    during training and is implemented at the loss level.
- ``ignore_index`` (int)
    Specifies a target value that is ignored and does not contribute to the input gradient.

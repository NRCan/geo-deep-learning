# Dataset config

> Also see [dataset documentation](../../dataset/README.md) for general information about input data.

## Dataset-wide configuration

### Raw data csv

Path to csv referencing source data for tiling.

### Raw data directory

The directory where online data may be downloaded.

### Download data

If True, all source data as url will be downloaded to raw data directory. The local copy will be used afterwards.

## Imagery

### Bands

Bands to be selected during the tiling process. Order matters (ie ["B", "G", "R"] is not equal to ["R", "G", "B"]). See
dataset/README.md for more information on input data and band selection and ordering.

## Ground truth (polygon features)

### "Attribute_field" & "Attribute_values": filter ground truth data with

Parameters "attribute_field" and "attribute_values" aim to filter only relevant classes from the ground truth data.
They both default to None.

For example, if a ground truth GeoPackage contains polygons belonging to 4 classes of interests (forests, waterbodies,
roads, buildings), a user can filter out all non-building polygon by choosing an attribute field and value that
separate building polygons from others. For example:

```yaml
dataset:
 ...
 attribute_field: "4-Class"
 attribute_values: 4
```

> Though attribute values may not be continuous numbers starting at 1 (0 being background), Geo-deep-learning ensures
> all values during training are continuous and, therefore, match values from predictions. See "Continuous values" in
> config/tiling/README.md.

### class_name, classes_dict, class_weights

[WIP]

### Ignore_index

Specifies a target value that is ignored and does not contribute to the input gradient. Defaults to None.

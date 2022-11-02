# Dataset config

## Dataset-wide configuration

### Input dimensions and overlap

These parameters respectively set the width and length of a single sample and stride from one sample to another as
outputted by tiling_segmentation.py. Default to 256 and 0, respectively.

### Train/validation percentage

This parameters sets the proportion of samples that should be selected for each dataset. Total should be 1. Trn/val
percentages default to 0.7/0.3.

> The "tst" dataset percentage is currently not implemented. AOIs meant to be part of test dataset should be tagged
> "tst" in the last column of [the input csv](https://github.com/NRCan/geo-deep-learning/blob/develop/data/images_to_samples_ci_csv.csv).


### Stratification bias

This feature's documentation is a work in progress. Stratification bias was implemented in [PR #204]
(https://github.com/NRCan/geo-deep-learning/pull/204)
For more information on the concept of stratified sampling, see [this Medium article]
(https://medium.com/analytics-vidhya/stratified-sampling-in-machine-learning-f5112b5b9cfe)

### Raw data directory and csv

[WIP]

## Imagery

### Band selection and ordering

Bands to be selected during the tiling process. Order matters (ie "BGR" is not equal to "RGB").
The use of this feature for band selection is a work in progress. It currently serves to indicate how many bands are in
source imagery.

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
> all values during training are continuous and, therefore, match values from predictions.

### Minimum annotated percent

Minimum % of non background pixels in label in order to add sample to final training dataset. Defaults to 0.

### class_name, classes_dict, class_weights

[WIP]

### Ignore_index

Specifies a target value that is ignored and does not contribute to the input gradient. Defaults to None.

## Output

### Samples data directory

Sets the path to the directory where samples will be written.




# Dataset config

## Imagery

### Modalities

(WIP)

## Ground truth (polygon features)

### Filter ground truth data with "attribute_field" & "attribute_vals"

Parameters "attribute_field" and "attribute_vals" aim to filter only relevant classes from the ground truth data.
For example, if a ground truth GeoPackage contains polygons belonging to 4 classes of interests (forests, waterbodies, 
roads, buildings), a user can filter out all non-building polygon by choosing an attribute field and value that 
differentiate building polygons from others. For example:

```yaml
dataset:
 ...
 attribute_field: "4-Class"
 attribute_vals: 4
```

## Output

### tiles_data_dir

(WIP)

### ...




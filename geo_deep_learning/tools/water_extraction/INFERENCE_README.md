# Water Extraction Inference

End-to-end inference script for extracting waterbodies from LiDAR elevation data using trained deep learning models.

## Overview

This script performs the complete inference pipeline:

1. **Preprocessing**: Aligns rasters, computes derivatives (nDSM, TWI), stacks bands
2. **Inference**: Runs sliding-window prediction with overlap handling
3. **Raster Output**: Generates georeferenced binary prediction raster
4. **Vector Output**: Polygonizes water class and exports to GeoPackage

## Requirements

### Input Data Structure

Your AOI folder must contain:
```
my_aoi/
├── dtm.tif          # Digital Terrain Model (required)
├── dsm.tif          # Digital Surface Model (required)
├── intensity.tif    # LiDAR intensity (optional)
└── aoi.gpkg         # Area of Interest polygon (optional, for masking outputs)
```

### Model Checkpoint

- A trained PyTorch Lightning checkpoint (`.ckpt` file)
- Normalization statistics (mean/std) used during training

## Usage

### Basic Command

```bash
python -m geo_deep_learning.tools.water_extraction.inference \
    --checkpoint path/to/model.ckpt \
    --data_folder path/to/aoi \
    --output_folder path/to/outputs \
    --mean 0.0 0.0 0.0 \
    --std 1.0 1.0 1.0
```

### Full Example

```bash
python -m geo_deep_learning.tools.water_extraction.inference \
    --checkpoint /path/to/checkpoints/best_model.ckpt \
    --data_folder /data/my_watershed \
    --output_folder /outputs/my_watershed_predictions \
    --mean 5.234 2.145 87.532 \
    --std 3.421 1.876 45.231 \
    --window_size 512 \
    --stride 256 \
    --batch_size 8 \
    --device cuda \
    --simplify_tolerance 2.0
```

## Arguments

### Required

| Argument | Type | Description |
|----------|------|-------------|
| `--checkpoint` | str | Path to trained model checkpoint (`.ckpt`) |
| `--data_folder` | str | Path to folder containing input rasters |
| `--output_folder` | str | Path where outputs will be saved |
| `--mean` | float+ | Per-band mean for standardization (space-separated) |
| `--std` | float+ | Per-band std for standardization (space-separated) |

### Optional

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--window_size` | int | 512 | Size of inference windows (pixels) |
| `--stride` | int | 256 | Stride between windows (lower = more overlap) |
| `--batch_size` | int | 4 | Number of windows to process in parallel |
| `--device` | str | `cuda` | Device for inference (`cuda` or `cpu`) |
| `--no_intensity` | flag | False | Exclude intensity band from inputs |
| `--no_vector` | flag | False | Skip vector export (raster only) |
| `--simplify_tolerance` | float | 1.0 | Geometry simplification tolerance (raster units) |

## Outputs

The script generates the following outputs in `output_folder`:

**Preprocessing outputs** (saved in `preprocessed_{aoi_name}/`, only if preprocessing from raw data):
```
preprocessed_02NB000/
├── dsm_aligned.tif          # DSM aligned to DTM grid
├── intensity_aligned.tif    # Intensity aligned to DTM grid (if available)
├── ndsm.tif                 # Normalized DSM (DSM - DTM)
├── twi.tif                  # Topographic Wetness Index
└── stacked_inputs.tif       # Multi-band stack (TWI, nDSM, [intensity])
```

**Inference outputs** (saved in main `output_folder/`):
```
output_folder/
├── water_prediction_{aoi_name}_full.tif  # Full extent prediction (uncropped)
├── water_prediction_{aoi_name}.tif       # Prediction cropped to AOI boundary
└── water_bodies_{aoi_name}.gpkg          # Vectorized waterbody polygons
```

**AOI Name Extraction**:
- **From `--data_folder`**: The AOI name is extracted from the folder name
  - Example: `--data_folder /data/02NB000` → AOI name = "02NB000"
- **From `--stacked_inputs`** (if no data_folder): Inferred from parent folder name
  - Example: `--stacked_inputs /data/preprocessed_02NB000/stacked_inputs.tif` → AOI name = "02NB000"

**AOI Boundary Cropping**:
- The AOI vector (`aoi.gpkg` or `aoi.shp`) must be in the `--data_folder` (raw data folder)
- If found, predictions are automatically cropped to the AOI boundary
- The `_full.tif` file contains the complete prediction (padded to DTM extent)
- The cropped `.tif` file contains only predictions within the AOI boundary
- Vectors are extracted from the cropped raster for cleaner results

**Typical Usage Pattern**:
```bash
# Use BOTH arguments for best results:
# --stacked_inputs: points to preprocessed data for inference
# --data_folder: points to raw data folder containing aoi.gpkg for cropping
python -m geo_deep_learning.tools.water_extraction.inference \
    --checkpoint model.ckpt \
    --stacked_inputs /data/preprocessed_02NB000/stacked_inputs.tif \
    --data_folder /data/02NB000 \
    --output_folder /results/ \
    --mean 3.96 8.98 137.23 \
    --std 2.22 6.30 63.83
```

## Normalization Statistics

### Where to Find Training Stats

Normalization statistics **must match** those used during training. You can find them in:

1. **Training config file** (e.g., `config.yaml`):
   ```yaml
   data:
     mean: [5.234, 2.145, 87.532]
     std: [3.421, 1.876, 45.231]
   ```

2. **MLflow run artifacts** (if using MLflow tracking):
   - Navigate to your training run
   - Check `params/data.mean` and `params/data.std`

3. **Recompute from training tiles** (if not logged):
   ```python
   from geo_deep_learning.utils.rasters import compute_dataset_stats_from_list
   
   tile_paths = [...list of training tile paths...]
   mean, std = compute_dataset_stats_from_list(tile_paths)
   print(f"--mean {' '.join(map(str, mean))}")
   print(f"--std {' '.join(map(str, std))}")
   ```

### Band Order

Statistics must be provided in the same order as training:
- **With intensity**: `[TWI, nDSM, Intensity]`
- **Without intensity**: `[TWI, nDSM]`

## Sliding Window Strategy

### Overlap Handling

The script uses overlapping windows with weighted averaging to reduce edge artifacts:

- **Window size**: Size of each inference patch (default: 512×512)
- **Stride**: Spacing between window positions (default: 256)
- **Overlap**: `(1 - stride/window_size) × 100%` (default: 50%)

Weights are computed using a linear distance-from-center function:
- Center pixels: weight = 1.0
- Edge pixels: weight = 0.1

### Performance Tuning

| Scenario | Recommended Settings |
|----------|---------------------|
| **High accuracy** | `--window_size 512 --stride 128` (75% overlap) |
| **Balanced** | `--window_size 512 --stride 256` (50% overlap) |
| **Fast inference** | `--window_size 512 --stride 512` (no overlap) |

Memory usage: `batch_size × window_size² × num_bands × 4 bytes`

## Preprocessing Details

### Alignment

All inputs are aligned to the DTM grid using bilinear resampling:
- DSM → `dsm_aligned.tif`
- Intensity → `intensity_aligned.tif`

### Derivative Computation

**nDSM (Normalized Digital Surface Model)**:
```
nDSM = DSM - DTM
```
Represents above-ground height (vegetation, buildings).

**TWI (Topographic Wetness Index)**:
```
TWI = ln(α / tan(β))
```
Where:
- α = specific catchment area (from D8 flow accumulation)
- β = local slope

Computed using WhiteboxTools:
1. Breach depressions
2. Compute slope
3. Compute D8 flow accumulation
4. Compute wetness index

## Vector Export

### Polygonization

The water class (value=1) is extracted and converted to polygons using:
- **Geometry simplification**: Reduces vertex count while preserving topology
- **Validation**: Invalid geometries are automatically fixed
- **Filtering**: Empty or very small polygons (<1 m²) are excluded

### Output Schema

GeoPackage schema:
```
Geometry: Polygon
CRS: Same as input rasters
Properties:
  - class (int): Always 1 (water)
  - area (float): Polygon area in square raster units
```

## Example Workflows

### 1. Single AOI Inference (Recommended)

```bash
# Use preprocessed data + raw data folder for AOI boundary
python -m geo_deep_learning.tools.water_extraction.inference \
    --checkpoint models/water_unetpp_epoch50.ckpt \
    --stacked_inputs data/preprocessed_watershed_A/stacked_inputs.tif \
    --data_folder data/watershed_A \
    --output_folder results/watershed_A \
    --mean 4.12 1.89 92.45 \
    --std 2.87 1.23 38.67
```

### 2. Inference Without AOI Cropping

```bash
# If you don't have aoi.gpkg or don't want cropping
python -m geo_deep_learning.tools.water_extraction.inference \
    --checkpoint models/water_unetpp_epoch50.ckpt \
    --stacked_inputs data/preprocessed_watershed_A/stacked_inputs.tif \
    --output_folder results/watershed_A \
    --mean 4.12 1.89 92.45 \
    --std 2.87 1.23 38.67
```

### 3. Batch Processing Multiple AOIs

```bash
#!/bin/bash
# batch_inference.sh

CHECKPOINT="models/best_model.ckpt"
MEAN="4.12 1.89 92.45"
STD="2.87 1.23 38.67"

for AOI_DIR in data/preprocessed_*/; do
    AOI_NAME=$(basename $AOI_DIR | sed 's/preprocessed_//')
    echo "Processing $AOI_NAME..."
    
    python -m geo_deep_learning.tools.water_extraction.inference \
        --checkpoint $CHECKPOINT \
        --stacked_inputs ${AOI_DIR}stacked_inputs.tif \
        --data_folder data/${AOI_NAME} \
        --output_folder results/ \
        --mean $MEAN \
        --std $STD \
        --batch_size 8
done
```

### 4. CPU-Only Inference

```bash
# For systems without GPU
python -m geo_deep_learning.tools.water_extraction.inference \
    --checkpoint model.ckpt \
    --data_folder data/aoi \
    --output_folder results \
    --mean 4.12 1.89 \
    --std 2.87 1.23 \
    --device cpu \
    --batch_size 1 \
    --no_intensity
```

### 4. Raster-Only Output (Skip Vectorization)

```bash
# When you only need the raster prediction
python -m geo_deep_learning.tools.water_extraction.inference \
    --checkpoint model.ckpt \
    --data_folder data/aoi \
    --output_folder results \
    --mean 4.12 1.89 92.45 \
    --std 2.87 1.23 38.67 \
    --no_vector
```

## Troubleshooting

### Memory Issues

**Symptom**: CUDA out of memory error

**Solutions**:
1. Reduce batch size: `--batch_size 1`
2. Reduce window size: `--window_size 256`
3. Use CPU: `--device cpu`

### Mismatched Statistics

**Symptom**: Poor predictions or uniform outputs

**Cause**: Mean/std don't match training statistics

**Solution**: Double-check training config and ensure band order matches

### Missing Intensity Band

**Symptom**: Error about channel mismatch

**Cause**: Model trained with 3 bands but intensity.tif not found

**Solutions**:
1. Add `intensity.tif` to data folder, OR
2. Use `--no_intensity` flag (requires model trained on 2 bands)

### Preprocessing Errors

**Symptom**: WhiteboxTools errors during TWI computation

**Solutions**:
1. Check DTM has valid CRS and transform
2. Ensure DTM covers full AOI extent
3. Check for corrupted input files

### Empty Vector Output

**Symptom**: No water polygons exported

**Causes**:
1. Model predicted no water (check raster output)
2. Simplification tolerance too aggressive
3. All polygons filtered as too small

**Solutions**:
1. Inspect `water_prediction.tif` visually
2. Reduce `--simplify_tolerance 0.5`
3. Lower area threshold in code (currently 1.0 m²)

## Performance Notes

### Inference Speed

Approximate times for 10,000×10,000 pixel AOI:

| Configuration | GPU | Time |
|---------------|-----|------|
| Default (512/256, batch=4) | A100 | ~5 min |
| Fast (512/512, batch=8) | A100 | ~2 min |
| Accurate (512/128, batch=2) | A100 | ~15 min |
| CPU (256/256, batch=1) | 8-core | ~2 hours |

### Disk Space

Temporary files require ~2× the size of input rasters:
- Aligned rasters: ~same size as inputs
- Derivatives (nDSM, TWI): ~same size as DTM
- Stacked inputs: ~3× size of DTM (for 3 bands)
- Prediction raster: ~same size as DTM

## Advanced Usage

### Using as Python Module

```python
from geo_deep_learning.tools.water_extraction.inference import run_inference

outputs = run_inference(
    checkpoint_path="model.ckpt",
    data_folder="data/aoi",
    output_folder="results",
    mean=[4.12, 1.89, 92.45],
    std=[2.87, 1.23, 38.67],
    window_size=512,
    stride=256,
    batch_size=4,
    device="cuda",
)

print(f"Prediction raster: {outputs['prediction_raster']}")
print(f"Vector output: {outputs['prediction_vector']}")
```

### Preprocessing Only

```python
from geo_deep_learning.tools.water_extraction.inference import preprocess_aoi

stacked_inputs = preprocess_aoi(
    data_folder="data/aoi",
    output_folder="data/aoi/processed",
    include_intensity=True,
)

# Now you can use stacked_inputs for other purposes
```

### Custom Inference (Skip Preprocessing)

If you already have preprocessed `stacked_inputs.tif`:

```python
from geo_deep_learning.tools.water_extraction.inference import (
    load_model,
    sliding_window_inference,
    export_vectors,
)

# Load model
model = load_model("model.ckpt", device="cuda")

# Run inference on preprocessed data
sliding_window_inference(
    model=model,
    input_raster_path="data/aoi/stacked_inputs.tif",
    output_raster_path="results/prediction.tif",
    mean=[4.12, 1.89, 92.45],
    std=[2.87, 1.23, 38.67],
)

# Export vectors
export_vectors(
    prediction_raster_path="results/prediction.tif",
    output_vector_path="results/water.gpkg",
)
```

## Integration with Training Pipeline

### Extract Statistics from Training Run

```python
import yaml

# Load training config
with open("configs/my_training.yaml") as f:
    config = yaml.safe_load(f)

mean = config["data"]["mean"]
std = config["data"]["std"]

print(f"--mean {' '.join(map(str, mean))}")
print(f"--std {' '.join(map(str, std))}")
```

### Find Best Checkpoint

```bash
# If using PyTorch Lightning ModelCheckpoint
ls logs/lightning_logs/version_0/checkpoints/

# Use the checkpoint with lowest validation loss
# Example: epoch=49-val_loss=0.234.ckpt
```

## Citation

If you use this inference script in your research, please cite:

```bibtex
@software{geo_deep_learning,
  title = {Geo-Deep-Learning: Water Extraction from LiDAR},
  author = {Natural Resources Canada},
  year = {2024},
  url = {https://github.com/NRCan/geo-deep-learning}
}
```

## Support

For issues or questions:
1. Check [troubleshooting section](#troubleshooting)
2. Review [example workflows](#example-workflows)
3. Open an issue on GitHub with:
   - Full command used
   - Error message
   - Input data characteristics (size, CRS, resolution)

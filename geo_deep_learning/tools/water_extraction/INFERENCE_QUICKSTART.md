# Water Extraction Inference - Quick Start Guide

This guide provides a quick overview of the inference pipeline for water extraction from LiDAR elevation data.

## Files Overview

```
geo_deep_learning/tools/water_extraction/
├── inference.py              # Main inference script (end-to-end pipeline)
├── compute_stats.py          # Helper to compute normalization statistics
├── example_inference.sh      # Example bash script
├── INFERENCE_README.md       # Comprehensive documentation
├── prepare_inputs.py         # Preprocessing functions (used internally)
├── elevation_stack_dataset.py
├── elevation_stack_datamodule.py
└── segmentation_task.py      # Model task definition
```

## Quick Start (5 Steps)

### 1. Prepare Your Data

Create a folder with your AOI data:
```
my_aoi/
├── dtm.tif          # Digital Terrain Model (required)
├── dsm.tif          # Digital Surface Model (required)
└── intensity.tif    # LiDAR intensity (optional but recommended)
```

### 2. Get Normalization Statistics

**Option A: From training config**
```bash
# Check your training config file
cat configs/my_training.yaml | grep -A 2 "mean:"
# Output:
#   mean: [5.234, 2.145, 87.532]
#   std: [3.421, 1.876, 45.231]
```

**Option B: Compute from training tiles**
```bash
python -m geo_deep_learning.tools.water_extraction.compute_stats \
    --csv_path data/training.csv \
    --split trn \
    --output stats.txt
```

### 3. Locate Your Model Checkpoint

Find the best checkpoint from training:
```bash
# Look in your training output directory
ls logs/lightning_logs/version_X/checkpoints/
# Use the one with lowest validation loss, e.g.:
#   epoch=49-val_loss=0.234.ckpt
```

### 4. Run Inference

```bash
python -m geo_deep_learning.tools.water_extraction.inference \
    --checkpoint logs/.../epoch=49-val_loss=0.234.ckpt \
    --data_folder my_aoi \
    --output_folder results/my_aoi \
    --mean 5.234 2.145 87.532 \
    --std 3.421 1.876 45.231
```

### 5. View Results

```bash
# Check outputs
ls results/my_aoi/
# Output:
#   water_prediction.tif   # Binary raster (0=background, 1=water)
#   water_bodies.gpkg      # Vector polygons
#   stacked_inputs.tif     # Preprocessed inputs
#   ... (other intermediate files)

# Load in QGIS or your preferred GIS software
```

## Common Use Cases

### Full AOI with High Accuracy

```bash
python -m geo_deep_learning.tools.water_extraction.inference \
    --checkpoint model.ckpt \
    --data_folder aoi_folder \
    --output_folder results \
    --mean 5.234 2.145 87.532 \
    --std 3.421 1.876 45.231 \
    --window_size 512 \
    --stride 128 \        # 75% overlap for highest accuracy
    --batch_size 4
```

### Fast Inference (Lower Overlap)

```bash
python -m geo_deep_learning.tools.water_extraction.inference \
    --checkpoint model.ckpt \
    --data_folder aoi_folder \
    --output_folder results \
    --mean 5.234 2.145 87.532 \
    --std 3.421 1.876 45.231 \
    --stride 512 \        # No overlap = faster
    --batch_size 16
```

### CPU-Only (No GPU)

```bash
python -m geo_deep_learning.tools.water_extraction.inference \
    --checkpoint model.ckpt \
    --data_folder aoi_folder \
    --output_folder results \
    --mean 5.234 2.145 \    # 2 bands without intensity
    --std 3.421 1.876 \
    --device cpu \
    --batch_size 1 \
    --no_intensity          # Exclude intensity if not available
```

### Batch Processing

```bash
#!/bin/bash
# Process multiple AOIs

CHECKPOINT="model.ckpt"
MEAN="5.234 2.145 87.532"
STD="3.421 1.876 45.231"

for AOI in data/*/; do
    AOI_NAME=$(basename $AOI)
    echo "Processing $AOI_NAME..."
    
    python -m geo_deep_learning.tools.water_extraction.inference \
        --checkpoint $CHECKPOINT \
        --data_folder $AOI \
        --output_folder results/$AOI_NAME \
        --mean $MEAN \
        --std $STD
done
```

## Pipeline Details

### What Happens During Inference?

1. **Preprocessing** (reuses training preprocessing)
   - Align DSM and intensity to DTM grid
   - Compute nDSM = DSM - DTM
   - Compute TWI using WhiteboxTools
   - Stack bands: [TWI, nDSM, Intensity]

2. **Inference** (sliding window with overlap)
   - Divide AOI into overlapping 512×512 windows
   - Apply standardization (same as training)
   - Run model on each window
   - Merge predictions with weighted averaging

3. **Output Generation**
   - Create binary raster (threshold = 0.5)
   - Polygonize water class
   - Simplify and validate geometries
   - Export to GeoPackage

### Key Parameters

| Parameter | Purpose | Default | Tuning |
|-----------|---------|---------|--------|
| `window_size` | Size of inference patches | 512 | Larger = more context, more memory |
| `stride` | Spacing between windows | 256 | Smaller = more overlap, slower but smoother |
| `batch_size` | Windows processed together | 4 | Larger = faster (if GPU memory permits) |
| `simplify_tolerance` | Vector simplification | 1.0 | Larger = fewer vertices, simpler shapes |

## Troubleshooting

### Issue: CUDA out of memory

**Solution**: Reduce batch size or window size
```bash
--batch_size 1 --window_size 256
```

### Issue: Poor predictions

**Possible causes**:
1. Wrong normalization statistics
2. Model trained on different band configuration
3. Input data quality issues

**Debug steps**:
```bash
# 1. Verify statistics match training
cat configs/training.yaml | grep mean
cat configs/training.yaml | grep std

# 2. Check band count matches
rasterio info my_aoi/stacked_inputs.tif | grep count

# 3. Visually inspect inputs
qgis my_aoi/stacked_inputs.tif
```

### Issue: Missing intensity.tif

**Solution A**: Add `--no_intensity` flag
```bash
--no_intensity --mean 5.234 2.145 --std 3.421 1.876
```

**Solution B**: Use model trained without intensity

### Issue: Empty vector output

**Causes**:
- No water predicted (check raster first)
- All polygons filtered as too small

**Solutions**:
```bash
# Check raster visually
qgis results/water_prediction.tif

# Reduce simplification
--simplify_tolerance 0.5
```

## Performance Tips

### Speed Optimization

1. **Increase stride** (less overlap):
   ```bash
   --stride 512  # No overlap, 4× faster than default
   ```

2. **Increase batch size** (if GPU memory allows):
   ```bash
   --batch_size 16  # Process more windows in parallel
   ```

3. **Skip preprocessing** if already done:
   ```python
   # Use Python API to run only inference
   from geo_deep_learning.tools.water_extraction.inference import (
       load_model, sliding_window_inference
   )
   
   model = load_model("model.ckpt")
   sliding_window_inference(
       model=model,
       input_raster_path="aoi/stacked_inputs.tif",  # Pre-existing
       output_raster_path="results/prediction.tif",
       mean=[5.234, 2.145, 87.532],
       std=[3.421, 1.876, 45.231],
   )
   ```

### Accuracy Optimization

1. **Increase overlap**:
   ```bash
   --stride 128  # 75% overlap
   ```

2. **Ensemble predictions** (run multiple times with different windows):
   ```python
   # Advanced: Run with different offsets and average
   for offset in [0, 64, 128]:
       run_inference(..., stride=256, offset=offset)
   # Then average the prediction rasters
   ```

## Integration with Existing Workflow

### After Training

```bash
# 1. Extract best checkpoint
BEST_CKPT=$(ls logs/lightning_logs/version_0/checkpoints/*.ckpt | head -n1)

# 2. Get normalization stats from config
MEAN=$(grep "mean:" configs/training.yaml | cut -d: -f2)
STD=$(grep "std:" configs/training.yaml | cut -d: -f2)

# 3. Run inference on test AOI
python -m geo_deep_learning.tools.water_extraction.inference \
    --checkpoint $BEST_CKPT \
    --data_folder data/test_aoi \
    --output_folder results/test_aoi \
    --mean $MEAN \
    --std $STD
```

### Production Deployment

```python
# inference_service.py
from geo_deep_learning.tools.water_extraction.inference import run_inference

def process_aoi(aoi_path, output_path, checkpoint, stats):
    """Process a single AOI for production."""
    try:
        outputs = run_inference(
            checkpoint_path=checkpoint,
            data_folder=aoi_path,
            output_folder=output_path,
            mean=stats['mean'],
            std=stats['std'],
            window_size=512,
            stride=256,
            batch_size=8,
        )
        return outputs
    except Exception as e:
        log_error(f"Failed to process {aoi_path}: {e}")
        raise
```

## Next Steps

1. **Read full documentation**: [INFERENCE_README.md](INFERENCE_README.md)
2. **Review example script**: [example_inference.sh](example_inference.sh)
3. **Try test run**: Use a small AOI to verify setup
4. **Adjust parameters**: Tune for your speed/accuracy requirements
5. **Batch process**: Apply to multiple AOIs

## Support

For detailed information:
- **Full documentation**: See [INFERENCE_README.md](INFERENCE_README.md)
- **Preprocessing details**: See [README.md](README.md)
- **Training pipeline**: See main project documentation

For issues:
1. Check troubleshooting section above
2. Verify all prerequisites are met
3. Test with example data
4. Contact maintainers with full error logs

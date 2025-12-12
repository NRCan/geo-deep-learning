# Water Extraction Tools

Tools for automated water body extraction from LiDAR-derived elevation data using deep learning.

## Overview

This module extends the [geo-deep-learning](https://github.com/NRCan/geo-deep-learning) framework with specialized data preparation and processing pipelines for water body segmentation. While geo-deep-learning provides the core training infrastructure, model architectures, and generic data loading capabilities, the water extraction tools add domain-specific preprocessing for elevation data.

### Key Features

- **Automated data preparation pipeline** from raw elevation rasters to training-ready tiles
- **Topographic feature computation**: TWI (Topographic Wetness Index), nDSM (normalized Digital Surface Model)
- **Multi-source data integration**: DTM, DSM, and optional LiDAR intensity
- **AOI-based masking** to focus training on areas of interest
- **Seamless integration** with geo-deep-learning's Lightning-based training framework

## Architecture

The water extraction module consists of four main components:

1. **`prepare_inputs.py`**: Core preprocessing functions for elevation data processing
2. **`elevation_stack_datamodule.py`**: PyTorch Lightning DataModule that orchestrates the full pipeline
3. **`elevation_stack_dataset.py`**: Custom Dataset for loading tiled elevation data
4. **`download_elevation.py`**: Optional utilities for downloading elevation data from WCS services

### Integration with Geo-Deep-Learning

```
geo-deep-learning (core framework)
├── train.py                    # Training CLI
├── tasks_with_models/          # Model architectures (UNet++, SegFormer, etc.)
├── datamodules/
│   └── csv_datamodule.py       # Base DataModule
└── tools/
    └── water_extraction/       # THIS MODULE
        ├── prepare_inputs.py           # Preprocessing functions
        ├── elevation_stack_datamodule.py  # Extends CSVDataModule
        └── elevation_stack_dataset.py     # Custom dataset
```

## Quick Start

### Prerequisites

```bash
# Install geo-deep-learning and its dependencies
# Then install additional requirements for water extraction:
pip install whitebox fiona rasterio shapely
```

### Input Data Requirements

For each Area of Interest (AOI), prepare a folder with:

```
AOI_Name/
├── dtm.tif           # Digital Terrain Model (bare earth elevation)
├── dsm.tif           # Digital Surface Model (first-return elevation)
├── intensity.tif     # (Optional) LiDAR intensity
├── valid_lidar_mask.gpkg  # (Optional) polygon coverage of valid LiDAR regions
├── aoi.shp           # Polygon defining the area of interest
└── waterbodies.shp   # Polygon labels for water bodies
```

**Data specifications:**
- All rasters should be in the same CRS (coordinate reference system)
- DTM is used as the reference grid; DSM and intensity will be aligned to it
- Vector files (shapefiles) can be in any CRS; they will be reprojected as needed
- NoData values will be handled automatically
- If provided, `valid_lidar_mask.gpkg` is rasterized to `valid_mask.tif` and used
  to filter out tiles with insufficient valid coverage (>90% required by default)

## Usage

### Method 1: Fully Automated (Recommended)

This approach uses the Lightning CLI with a YAML configuration file. The DataModule automatically handles all preprocessing steps.

#### Step 1: Create a configuration file

```yaml
# config/water_extraction_config.yaml
seed_everything: 42

trainer:
  accelerator: "gpu"
  devices: 1
  max_epochs: 100
  logger:
    class_path: lightning.pytorch.loggers.mlflow.MLFlowLogger
    init_args:
      save_dir: ./logs
      experiment_name: "water_extraction"

model:
  class_path: tasks_with_models.segmentation_unetplus.SegmentationUnetPlus
  init_args:
    encoder: "resnet34"
    image_size: [512, 512]
    in_channels: 3  # TWI + nDSM + intensity
    num_classes: 2  # background + water
    loss:
      class_path: torch.nn.CrossEntropyLoss
      init_args:
        ignore_index: -1
    optimizer:
      class_path: torch.optim.AdamW
      init_args:
        lr: 1e-4
    class_labels: ["background", "water"]
    ignore_index: -1

data:
  class_path: tools.water_extraction.elevation_stack_datamodule.ElevationStackDataModule
  init_args:
    # Input/output paths
    input_folders:
      - "/path/to/AOI_1"
      - "/path/to/AOI_2"
    output_root: "/path/to/processed_data"
    csv_path: "/path/to/processed_data/training.csv"
    csv_infer_path: "/path/to/processed_data/inference.csv"

    # Data preparation parameters
    include_intensity: true
    stride: 256              # Tile stride (50% overlap with 512px patches)
    test_ratio: 0.2          # 20% for test, 20% for val, 60% for train

    # Training parameters
    patch_size: [512, 512]
    batch_size: 8
    num_workers: 4

    # Paths for loading pre-processed data
    csv_root_folder: "/path/to/processed_data"
    patches_root_folder: "/path/to/processed_data"
```

#### Step 2: Run training

```bash
python -m geo_deep_learning.train fit --config config/water_extraction_config.yaml
```

**What happens automatically:**

1. **Data preparation** (runs once, skipped if data exists):
   - For each AOI in `input_folders`:
     - Aligns DSM and intensity to DTM grid
     - Computes nDSM = DSM - DTM
     - Computes TWI using WhiteboxTools
     - Stacks layers into multi-band raster: [TWI, nDSM, intensity]
     - Rasterizes `valid_lidar_mask.gpkg` to `valid_mask.tif` when present and
       uses it to filter tiles by valid coverage
     - Rasterizes water body labels with AOI masking
     - Tiles into 512×512 patches with 256px stride
   - Generates CSV with train/val/test splits
   - Computes normalization statistics (mean/std per channel)

2. **Training**: Standard PyTorch Lightning training loop

3. **Validation & Testing**: Automatic evaluation on held-out data

### Method 2: Manual Preprocessing

For more control over preprocessing, use individual functions:

```python
from pathlib import Path
from geo_deep_learning.tools.water_extraction.prepare_inputs import (
    align_to_reference,
    compute_ndsm,
    compute_twi_whitebox,
    stack_rasters,
    rasterize_labels_binary_aoi_mask,
    tile_raster_pair,
    generate_csv_from_tiles,
)

# Define paths
aoi_folder = Path("/path/to/AOI_Name")
output_folder = Path("/path/to/processed/AOI_Name")
output_folder.mkdir(parents=True, exist_ok=True)

# Step 1: Align rasters to DTM grid
dtm = str(aoi_folder / "dtm.tif")
dsm = str(aoi_folder / "dsm.tif")
intensity = str(aoi_folder / "intensity.tif")

dsm_aligned = str(output_folder / "dsm_aligned.tif")
intensity_aligned = str(output_folder / "intensity_aligned.tif")

align_to_reference(dtm, dsm, dsm_aligned)
align_to_reference(dtm, intensity, intensity_aligned)

# Step 2: Compute nDSM
ndsm_path = str(output_folder / "ndsm.tif")
compute_ndsm(dsm_aligned, dtm, ndsm_path)

# Step 3: Compute TWI
twi_path = str(output_folder / "twi.tif")
compute_twi_whitebox(dtm, twi_path)

# Step 4: Stack rasters
stacked_path = str(output_folder / "stacked_inputs.tif")
stack_rasters([twi_path, ndsm_path, intensity_aligned], stacked_path)

# Step 5: Rasterize labels
labels_path = str(output_folder / "labels.tif")
rasterize_labels_binary_aoi_mask(
    label_vector_path=str(aoi_folder / "waterbodies.shp"),
    aoi_vector_path=str(aoi_folder / "aoi.shp"),
    reference_raster_path=stacked_path,
    output_path=labels_path,
    burn_value=1,
    fill_value=0,
    ignore_value=-1,
)

# Step 6: Tile
tiles_folder = str(output_folder / "tiles")
tile_raster_pair(
    input_path=stacked_path,
    label_path=labels_path,
    output_dir=tiles_folder,
    patch_size=512,
    stride=256,
)

# Step 7: Generate CSV
generate_csv_from_tiles(
    root_output_folder=str(output_folder.parent),
    csv_tiling_path=str(output_folder.parent / "training.csv"),
    csv_inference_path=str(output_folder.parent / "inference.csv"),
    test_ratio=0.2,
)
```

### Method 3: Inference on New AOIs

Process new data without labels for prediction:

```python
from geo_deep_learning.tools.water_extraction.prepare_inputs import (
    prepare_inference_dataset,
)

# Prepare stacked inputs
aoi_folder = "/path/to/new_AOI"  # Contains dtm.tif, dsm.tif, intensity.tif
output_folder = "/path/to/inference_output"

stacked_path = prepare_inference_dataset(
    aoi_folder=aoi_folder,
    output_folder=output_folder,
)

# Then use with your trained model
# (tile the stacked raster, run inference, mosaic results)
```

## Data Processing Pipeline Details

### 1. Alignment (`align_to_reference`)
- **Purpose**: Ensure all input rasters have matching extent, resolution, and CRS
- **Process**: Reprojects and resamples DSM and intensity to match DTM grid
- **Method**: Bilinear resampling (configurable)

### 2. nDSM Computation (`compute_ndsm`)
- **Formula**: nDSM = DSM - DTM
- **Purpose**: Height of objects above ground (vegetation, buildings)
- **Processing**: Chunked to handle large rasters efficiently
- **Output**: Float32 raster with NoData handling

### 3. TWI Computation (`compute_twi_whitebox`)
- **Tool**: WhiteboxTools hydrological analysis
- **Steps**:
  1. Breach depressions in DTM
  2. Compute slope
  3. Compute D8 flow accumulation (specific catchment area)
  4. Calculate TWI = ln(SCA / tan(slope))
- **Purpose**: Topographic wetness indicator (higher values = wetter areas)
- **Output**: Float32 raster

### 4. Raster Stacking (`stack_rasters`)
- **Combines**: TWI, nDSM, and optionally intensity into one multi-band raster
- **Band order**: [TWI, nDSM, Intensity] (if intensity included)
- **Output**: Multi-band GeoTIFF with consistent NoData values

### 5. Label Rasterization (`rasterize_labels_binary_aoi_mask`)
- **Inputs**: Water body polygons, AOI polygon
- **Process**:
  - Burns water bodies as value 1
  - Sets background within AOI as 0
  - Sets pixels outside AOI as -1 (ignore_index)
- **Output**: Int16 raster matching input grid

### 6. Tiling (`tile_raster_pair`)
- **Strategy**: Sliding window with configurable stride
- **Default**: 512×512 patches with 256px stride (50% overlap)
- **Filtering**: Discards tiles with <90% valid pixels
- **Output**:
  - `tiles/inputs/tile_XXXX.tif` (multi-band)
  - `tiles/labels/tile_XXXX_label.tif` (single-band)

### 7. CSV Generation (`generate_csv_from_tiles`)
- **Scans**: All AOI tile folders
- **Filtering**: Removes tiles with zero water pixels (optional)
- **Splitting**: Stratified by AOI, shuffled globally
  - Training: 60% (default with test_ratio=0.2)
  - Validation: 20%
  - Test: 20%
- **Output**: CSV with columns: `tif`, `gpkg`, `split`, `aoi`

## Configuration Parameters

### DataModule Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_folders` | list[str] | `[]` | List of AOI folders with raw data |
| `output_root` | str | `""` | Root directory for processed outputs |
| `csv_path` | str | `""` | Path for training/validation CSV |
| `csv_infer_path` | str | `""` | Path for inference CSV |
| `include_intensity` | bool | `False` | Whether to use intensity as 3rd channel |
| `stride` | int | `256` | Stride for tiling (overlap = patch_size - stride) |
| `test_ratio` | float | `0.2` | Proportion of data for test split |
| `patch_size` | tuple | `(512, 512)` | Size of training patches |
| `batch_size` | int | `16` | Training batch size |
| `num_workers` | int | `8` | DataLoader workers |

### Model Considerations

- **`in_channels`**: Must match number of stacked bands (2 or 3)
- **`num_classes`**: 2 for binary water/background
- **`ignore_index`**: -1 (pixels outside AOI)
- **`image_size`**: Should match `patch_size`

## Output Directory Structure

```
processed_data/
├── training.csv                 # Full dataset with splits
├── inference.csv                # Test split for inference
├── norm_stats.json             # Mean/std per channel
└── AOI_Name/
    ├── processed/
    │   ├── dsm_aligned.tif
    │   ├── intensity_aligned.tif
    │   ├── ndsm.tif
    │   ├── twi.tif
    │   ├── stacked_inputs.tif
    │   └── temp_whitebox/      # Temporary TWI intermediate files
    ├── labels.tif
    └── tiles/
        ├── inputs/
        │   ├── tile_0000.tif
        │   ├── tile_0001.tif
        │   └── ...
        └── labels/
            ├── tile_0000_label.tif
            ├── tile_0001_label.tif
            └── ...
```

## Advanced Usage

### Skipping Pre-processing

If data is already processed, the DataModule will detect and skip:

```python
# DataModule checks for existence of:
# - csv_path
# - All tile directories
# If found, skips prepare_data()
```

### Custom Tile Filtering

Modify filtering logic in `generate_csv_from_tiles`:

```python
# Default: removes tiles with zero water pixels
generate_csv_from_tiles(
    ...,
    remove_empty_labels=True,  # Set to False to keep all tiles
)
```

### Multi-AOI Training

The framework handles multiple AOIs automatically:

```yaml
data:
  init_args:
    input_folders:
      - "/data/region_1/site_A"
      - "/data/region_1/site_B"
      - "/data/region_2/site_C"
```

Each AOI is processed independently, then combined in the CSV with stratified splitting.

### Computing Custom Statistics

Override normalization statistics:

```python
# In config
data:
  init_args:
    mean: [0.5, 0.5, 0.5]  # Manual mean per channel
    std: [0.2, 0.2, 0.2]   # Manual std per channel
```

## Troubleshooting

### WhiteboxTools Issues

If TWI computation fails:
- Ensure WhiteboxTools is installed: `pip install whitebox`
- Check DTM has valid elevation data (not all NoData)
- Verify DTM CRS is projected (not geographic)

### Memory Issues

For large rasters:
- Increase `block_size` in `compute_ndsm` for larger chunks
- Process AOIs sequentially rather than in parallel
- Use smaller `patch_size` (e.g., 256×256)

### Missing Intensity Data

If intensity is unavailable:
- Set `include_intensity: false` in config
- Update model `in_channels: 2`
- Only TWI and nDSM will be used

### Alignment Warnings

If rasters don't align perfectly:
- Check that all inputs cover the same geographic area
- Verify CRS compatibility
- Ensure DTM quality (it's the reference)

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{geodl_water_extraction,
  title = {Water Extraction Tools for Geo-Deep-Learning},
  author = {Natural Resources Canada},
  year = {2025},
  url = {https://github.com/NRCan/geo-deep-learning}
}
```

## Contributing

This module is part of the geo-deep-learning framework. For contributions:

1. Follow the [geo-deep-learning contribution guidelines](https://github.com/NRCan/geo-deep-learning/CONTRIBUTING.md)
2. Ensure all pre-commit hooks pass (`ruff`, `ruff format`)
3. Add tests for new preprocessing functions
4. Update this README for new features

## License

See the main [geo-deep-learning LICENSE](https://github.com/NRCan/geo-deep-learning/blob/main/LICENSE).

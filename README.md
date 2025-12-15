# Geo Deep Learning

A PyTorch Lightning-based framework for geospatial deep learning with multi-sensor Earth observation data.

## Overview

Geo Deep Learning (GDL) is a modular framework designed to support a wide range of geospatial deep learning tasks such as semantic segmentation, object detection, and regression.
Built on PyTorch Lightning, it provides efficient training pipelines for multi-sensor data.

## Features

- **Multi-sensor Support**: Handle multiple Earth observation sensors simultaneously.
- **Modular Architecture**: Encoder-neck-decoder pattern with interchangeable components.
- **WebDataset Integration**: Efficient large-scale data loading and processing.
- **Multiple Model Types**: UNet++, SegFormer, DOFA (Dynamic-one-for-all Architecture).
- **Distributed Training**: Multi-GPU training with supported strategies.
- **MLflow Logging**: Comprehensive experiment tracking and model versioning.
- **Flexible Data Pipeline**: Support for CSV and WebDataset formats.

## Architecture

```
├── models/
│   ├── encoders/          # DOFA, MixTransformer backbones
│   ├── necks/             # Multi-level feature processing
│   ├── decoders/          # UperNet decoder implementation
│   └── heads/             # Segmentation heads (FCN, etc.)
├── datamodules/           # Lightning DataModules
├── datasets/              # WebDataset and CSV dataset implementations
├── tasks_with_models/     # Lightning modules for training
├── tools/                 # Utilities, callbacks, visualization
└── samplers/              # Custom data sampling strategies
```

## Requirements
- Install [uv](https://docs.astral.sh/uv/) package manager for your OS.

## Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/NRCan/geo-deep-learning.git
cd geo-deep-learning
```
2. **Install dependencies:**

For **GPU training** with CUDA 12.8:
```bash
uv sync --extra cu128
```

For **CPU-only** training:
```bash
uv sync --extra cpu
```
This creates a virtual environment in `.venv/` and installs all dependencies.

3. **Activate the environment:**
```bash
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Or use `uv run` to execute commands without manual activation:
```bash
uv run python geo_deep_learning/train.py fit --config configs/dofa_config_RGB.yaml
```
**Note:** *If you prefer to use conda or another environment manager, you can generate a `requirements.txt` file from the dependencies listed in `pyproject.toml` for manual installation.*

### Configuration

Models are configured via YAML files in the `configs/` directory:

```yaml
model:
  class_path: tasks_with_models.segmentation_dofa.SegmentationDOFA
  init_args:
    encoder: "dofa_base"
    pretrained: true
    image_size: [512, 512]
    num_classes: 5
    # ... other parameters

data:
  class_path: datamodules.wds_datamodule.MultiSensorDataModule
  init_args:
    sensor_configs_path: "path/to/sensor_configs.yaml"
    batch_size: 16
    patch_size: [512, 512]

trainer:
  max_epochs: 100
  precision: 16-mixed
  accelerator: gpu
  devices: 1
```

## Supported Models

### UNet++
- Classic U-Net architecture with dense skip connections.
- Multiple encoder backbones (ResNet, EfficientNet, etc.).
- Available through segmentation-models-pytorch.

### SegFormer
- Transformer-based architecture for semantic segmentation.
- Hierarchical feature representation (MixTransformer encoder).
- Multiple model sizes (B0-B5).

### DOFA (Dynamic One-For-All foundation model)
- **DOFA Base**: 768-dim embeddings, suitable for most tasks.
- **DOFA Large**: 1024-dim embeddings, higher capacity.
- Multi-scale feature extraction with UperNet decoder.
- Support for wavelength-specific processing.

### MAE
- Masked Autoencoder with Vision Transformer backbone for self-supervised pretraining
- Reconstructs masked image patches to learn rich visual representations
- Configurable mask ratio and patch-based reconstruction loss

### Swin MAE
- Masked Autoencoder with Swin Transformer backbone for self-supervised pretraining
- Hierarchical feature extraction with shifted window attention mechanisms
- Efficient pretraining for downstream geospatial tasks

### Swin Unet
- Swin Transformer-based encoder-decoder architecture for semantic segmentation
- Hierarchical feature representation with skip connections
- U-Net style decoder for pixel-level classification

### Swin Classifier
- Instance-level building classifier using Swin Transformer encoders
- Processes RGB images and binary masks through separate encoders
- Fuses image features with bounding box information for classification
  
## Data Pipeline

### Multi-Sensor DataModule
- **Sensor Mixing**: Combine data from multiple sensors during training.
- **WebDataset Format**: Efficient sharded data storage and loading.

### Supported Data Formats
- **WebDataset**: Sharded tar files with metadata.
- **CSV**: Traditional CSV with file paths and labels.
- **Multi-sensor**: YAML configuration for sensor-specific settings.

## Training Features
- **Large-scale training**: Distributed training strategies enabled with pytorch lightning.
- **Mixed Precision Training**: 16-bit mixed precision for faster training.
- **Gradient Clipping**: Configurable gradient clipping for stability.
- **Early Stopping**: Automatic training termination based on validation metrics.
- **Model Checkpointing**: Saves best models based on validation performance.
- **MLflow Integration**: Experiment tracking, metrics logging, and model registry.
- **Visualization Callbacks**: Built-in prediction visualization during training.
- **Learning Rate Scheduling**: Cosine annealing, step decay, and more.

## Development

### Code Style
- Follows PEP 8 with 88-character line limit
- Uses Ruff for linting and formatting
- Type hints for all function signatures
- Comprehensive docstrings

```bash
# Lint code
ruff check .

# Format code
ruff format .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Open an issue on GitHub

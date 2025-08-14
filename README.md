# Geo Deep Learning

A PyTorch Lightning-based framework for geospatial deep learning with multi-sensor Earth observation data.

## Overview

Geo Deep Learning (GDL) is a modular framework designed for semantic segmentation of geospatial imagery using state-of-the-art deep learning models. Built on PyTorch Lightning, it provides efficient training pipelines for multi-sensor data with WebDataset support.

## Features

- **Multi-sensor Support**: Handle multiple Earth observation sensors simultaneously
- **Modular Architecture**: Encoder-neck-decoder pattern with interchangeable components
- **WebDataset Integration**: Efficient large-scale data loading and processing
- **Multiple Model Types**: UNet++, SegFormer, DOFA (Dynamic-one-for-all Architecture)
- **Distributed Training**: Multi-GPU training with DDP strategy
- **MLflow Logging**: Comprehensive experiment tracking and model versioning
- **Flexible Data Pipeline**: Support for CSV and WebDataset formats

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

## Quick Start

```bash
git clone <repository-url>
cd geo-deep-learning
```

### Training

```bash
# Single GPU training
python geo_deep_learning/train.py fit --config configs/dofa_config_RGB.yaml
```

### Configuration

Models are configured via YAML files in `configs/`:

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
```

## Supported Models

### DOFA (Domain-Oriented Foundation Architecture)
- **DOFA Base**: 768-dim embeddings, suitable for most tasks
- **DOFA Large**: 1024-dim embeddings, higher capacity
- Multi-scale feature extraction with UperNet decoder
- Support for wavelength-specific processing

### UNet++
- Classic U-Net architecture with dense skip connections
- Multiple encoder backbones (ResNet, EfficientNet, etc.)
- Optimized for medical and satellite imagery

### SegFormer
- Transformer-based architecture for semantic segmentation
- Hierarchical feature representation
- Efficient attention mechanisms

## Data Pipeline

### Multi-Sensor DataModule
- **Sensor Mixing**: Combine data from multiple sensors during training
- **WebDataset Format**: Efficient sharded data storage and loading
- **Patch-based Processing**: Configurable patch sizes (default: 512x512)
- **Data Augmentation**: Built-in augmentation pipeline

### Supported Data Formats
- **WebDataset**: Sharded tar files with metadata
- **CSV**: Traditional CSV with file paths and labels
- **Multi-sensor**: YAML configuration for sensor-specific settings

## Training Features

- **Mixed Precision**: 16-bit mixed precision training
- **Gradient Clipping**: Configurable gradient clipping
- **Early Stopping**: Automatic training termination
- **Model Checkpointing**: Best model saving based on validation metrics
- **Visualization**: Built-in prediction visualization callbacks

## Distributed Training

The framework supports multi-GPU training with:
- DDP (Distributed Data Parallel) strategy
- Automatic mixed precision
- Synchronized batch normalization
- Efficient NCCL communication

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

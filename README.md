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

## Installation

This project supports **CPU-only** and **GPU (CUDA 12.8)** PyTorch builds.
The two environments are fully isolated and reproducible via separate lockfiles.

### Requirements

- Python **3.12**
- [uv](https://docs.astral.sh/uv/) package manager

### CPU-only

Installs CPU-only PyTorch and works everywhere.

```bash
git clone https://github.com/NRCan/geo-deep-learning.git
cd geo-deep-learning

cp uv.lock.cpu uv.lock
uv sync --extra cpu --extra dev --frozen
```

Verify:

```bash
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
# Expected: 2.x.x+cpu CUDA: False
```

### GPU (CUDA 12.8)

Use only on systems with NVIDIA GPUs and CUDA 12.8–compatible drivers.

```bash
cp uv.lock.cu128 uv.lock
uv sync --extra cu128 --extra dev --frozen
```

Verify:

```bash
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
# Expected: 2.x.x+cu128 CUDA: True
```

### Lockfiles

| File        | PyTorch build | Use case                    |
|-------------|---------------|-----------------------------|
| `uv.lock.cpu`  | CPU-only      | CI, laptops with no GPUs   |
| `uv.lock.cu128`| CUDA 12.8     | GPU training                |

Copy the appropriate lockfile to `uv.lock` before running `uv sync`. Do not edit `uv.lock` manually.

### Activate and run

```bash
source .venv/bin/activate   # Linux/macOS
# or: .venv\Scripts\activate  # Windows

uv run python geo_deep_learning/train.py fit --config configs/dofa_config_RGB.yaml
```

### Troubleshooting

If you see `libcudart`, `nvidia-*`, or `cu12` errors on a CPU machine: ensure `uv.lock.cpu` is active and re-run `uv sync --extra cpu --frozen`.

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

<br><br>

# Geo Deep Learning

Un cadre d'applications basé sur PyTorch Lightning pour l’apprentissage profond géospatial exploitant des données d’observation de la Terre multisources.

## Aperçu

Geo Deep Learning est un cadre modulaire conçu pour prendre en charge un large éventail de tâches d’apprentissage profond géospatial, notamment la segmentation sémantique, la détection d’objets et la régression. 
Basé sur PyTorch Lightning, il fournit des pipelines d’entraînement efficaces pour les données multisources.

## Fonctionnalités

- **Prise en charge multisource** : traitement simultané de données provenant de plusieurs capteurs d’observation de la Terre.
- **Architecture modulaire** : structure encodeur-neck-décodeur avec composants interchangeables.
- **Intégration de WebDataset** : chargement et traitement efficaces de grands volumes de données.
- **Multiples types de modèles** : UNet++, SegFormer, DOFA (Dynamic One-For-All Architecture).
- **Entraînement distribué** : entraînement multi-GPU avec différentes stratégies prises en charge.
- **Journalisation avec MLflow** : suivi complet des expériences et gestion des versions des modèles.
- **Pipeline de données flexible** : prise en charge des formats CSV et WebDataset.

## Architecture

```
├── models/
│   ├── encoders/          # Architectures DOFA et MixTransformer
│   ├── necks/             # Traitement de caractéristiques multimodales
│   ├── decoders/          # Implémentation du décodeur UperNet
│   └── heads/             # Têtes de segmentation (FCN, etc.)
├── datamodules/           # DataModules Lightning
├── datasets/              # Implémentations WebDataset et CSV
├── tasks_with_models/     # Modules Lightning pour l’entraînement
├── tools/                 # Utilitaires, rappels et visualisation
└── samplers/              # Stratégies personnalisées d’échantillonnage
```

## Installation

Ce projet prend en charge les versions **CPU-only** et **GPU (CUDA 12.8)** de PyTorch. 
Les deux environnements sont entièrement isolés et reproductibles grâce à des fichiers de verrouillage distincts.

### Exigences

- Python **3.12**
- Gestionnaire de paquets [uv](https://docs.astral.sh/uv/)

### CPU uniquement

Installe une version de PyTorch pour processeur (CPU uniquement) et fonctionne sur toutes les plateformes.

```bash
git clone https://github.com/NRCan/geo-deep-learning.git
cd geo-deep-learning

cp uv.lock.cpu uv.lock
uv sync --extra cpu --extra dev --frozen
```

Vérification:

```bash
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
# Expected: 2.x.x+cpu CUDA: False
```

### GPU (CUDA 12.8)

Utilisez uniquement sur les systèmes équipés de processeurs graphiques (GPU) NVIDIA et de pilotes compatibles avec CUDA 12.8.

```bash
cp uv.lock.cu128 uv.lock
uv sync --extra cu128 --extra dev --frozen
```

Vérification:

```bash
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
# Expected: 2.x.x+cu128 CUDA: True
```

### Fichiers de verrouillage

| File        | PyTorch build | Use case                    |
|-------------|---------------|-----------------------------|
| `uv.lock.cpu`  | CPU-only      | CI, laptops with no GPUs   |
| `uv.lock.cu128`| CUDA 12.8     | GPU training                |

Copiez le fichier approprié vers « uv.lock » avant d’exécuter « uv sync ». Ne modifiez pas manuellement le fichier « uv.lock ».

### Activation et exécution

```bash
source .venv/bin/activate   # Linux/macOS
# or: .venv\Scripts\activate  # Windows

uv run python geo_deep_learning/train.py fit --config configs/dofa_config_RGB.yaml
```

### Dépannage

Si vous obtenez des erreurs liées à « libcudart », « nvidia-* » ou « cu12 » sur une machine sans GPU, assurez-vous que « uv.lock.cpu » est actif et réexécutez : `uv sync --extra cpu --frozen`.

### Configuration

Les modèles sont configurés au moyen de fichiers YAML situés dans le répertoire « configs/ ».

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

## Modèles pris en charge

### UNet++
- Architecture U-Net classique avec connexions de saut denses.
- Prise en charge de plusieurs encodeurs (ResNet, EfficientNet, etc.).
- Disponible via segmentation-models-pytorch.

### SegFormer
- Architecture de segmentation sémantique basée sur les transformateurs.
- Représentation hiérarchique des caractéristiques (encodeur MixTransformer).
- Plusieurs tailles de modèles disponibles (B0 à B5).

### DOFA (Dynamic One-For-All Foundation Model)
- **DOFA Base** : représentations de 768 dimensions, adaptées à la plupart des tâches.
- **DOFA Large** : représentations de 1 024 dimensions, capacité accrue.
- Extraction de caractéristiques multi-échelles avec décodeur UperNet.
- Prise en charge du traitement spécifique aux longueurs d’onde.

## Pipeline de données

### DataModule multisource
- **Combinaison de capteurs** : intégration de données provenant de plusieurs capteurs durant l’entraînement.
- **Format WebDataset** : stockage et chargement efficaces à partir de jeux de données fragmentés.

### Formats de données pris en charge
- **WebDataset** : fichiers TAR segmentés accompagnés de métadonnées.
- **CSV** : format CSV traditionnel contenant les chemins d’accès aux fichiers et les étiquettes.
- **Multisource** : configuration YAML pour les paramètres propres à chaque capteur.

## Fonctionnalités d’entraînement
- **Entraînement à grande échelle** : stratégies d’entraînement distribué avec PyTorch Lightning.
- **Précision mixte** : calcul en précision mixte 16 bits pour accélérer l’entraînement.
- **Écrêtage des gradients** : paramétrable pour améliorer la stabilité.
- **Arrêt anticipé** : interruption automatique selon les performances de validation.
- **Sauvegarde des modèles** : enregistrement des meilleurs modèles en fonction des métriques de validation.
- **Intégration MLflow** : suivi des expériences, journalisation des métriques et registre de modèles.
- **Rappels de visualisation** : visualisation intégrée des prédictions pendant l’entraînement.
- **Planification du taux d’apprentissage** : recuit cosinus, décroissance par paliers, etc.

## Développement

### Style de code
- Conforme à la norme PEP 8 avec une limite de 88 caractères par ligne.
- Utilise Ruff pour l’analyse et le formatage du code.
- Annotations de type pour toutes les signatures de fonctions.
- Documentation complète sous forme de docstrings.

```bash
# Vérifier le code
ruff check .

# Formater le code
ruff format .
```

## Contribution

1.	Dupliquez (fork) le dépôt.
2.	Créez une branche pour votre fonctionnalité.
3.	Apportez vos modifications.
4.	Ajoutez des tests, le cas échéant.
5.	Soumettez une demande d’intégration (pull request).

## Soutien

Pour toute question ou problème :
- Ouvrez un ticket (issue) sur GitHub.

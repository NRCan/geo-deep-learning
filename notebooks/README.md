# Notebooks

This folder contains example notebooks to help you get started with **Geo Deep Learning (GDL)**.

## Available Notebooks

- **`00_quickstart.ipynb`**
  Minimal end-to-end demo:
  1. Prepare a small sample dataset
  2. Train a UNet++ model on CPU
  3. Run inference & visualize predictions

  This version calls **GDL’s core classes directly** (no config files).
  It is meant as the simplest entry point to verify everything works.

- **`01_quickstart_config.ipynb`** *(coming soon)*
  Same workflow as above, but using **LightningCLI** and GDL’s config files.
  This is the recommended way for reproducible experiments.

## Requirements
  - GDL repository cloned locally
  - Environment with proper dependencies (see `requirements.txt` or `pyproject.toml`

## Troubleshooting
    `ModuleNotFoundError: No module named 'geo_deep_learning'` (or other module)

    In general, this problem occurs when the paths are not properly defined. Make sure
    to add the repo to your PYTHONPATH.

    Example inside a notebook:

    ```python
    import sys
    sys.path.append("..")
    ```

"""
Water extraction tools for geo-deep-learning.

This module contains specialized datasets and datamodules for water body extraction
tasks using elevation data such as DTM, DSM, and derived products like TWI and nDSM.

Examples:
    >>> from geo_deep_learning.tools.water_extraction import ElevationStackDataModule
    >>> from geo_deep_learning.tools.water_extraction import ElevationStackDataset

    >>> # Use the datamodule for training
    >>> datamodule = ElevationStackDataModule(
    ...     csv_root_folder="path/to/csv",
    ...     mean=[0.5, 0.5, 0.5],
    ...     std=[0.25, 0.25, 0.25]
    ... )

"""

from .elevation_stack_datamodule import ElevationStackDataModule
from .elevation_stack_dataset import ElevationStackDataset

__all__ = ["ElevationStackDataModule", "ElevationStackDataset"]

#!/usr/bin/env python
"""Prepare data for water extraction training (no GPU required)."""

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

from geo_deep_learning.tools.water_extraction.elevation_stack_datamodule import (
    ElevationStackDataModule,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Prepare data from config file."""
    parser = argparse.ArgumentParser(
        description="Prepare data for water extraction training (no GPU required)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    logger.info("Loading config from: %s", config_path)
    config = OmegaConf.load(config_path)

    # Extract data module config
    if "data" not in config:
        msg = "Config must contain 'data' section"
        raise ValueError(msg)

    data_config = config["data"]
    if "init_args" not in data_config:
        msg = "Data config must contain 'init_args'"
        raise ValueError(msg)

    # Initialize datamodule
    logger.info("Initializing datamodule...")
    datamodule = ElevationStackDataModule(**data_config["init_args"])

    # Run data preparation
    logger.info("Starting data preparation...")
    datamodule.prepare_data()

    logger.info("Data preparation complete!")
    logger.info("CSV files saved to:")
    logger.info("  Training: %s", datamodule.csv_path)
    logger.info("  Inference: %s", datamodule.csv_infer_path)
    logger.info("")
    logger.info("You can now run training with:")
    logger.info("  python -m geo_deep_learning.train fit --config %s", args.config)


if __name__ == "__main__":
    main()

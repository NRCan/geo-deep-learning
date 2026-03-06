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
    """Prepare data from config file with optional CLI overrides."""
    parser = argparse.ArgumentParser(
        description="Prepare data for water extraction training (no GPU required)",
        epilog="""
Examples:
  # Use config as-is
  python -m geo_deep_learning.tools.water_extraction.prepare_data \\
      --config config.yaml

  # Override specific parameters
  python -m geo_deep_learning.tools.water_extraction.prepare_data \\
      --config config.yaml \\
      --data.init_args.project_extents_path=/path/to/extents.gpkg \\
      --data.init_args.regenerate_csv=true
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    # Parse known args to allow OmegaConf overrides from command line
    args, cli_overrides = parser.parse_known_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    logger.info("Loading config from: %s", config_path)
    config = OmegaConf.load(config_path)

    # Apply CLI overrides using OmegaConf
    if cli_overrides:
        logger.info("Applying CLI overrides: %s", cli_overrides)
        cli_config = OmegaConf.from_cli(cli_overrides)
        config = OmegaConf.merge(config, cli_config)

    # Extract data module config
    if "data" not in config:
        msg = "Config must contain 'data' section"
        raise ValueError(msg)

    data_config = config["data"]
    if "init_args" not in data_config:
        msg = "Data config must contain 'init_args'"
        raise ValueError(msg)

    # Convert OmegaConf to dict for datamodule initialization
    init_args = OmegaConf.to_container(data_config["init_args"], resolve=True)

    # Initialize datamodule
    logger.info("Initializing datamodule...")
    logger.info("Data config: %s", OmegaConf.to_yaml(data_config["init_args"]))
    datamodule = ElevationStackDataModule(**init_args)

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

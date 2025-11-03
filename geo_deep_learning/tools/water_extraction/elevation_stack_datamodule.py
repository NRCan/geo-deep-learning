"""Elevation Stack DataModule that extends CSVDataModule for water extraction tasks."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from lightning.pytorch.utilities import rank_zero_only
from torch.utils.data import DataLoader

from geo_deep_learning.datamodules.csv_datamodule import CSVDataModule
from geo_deep_learning.utils.rasters import compute_dataset_stats_from_list

from .elevation_stack_dataset import ElevationStackDataset
from .prepare_inputs import (
    align_to_reference,
    compute_ndsm,
    compute_twi_whitebox,
    generate_csv_from_tiles,
    rasterize_labels_binary_aoi_mask,
    stack_rasters,
    tile_raster_pair,
)

# Define logger
log = logging.getLogger(__name__)


class ElevationStackDataModule(CSVDataModule):
    """
    DataModule for elevation stack data with water body labels.

    Extends CSVDataModule to handle data preparation for water extraction tasks
    including elevation data processing, stacking, tiling, and CSV generation.

    This module handles the full pipeline from raw elevation data to tiled
    training/validation/test datasets ready for deep learning models.
    """

    def __init__(  # noqa: PLR0913
        self,
        batch_size: int = 16,
        num_workers: int = 8,
        data_type_max: int = 255,
        patch_size: tuple[int, int] = (512, 512),
        mean: list[float] | None = None,
        std: list[float] | None = None,
        csv_root_folder: str = "",
        patches_root_folder: str = "",
        # Additional parameters for elevation stack processing
        input_folders: list[str] | None = None,
        output_root: str = "",
        csv_path: str = "",
        csv_infer_path: str = "",
        include_intensity: bool = False,
        stride: int = 256,
        test_ratio: float = 0.2,
    ) -> None:
        """
        Initialize ElevationStackDataModule.

        Args:
            batch_size (int): Batch size for dataloaders. Defaults to 16.
            num_workers (int): Number of workers for dataloaders. Defaults to 8.
            data_type_max (int): Maximum data type value. Defaults to 255.
            patch_size (tuple[int, int]): Size of patches. Defaults to (512, 512).
            mean (list[float] | None): Mean values for normalization.
            std (list[float] | None): Std values for normalization.
            csv_root_folder (str): Root folder for CSV files.
            patches_root_folder (str): Root folder for patches.
            input_folders (list[str] | None): List of AOI folders containing raw data.
            output_root (str): Root directory for processed outputs.
            csv_path (str): Path for training/validation CSV file.
            csv_infer_path (str): Path for inference CSV file.
            include_intensity (bool): Whether to include intensity data.
                Defaults to False.
            stride (int): Stride for tiling. Defaults to 256.
            test_ratio (float): Ratio for test split. Defaults to 0.2.

        """
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            data_type_max=data_type_max,
            patch_size=patch_size,
            mean=mean,
            std=std,
            csv_root_folder=csv_root_folder,
            patches_root_folder=patches_root_folder,
        )

        # Additional elevation-specific parameters
        self.input_folders = input_folders or []
        self.output_root = output_root
        self.csv_path = csv_path
        self.csv_infer_path = csv_infer_path
        self.intensity = include_intensity
        self.stride = stride
        self.test_ratio = test_ratio

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """
        Create datasets for each split.

        Overrides the base class setup to use ElevationStackDataset
        with single CSV file containing split information.

        Args:
            stage: Stage of training (not used in this implementation).

        """
        # Use our custom dataset that handles split column
        self.train_dataset = ElevationStackDataset(
            split="trn",
            norm_stats=self.norm_stats,
            csv_root_folder=self.csv_root_folder,
            patches_root_folder=self.patches_root_folder,
        )
        self.val_dataset = ElevationStackDataset(
            split="val",
            norm_stats=self.norm_stats,
            csv_root_folder=self.csv_root_folder,
            patches_root_folder=self.patches_root_folder,
        )
        self.test_dataset = ElevationStackDataset(
            split="tst",
            norm_stats=self.norm_stats,
            csv_root_folder=self.csv_root_folder,
            patches_root_folder=self.patches_root_folder,
        )

    @rank_zero_only
    def prepare_data(self) -> None:
        """
        Prepare data for training/validation/testing.

        Extends the base class functionality to handle elevation data processing:
        1. Check if data already exists
        2. Process each AOI: align, compute derivatives, stack, rasterize, tile
        3. Generate CSV files for training/inference
        4. Compute and save normalization statistics

        This method handles the full pipeline from raw elevation data to
        processed tiles ready for training.
        """
        # Check if data preparation is needed
        if self._data_already_exists():
            log.info(
                "[SKIP] All tiles and CSV already exist. "
                "Skipping full data preparation.",
            )
            self._load_or_compute_stats()
            return

        # Process each AOI
        for aoi_path in self.input_folders:
            self._process_aoi(aoi_path)

        # Generate CSV files
        log.info("Generating CSV...")
        generate_csv_from_tiles(
            root_output_folder=self.output_root,
            csv_tiling_path=self.csv_path,
            csv_inference_path=self.csv_infer_path,
            test_ratio=self.test_ratio,
            remove_empty_labels=True,
        )

        # Compute and save statistics
        self._compute_and_save_stats()

    def _data_already_exists(self) -> bool:
        """
        Check if all required data already exists.

        Returns:
            bool: True if all data exists, False otherwise.

        """
        # Check if CSV exists
        if not Path(self.csv_path).exists():
            return False

        # Check if all AOI tiles exist
        for aoi_path in self.input_folders:
            aoi_name = Path(aoi_path).name
            input_tiles = Path(self.output_root) / aoi_name / "tiles" / "inputs"
            label_tiles = Path(self.output_root) / aoi_name / "tiles" / "labels"

            if not input_tiles.is_dir() or not label_tiles.is_dir():
                return False

            input_tifs = list(input_tiles.glob("*.tif"))
            label_tifs = list(label_tiles.glob("*.tif"))
            if len(input_tifs) == 0 or len(label_tifs) == 0:
                return False

        return True

    def _load_or_compute_stats(self) -> None:
        """Load existing statistics or compute them from tiles."""
        stats_path = Path(self.output_root) / "stats.npy"
        log.info("Stats path: %s", stats_path)

        if stats_path.exists():
            stats = np.load(stats_path, allow_pickle=True).item()
            self.norm_stats["mean"] = stats["means"]
            self.norm_stats["std"] = stats["stds"]
            log.info("Loaded existing statistics")
        else:
            self._compute_and_save_stats()

    def _compute_and_save_stats(self) -> None:
        """Compute dataset statistics and save them."""
        # Find all input tile directories
        input_tile_dirs = list(Path(self.output_root).glob("*/tiles/inputs"))
        all_tile_paths = []
        for folder in input_tile_dirs:
            all_tile_paths.extend(str(p) for p in folder.glob("*.tif"))

        if not all_tile_paths:
            log.warning("No tiles found for statistics computation")
            return

        # Compute statistics
        means, stds = compute_dataset_stats_from_list(all_tile_paths)
        self.norm_stats["mean"] = means
        self.norm_stats["std"] = stds

        # Save statistics
        stats_path = Path(self.output_root) / "stats.npy"
        np.save(stats_path, {"means": means, "stds": stds})
        log.info("Computed and saved statistics to: %s", stats_path)

    def _process_aoi(self, aoi_path: str) -> None:
        """
        Process a single AOI through the full pipeline.

        Args:
            aoi_path: Path to the AOI folder containing raw data.

        """
        aoi_name = Path(aoi_path).name
        out_dir = Path(self.output_root) / aoi_name
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("Preparing data for AOI: %s", aoi_name)

        # Define paths
        dtm = Path(aoi_path) / "dtm.tif"
        dsm = Path(aoi_path) / "dsm.tif"
        intensity = Path(aoi_path) / "intensity.tif"
        labels_vector = Path(aoi_path) / "waterbodies.shp"

        # Step 1: Align inputs to DTM
        log.info("Aligning inputs to DTM")
        dsm_aligned = out_dir / "dsm_aligned.tif"
        intensity_aligned = out_dir / "intensity_aligned.tif"

        if not dsm_aligned.exists():
            log.info("Aligning DSM: %s", dsm_aligned)
            align_to_reference(str(dtm), str(dsm), str(dsm_aligned))
        else:
            log.info("Skipping DSM alignment (already exists)")

        if self.intensity and intensity.exists():
            if not intensity_aligned.exists():
                log.info("Aligning Intensity: %s", intensity_aligned)
                align_to_reference(str(dtm), str(intensity), str(intensity_aligned))
            else:
                log.info("Skipping Intensity alignment (already exists)")

        # Step 2: Compute derivatives
        twi_path = out_dir / "twi.tif"
        ndsm_path = out_dir / "ndsm.tif"

        if not twi_path.exists():
            log.info("Computing TWI: %s", twi_path)
            compute_twi_whitebox(str(dtm), str(twi_path))
        else:
            log.info("Skipping TWI (already exists at %s)", twi_path)

        if not ndsm_path.exists():
            log.info("Computing nDSM: %s", ndsm_path)
            compute_ndsm(str(dsm_aligned), str(dtm), str(ndsm_path))
        else:
            log.info("Skipping nDSM (already exists at %s)", ndsm_path)

        # Step 3: Stack inputs
        stack_path = out_dir / "stacked_inputs.tif"
        log.info("Stacking Inputs")

        stack_inputs = [str(twi_path), str(ndsm_path)]
        if self.intensity and intensity_aligned.exists():
            stack_inputs.append(str(intensity_aligned))
            log.info("Adding Intensity")

        log.info("Stacking %d bands: %s", len(stack_inputs), stack_inputs)
        stack_rasters(stack_inputs, str(stack_path))

        # Step 4: Rasterize labels
        label_raster = out_dir / "labels_aligned.tif"
        rasterize_labels_binary_aoi_mask(
            label_vector_path=str(labels_vector),
            aoi_vector_path=str(Path(aoi_path) / "aoi.shp"),
            reference_raster_path=str(dtm),
            output_path=str(label_raster),
            burn_value=1,
            fill_value=0,
            ignore_value=-1,
        )

        # Step 5: Tile
        log.info("Tiling...")
        tile_raster_pair(
            input_path=str(stack_path),
            label_path=str(label_raster),
            output_dir=str(out_dir / "tiles"),
            patch_size=self.patch_size[0],
            stride=self.stride,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """
        Create training dataloader.

        Returns:
            DataLoader for training data.

        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """
        Create validation dataloader.

        Returns:
            DataLoader for validation data.

        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """
        Create test dataloader.

        Returns:
            DataLoader for test data.

        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            shuffle=False,
        )

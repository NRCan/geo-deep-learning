"""Elevation Stack DataModule for water extraction."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightning.pytorch.utilities import rank_zero_only
from torch.utils.data import DataLoader

from geo_deep_learning.datamodules.csv_datamodule import CSVDataModule
from geo_deep_learning.utils.rasters import compute_dataset_stats_from_list

from .elevation_stack_dataset import ElevationStackDataset
from .prepare_inputs import (
    align_to_reference,
    compute_ndsm,
    compute_twi_whitebox,
    rasterize_labels_binary_aoi_mask,
    rasterize_valid_lidar_mask,
    stack_rasters,
)

log = logging.getLogger(__name__)


class ElevationStackDataModule(CSVDataModule):
    """
    DataModule handling the full elevation-stack preprocessing pipeline
    for water extraction.
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
        *,
        input_folders: list[str] | None = None,
        output_root: str = "",
        csv_path: str = "",
        csv_infer_path: str = "",
        include_intensity: bool = False,
        stride: int = 256,
        test_ratio: float = 0.2,
        valid_mask_min_ratio: float | None = 0.9,
        save_rejected_tiles: bool = False,
        regenerate_csv: bool = False,
        min_water_pixels: int = 1,
        test_only: bool = False,
    ) -> None:
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

        self.input_folders = input_folders or []
        self.output_root = output_root
        self.csv_path = csv_path
        self.csv_infer_path = csv_infer_path
        log.info(
            f"[DEBUG] ElevationStackDataModule __init__: csv_infer_path = {csv_infer_path}"
        )
        self.include_intensity = include_intensity
        self.stride = stride
        self.test_ratio = test_ratio
        self.valid_mask_min_ratio = valid_mask_min_ratio
        self.save_rejected_tiles = save_rejected_tiles
        self.regenerate_csv = regenerate_csv
        self.min_water_pixels = min_water_pixels
        self.test_only = test_only

        # Track if user provided custom stats (to avoid overwriting with stats.npy)
        self.user_provided_stats = mean is not None and std is not None
        
        # Slice user-provided stats to match include_intensity setting
        if self.user_provided_stats and not self.include_intensity and len(self.norm_stats["mean"]) > 2:
            log.info("Slicing user-provided stats to 2 channels (excluding intensity)")
            self.norm_stats["mean"] = self.norm_stats["mean"][:2]
            self.norm_stats["std"] = self.norm_stats["std"][:2]

    # ------------------------------------------------------------------
    # Setup datasets
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_vector_file(
        aoi_path: str | Path,
        base_filename: str,
        *,
        required: bool = True,
    ) -> Path | None:
        """Return the first existing vector file (gpkg or shp) for a base name."""
        aoi_dir = Path(aoi_path)
        candidate_paths = [
            aoi_dir / f"{base_filename}.gpkg",
            aoi_dir / f"{base_filename}.shp",
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                return candidate

        if required:
            candidates = ", ".join(str(path) for path in candidate_paths)
            msg = f"Missing required vector file. Looked for: {candidates}"
            raise FileNotFoundError(msg)

        log.info(
            "No optional vector file found for '%s'. Checked: %s",
            base_filename,
            ", ".join(str(path) for path in candidate_paths),
        )
        return None

        @staticmethod
        def _crop_raster_to_aoi(
            input_raster_path: str,
            output_raster_path: str,
            aoi_vector_path: str,
        ) -> None:
            """Crop a raster to the extent of an AOI polygon."""
            import fiona
            import rasterio
            import rasterio.mask

            log.info(
                "Cropping raster to AOI: %s → %s", input_raster_path, output_raster_path
            )

            # Read AOI geometries
            with fiona.open(aoi_vector_path, "r") as aoi_src:
                aoi_geoms = [feature["geometry"] for feature in aoi_src]

            # Crop raster
            with rasterio.open(input_raster_path) as src:
                out_image, out_transform = rasterio.mask.mask(
                    src,
                    aoi_geoms,
                    crop=True,
                    nodata=src.nodata,
                )
                # Update metadata
                out_meta = src.meta.copy()
                out_meta.update(
                    {
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "compress": "lzw",
                        "BIGTIFF": "YES",
                    }
                )

            # Write cropped raster
            with rasterio.open(output_raster_path, "w", **out_meta) as dst:
                dst.write(out_image)

            log.info("Cropped raster saved: %s", output_raster_path)

    @staticmethod
    def _crop_raster_to_aoi(
        input_raster_path: str,
        output_raster_path: str,
        aoi_vector_path: str,
    ) -> None:
        """Crop a raster to the extent of an AOI polygon."""
        import fiona
        import rasterio
        import rasterio.mask

        log.info(
            "Cropping raster to AOI: %s → %s", input_raster_path, output_raster_path
        )

        # Read AOI geometries
        with fiona.open(aoi_vector_path, "r") as aoi_src:
            aoi_geoms = [feature["geometry"] for feature in aoi_src]

        # Crop raster
        with rasterio.open(input_raster_path) as src:
            out_image, out_transform = rasterio.mask.mask(
                src,
                aoi_geoms,
                crop=True,
                nodata=src.nodata,
            )

            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "compress": "lzw",
                    "BIGTIFF": "YES",
                }
            )

        # Write cropped raster
        with rasterio.open(output_raster_path, "w", **out_meta) as dst:
            dst.write(out_image)

        log.info("Cropped raster saved: %s", output_raster_path)

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        # Validate stats configuration before creating datasets
        expected_channels = 3 if self.include_intensity else 2
        mean_channels = len(self.norm_stats["mean"])
        std_channels = len(self.norm_stats["std"])
        if mean_channels != expected_channels:
            error_msg = (
                f"Normalization stats mismatch: expected {expected_channels} channels "
                f"(include_intensity={self.include_intensity}) but mean has {mean_channels} values: "
                f"{self.norm_stats['mean']}"
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        if std_channels != expected_channels:
            error_msg = (
                f"Normalization stats mismatch: expected {expected_channels} channels "
                f"(include_intensity={self.include_intensity}) but std has {std_channels} values: "
                f"{self.norm_stats['std']}"
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        log.info(
            "Setting up datasets: include_intensity=%s, expected_channels=%d, "
            "mean=%s, std=%s",
            self.include_intensity,
            expected_channels,
            self.norm_stats["mean"],
            self.norm_stats["std"],
        )

        if self.test_only:
            self.test_dataset = ElevationStackDataset(
                split="inference",
                norm_stats=self.norm_stats,
                csv_root_folder=self.csv_root_folder,
                patches_root_folder=self.patches_root_folder,
                csv_path=self.csv_infer_path,
                csv_infer_path=self.csv_infer_path,
                include_intensity=self.include_intensity,
            )
        else:
            self.train_dataset = ElevationStackDataset(
                split="trn",
                norm_stats=self.norm_stats,
                csv_root_folder=self.csv_root_folder,
                patches_root_folder=self.patches_root_folder,
                csv_path=self.csv_path,
                include_intensity=self.include_intensity,
            )
            self.val_dataset = ElevationStackDataset(
                split="val",
                norm_stats=self.norm_stats,
                csv_root_folder=self.csv_root_folder,
                patches_root_folder=self.patches_root_folder,
                csv_path=self.csv_path,
                include_intensity=self.include_intensity,
            )
            self.test_dataset = ElevationStackDataset(
                split="tst",
                norm_stats=self.norm_stats,
                csv_root_folder=self.csv_root_folder,
                patches_root_folder=self.patches_root_folder,
                csv_path=self.csv_path,
                include_intensity=self.include_intensity,
            )

    # ------------------------------------------------------------------
    # Data preparation entry point
    # ------------------------------------------------------------------

    @rank_zero_only
    def prepare_data(self) -> None:
        """
        Prepare data for training.

        Semantics:
        - Tiling is skipped if tiles exist OR regenerate_csv=True
        - CSV generation is forced when regenerate_csv=True
        """
        if self._data_already_exists() and not self.regenerate_csv:
            log.info("[SKIP] Tiles already exist and CSV handling resolved.")
            self._load_or_compute_stats()
            return

        # -------------------------------
        # AOI processing (tiling)
        # -------------------------------
        if self.input_folders and not self.regenerate_csv:
            for aoi_path in self.input_folders:
                self._process_aoi(aoi_path)
        elif self.regenerate_csv:
            log.info("regenerate_csv=True → skipping AOI processing")

        # -------------------------------
        # CSV generation
        # -------------------------------
        log.info("[WARNING] Generating CSV files and STATS TEMPORARILY DISABLED")
        # log.info("Generating CSV files")
        # if self.test_only:
        #     # Only generate inference CSV
        #     generate_csv_from_tiles(
        #         root_output_folder=self.output_root,
        #         csv_tiling_path=self.csv_path,
        #         csv_inference_path=self.csv_infer_path,
        #         test_ratio=1.0,  # All tiles go to inference
        #         min_water_pixels=self.min_water_pixels,
        #     )
        #     log.info("Test only, no stats computation")
        # else:
        #     generate_csv_from_tiles(
        #         root_output_folder=self.output_root,
        #         csv_tiling_path=self.csv_path,
        #         csv_inference_path=self.csv_infer_path,
        #         test_ratio=self.test_ratio,
        #         min_water_pixels=self.min_water_pixels,
        #     )

        #     self._compute_and_save_stats()

    # ------------------------------------------------------------------
    # Existence checks
    # ------------------------------------------------------------------

    def _data_already_exists(self) -> bool:
        """
        Determine whether data preparation can be skipped.
        """
        if self.regenerate_csv:
            log.info("regenerate_csv=True → bypassing CSV existence check")
        elif not Path(self.csv_path).exists():
            return False

        if not self.input_folders:
            return True

        log.info(f"[DEBUG] self.input_folders = {self.input_folders}")

        for aoi_path in self.input_folders:
            aoi_name = Path(aoi_path).name
            tiles_root = Path(self.output_root) / aoi_name / "tiles"

            log.info(f"[DEBUG] aoi_name = {aoi_name}")
            log.info(f"[DEBUG] tiles_root = {tiles_root}")

            if not (tiles_root / "inputs").is_dir():
                return False
            if not (tiles_root / "labels").is_dir():
                return False

            if not any((tiles_root / "inputs").glob("*.tif")):
                return False
            if not any((tiles_root / "labels").glob("*.tif")):
                return False

        return True

    # ------------------------------------------------------------------
    # Stats handling
    # ------------------------------------------------------------------

    def _load_or_compute_stats(self) -> None:
        # If user provided stats in config, don't load from stats.npy
        if self.user_provided_stats:
            log.info("Using user-provided statistics from config")
            log.info(
                "[DEBUG] User stats (include_intensity=%s): mean=%s, std=%s",
                self.include_intensity,
                self.norm_stats["mean"],
                self.norm_stats["std"],
            )
            return
        
        stats_path = Path(self.output_root) / "stats.npy"
        log.info("[DEBUG] loaded stats path: %s", stats_path)

        if stats_path.exists():
            stats = np.load(stats_path, allow_pickle=True).item()
            self.norm_stats["mean"] = stats["means"]
            self.norm_stats["std"] = stats["stds"]
            
            # Slice stats to match include_intensity setting
            # Always use first 2 channels (TWI, nDSM) if intensity is not included
            if not self.include_intensity and len(self.norm_stats["mean"]) > 2:
                log.info("Slicing stats to 2 channels (excluding intensity)")
                self.norm_stats["mean"] = self.norm_stats["mean"][:2]
                self.norm_stats["std"] = self.norm_stats["std"][:2]
            
            log.info("Loaded existing statistics from stats.npy")
            log.info(
                "[DEBUG] stats (include_intensity=%s): mean=%s, std=%s",
                self.include_intensity,
                self.norm_stats["mean"],
                self.norm_stats["std"],
            )
        else:
            self._compute_and_save_stats()

    def _compute_and_save_stats(self) -> None:
        """
        Compute normalization statistics strictly from tiles listed in the CSV.
        This guarantees consistency with training data.
        """
        from pathlib import Path

        if not Path(self.csv_path).exists():
            log.warning("CSV not found — cannot compute dataset statistics")
            return

        log.info("Computing dataset statistics from CSV tiles")

        df = pd.read_csv(self.csv_path)

        if "tif" not in df.columns:
            raise ValueError("CSV must contain a 'tif' column with tile paths")

        tile_paths = df["tif"].dropna().astype(str).unique().tolist()

        if not tile_paths:
            log.warning("No tiles found in CSV for statistics computation")
            return

        means, stds = compute_dataset_stats_from_list(tile_paths)

        self.norm_stats["mean"] = means
        self.norm_stats["std"] = stds

        stats_path = Path(self.output_root) / "stats.npy"
        np.save(stats_path, {"means": means, "stds": stds})

        log.info(
            "Saved normalization stats from %d tiles to %s",
            len(tile_paths),
            stats_path,
        )

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
        if not self.test_only:
            labels_vector = self._resolve_vector_file(aoi_path, "waterbodies")

        valid_mask_vector = self._resolve_vector_file(
            aoi_path,
            "valid_lidar_mask",
            required=False,
        )

        # Step 1: Align inputs to DTM
        log.info("Aligning inputs to DTM")
        dsm_aligned = out_dir / "dsm_aligned.tif"
        intensity_aligned = out_dir / "intensity_aligned.tif"

        if not dsm_aligned.exists():
            log.info("Aligning DSM: %s", dsm_aligned)
            align_to_reference(str(dtm), str(dsm), str(dsm_aligned))
        else:
            log.info("Skipping DSM alignment (already exists)")

        if self.include_intensity and intensity.exists():
            intensity_temp = out_dir / "intensity_temp.tif"
            needs_alignment = not intensity_aligned.exists()

            if needs_alignment:
                log.info("Aligning Intensity: %s", intensity_temp)
                # Patch: ensure BIGTIFF=YES for large files
                align_to_reference(
                    str(dtm),
                    str(intensity),
                    str(intensity_temp),
                )
                # Crop aligned intensity to AOI boundary
                aoi_vector = self._resolve_vector_file(aoi_path, "aoi")
                if aoi_vector:
                    log.info(
                        "Cropping intensity to AOI boundary: %s", intensity_aligned
                    )
                    self._crop_raster_to_aoi(
                        str(intensity_temp),
                        str(intensity_aligned),
                        str(aoi_vector),
                    )
                    # Remove temp file
                    intensity_temp.unlink()
                else:
                    # No AOI vector, just rename temp to final
                    intensity_temp.rename(intensity_aligned)
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
        # Always use same filename - channel selection happens at load time
        stack_path = out_dir / "stacked_inputs.tif"

        if not stack_path.exists():
            log.info("Stacking Inputs")

            stack_inputs = [str(twi_path), str(ndsm_path)]
            if self.include_intensity and intensity_aligned.exists():
                stack_inputs.append(str(intensity_aligned))
                log.info("Adding Intensity")

            log.info("Stacking %d bands: %s", len(stack_inputs), stack_inputs)
            stack_rasters(stack_inputs, str(stack_path))
        else:
            log.info("Skipping stacking (already exists at %s)", stack_path)

        # Optional: rasterize valid LiDAR mask
        valid_mask_raster = out_dir / "valid_mask.tif"
        if valid_mask_vector is not None:
            if not valid_mask_raster.exists():
                log.info("Rasterizing valid LiDAR mask: %s", valid_mask_vector)
                rasterize_valid_lidar_mask(
                    str(valid_mask_vector),
                    str(dtm),
                    str(valid_mask_raster),
                )
            else:
                log.info("Skipping valid mask rasterization (already exists)")

        label_raster = out_dir / "labels_aligned.tif"

        if not self.test_only:
            # Step 4: Rasterize labels
            aoi_vector = self._resolve_vector_file(aoi_path, "aoi")
            rasterize_labels_binary_aoi_mask(
                label_vector_path=str(labels_vector),
                aoi_vector_path=str(aoi_vector),
                reference_raster_path=str(dtm),
                output_path=str(label_raster),
                burn_value=1,
                fill_value=0,
                ignore_value=-1,
            )
        # Create a dummy label raster (all zeros, same shape as stack_path)
        elif not label_raster.exists():
            log.info("Creating dummy label because test_only mode")
            import rasterio

            with rasterio.open(str(stack_path)) as src:
                profile = src.profile.copy()
                profile.update({"count": 1, "dtype": "int16", "nodata": -1})
                data = np.full(
                    (src.height, src.width),
                    fill_value=1,  # ignore_index
                    dtype=np.int16,
                )
                with rasterio.open(str(label_raster), "w", **profile) as dst:
                    dst.write(data, 1)
        # Quick debug
        import sys

        log.info("TIME TAG")
        sys.exit("Exiting early, data preparation done")

        log.info("[DEBUG] TILING IS COMMENTED OUT TEMPORARILY")
        # # Step 5: Tile
        # log.info("Tiling...")
        # tile_raster_pair(
        #     input_path=str(stack_path),
        #     label_path=str(label_raster),
        #     output_dir=str(out_dir / "tiles"),
        #     patch_size=self.patch_size[0],
        #     stride=self.stride,
        #     valid_mask_path=str(valid_mask_raster) if valid_mask_raster.exists() else None,
        #     valid_mask_min_ratio=self.valid_mask_min_ratio,
        #     save_rejected_tiles=self.save_rejected_tiles,
        # )

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

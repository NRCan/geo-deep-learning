"""Elevation Stack DataModule for water extraction."""

import logging
import shutil
from pathlib import Path
from typing import Any

import fiona
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from lightning.pytorch.utilities import rank_zero_only
from torch.utils.data import DataLoader

from geo_deep_learning.datamodules.csv_datamodule import CSVDataModule
from geo_deep_learning.tools.water_extraction.elevation_stack_dataset import (
    ElevationStackDataset,
)
from geo_deep_learning.tools.water_extraction.prepare_inputs import (
    align_to_reference,
    compute_ndsm,
    compute_twi_whitebox,
    generate_csv_from_tiles,
    rasterize_labels_binary_aoi_mask,
    rasterize_valid_lidar_mask,
    stack_rasters,
    tile_raster_pair,
)
from geo_deep_learning.tools.water_extraction.seam_correction import correct_seams
from geo_deep_learning.utils.rasters import compute_dataset_stats_from_list

_NUM_BASE_CHANNELS = 2

log = logging.getLogger(__name__)


class ElevationStackDataModule(CSVDataModule):
    """
    DataModule handling the full elevation-stack preprocessing pipeline.

    Handles water extraction preprocessing end-to-end.
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
        extra_input_rasters: list[str] | None = None,
        selected_channel_indices: list[int] | None = None,
        stacked_inputs_filename: str = "stacked_inputs.tif",
        tiles_dirname: str = "tiles",
        tile_stats_filename: str = "tile_stats.csv",
        stats_filename: str = "stats.npy",
        stride: int = 256,
        test_ratio: float = 0.2,
        valid_mask_min_ratio: float | None = 0.9,
        save_rejected_tiles: bool = False,
        regenerate_csv: bool = False,
        min_water_pixels: int = 1,
        test_only: bool = False,
        workflow: str = "training",
        project_extents_path: str | None = None,
        seam_gaussian_sigma: float = 1.5,
    ) -> None:
        """Initialise the datamodule and propagate settings to the parent."""
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
            "[DEBUG] ElevationStackDataModule __init__: csv_infer_path = %s",
            csv_infer_path,
        )
        self.include_intensity = include_intensity
        self.extra_input_rasters = extra_input_rasters or []
        self.selected_channel_indices = selected_channel_indices
        self.stacked_inputs_filename = stacked_inputs_filename
        self.tiles_dirname = tiles_dirname
        self.tile_stats_filename = tile_stats_filename
        self.stats_filename = stats_filename
        self.stride = stride
        self.test_ratio = test_ratio
        self.valid_mask_min_ratio = valid_mask_min_ratio
        self.save_rejected_tiles = save_rejected_tiles
        self.regenerate_csv = regenerate_csv
        self.min_water_pixels = min_water_pixels
        self.test_only = test_only
        self.workflow = workflow
        self.project_extents_path = project_extents_path
        self.seam_gaussian_sigma = seam_gaussian_sigma

        self._validate_workflow()
        self._validate_workflow_paths()

        # Track if user provided custom stats (to avoid overwriting with stats.npy)
        self.user_provided_stats = mean is not None and std is not None

        if self.user_provided_stats:
            self._slice_norm_stats_to_selected_channels(
                source="user-provided config",
            )

    def _validate_workflow(self) -> None:
        """Validate the workflow mode used to organize dataset paths."""
        if self.workflow not in {"training", "inference"}:
            msg = (
                "workflow must be either 'training' or 'inference', "
                f"got: {self.workflow}"
            )
            raise ValueError(msg)

    def _validate_workflow_paths(self) -> None:
        """
        Validate that configured paths match the workflow layout.

        Expected layout:
          data/<workflow>/raw/<aoi>/
          data/<workflow>/preprocessed/<aoi>/
        """
        if not self.input_folders:
            return

        expected_raw_segment = f"/{self.workflow}/raw/"
        expected_preprocessed_segment = f"/{self.workflow}/preprocessed"

        for input_folder in self.input_folders:
            input_path = Path(input_folder).resolve().as_posix()
            if expected_raw_segment not in input_path:
                msg = (
                    f"Input folder does not match workflow='{self.workflow}' layout: "
                    f"{input_folder}. Expected path containing "
                    f"'{expected_raw_segment}'."
                )
                raise ValueError(msg)

        if self.output_root:
            output_path = Path(self.output_root).resolve().as_posix()
            if expected_preprocessed_segment not in output_path:
                msg = (
                    f"output_root does not match workflow='{self.workflow}' layout: "
                    f"{self.output_root}. Expected path containing "
                    f"'{expected_preprocessed_segment}'."
                )
                raise ValueError(msg)

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

    def _available_num_channels(self) -> int:
        """Return the number of channels available in the full stack."""
        return (
            _NUM_BASE_CHANNELS
            + int(self.include_intensity)
            + len(self.extra_input_rasters)
        )

    def _resolved_channel_indices(self) -> list[int]:
        """Return the selected channel indices in stack order."""
        available_channels = self._available_num_channels()
        if self.selected_channel_indices is None:
            return list(range(available_channels))

        invalid_indices = [
            idx
            for idx in self.selected_channel_indices
            if idx < 0 or idx >= available_channels
        ]
        if invalid_indices:
            msg = (
                "selected_channel_indices contains invalid indices "
                f"{invalid_indices} for {available_channels} available channels"
            )
            raise ValueError(msg)

        if len(set(self.selected_channel_indices)) != len(self.selected_channel_indices):
            msg = (
                "selected_channel_indices must not contain duplicates: "
                f"{self.selected_channel_indices}"
            )
            raise ValueError(msg)

        return self.selected_channel_indices

    def _expected_num_channels(self) -> int:
        """Return the configured number of input channels."""
        return len(self._resolved_channel_indices())

    def _slice_norm_stats_to_selected_channels(self, *, source: str) -> None:
        """Slice normalization stats to the selected channels, if needed."""
        selected_indices = self._resolved_channel_indices()
        if len(self.norm_stats["mean"]) == len(selected_indices):
            return

        if len(self.norm_stats["mean"]) < max(selected_indices) + 1:
            msg = (
                f"Normalization stats from {source} do not cover requested channels "
                f"{selected_indices}. mean has {len(self.norm_stats['mean'])} values, "
                f"std has {len(self.norm_stats['std'])} values."
            )
            raise ValueError(msg)

        log.info(
            "Slicing normalization stats from %s using channel indices %s",
            source,
            selected_indices,
        )
        self.norm_stats["mean"] = [self.norm_stats["mean"][idx] for idx in selected_indices]
        self.norm_stats["std"] = [self.norm_stats["std"][idx] for idx in selected_indices]

    @staticmethod
    def _crop_raster_to_aoi(
        input_raster_path: str,
        output_raster_path: str,
        aoi_vector_path: str,
    ) -> None:
        """Crop a raster to the extent of an AOI polygon."""
        log.info(
            "Cropping raster to AOI: %s → %s",
            input_raster_path,
            output_raster_path,
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
                },
            )

        # Write cropped raster
        with rasterio.open(output_raster_path, "w", **out_meta) as dst:
            dst.write(out_image)

        log.info("Cropped raster saved: %s", output_raster_path)

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Validate normalization stats and create train/val/test datasets."""
        # Validate stats configuration before creating datasets
        expected_channels = self._expected_num_channels()
        mean_channels = len(self.norm_stats["mean"])
        std_channels = len(self.norm_stats["std"])
        if mean_channels != expected_channels:
            error_msg = (
                f"Normalization stats mismatch: expected {expected_channels} channels "
                f"(include_intensity={self.include_intensity}, "
                f"extra_input_rasters={self.extra_input_rasters}) but mean has "
                f"{mean_channels} values: {self.norm_stats['mean']}"
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        if std_channels != expected_channels:
            error_msg = (
                f"Normalization stats mismatch: expected {expected_channels} channels "
                f"(include_intensity={self.include_intensity}, "
                f"extra_input_rasters={self.extra_input_rasters}) but std has "
                f"{std_channels} values: {self.norm_stats['std']}"
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        log.info("=" * 80)
        log.info("DATAMODULE SETUP DIAGNOSTICS")
        log.info("=" * 80)
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
                extra_input_rasters=self.extra_input_rasters,
                selected_channel_indices=self.selected_channel_indices,
            )
            log.info("Test-only mode: created inference dataset with %d samples", len(self.test_dataset))
        else:
            self.train_dataset = ElevationStackDataset(
                split="trn",
                norm_stats=self.norm_stats,
                csv_root_folder=self.csv_root_folder,
                patches_root_folder=self.patches_root_folder,
                csv_path=self.csv_path,
                include_intensity=self.include_intensity,
                extra_input_rasters=self.extra_input_rasters,
                selected_channel_indices=self.selected_channel_indices,
            )
            self.val_dataset = ElevationStackDataset(
                split="val",
                norm_stats=self.norm_stats,
                csv_root_folder=self.csv_root_folder,
                patches_root_folder=self.patches_root_folder,
                csv_path=self.csv_path,
                include_intensity=self.include_intensity,
                extra_input_rasters=self.extra_input_rasters,
                selected_channel_indices=self.selected_channel_indices,
            )
            self.test_dataset = ElevationStackDataset(
                split="tst",
                norm_stats=self.norm_stats,
                csv_root_folder=self.csv_root_folder,
                patches_root_folder=self.patches_root_folder,
                csv_path=self.csv_path,
                include_intensity=self.include_intensity,
                extra_input_rasters=self.extra_input_rasters,
                selected_channel_indices=self.selected_channel_indices,
            )
            
            log.info("Training mode: created datasets")
            log.info("  Train: %d samples", len(self.train_dataset))
            log.info("  Val: %d samples", len(self.val_dataset))
            log.info("  Test: %d samples", len(self.test_dataset))
            
            # === DIAGNOSTIC: Check unique mask values across splits ===
            self._log_unique_mask_values()
        
        log.info("=" * 80)
    
    def _log_unique_mask_values(self) -> None:
        """Log unique mask values for train/val/test splits to verify preprocessing consistency."""
        import torch
        
        log.info("=" * 80)
        log.info("MASK VALUE VERIFICATION")
        log.info("=" * 80)
        
        for split_name, dataset in [("train", self.train_dataset), ("val", self.val_dataset), ("test", self.test_dataset)]:
            # Sample first few masks to check unique values
            unique_values_set = set()
            samples_to_check = min(5, len(dataset))
            
            for i in range(samples_to_check):
                try:
                    sample = dataset[i]
                    mask = sample["mask"]
                    unique_vals = torch.unique(mask).tolist()
                    unique_values_set.update(unique_vals)
                except Exception as e:
                    log.warning("Error loading %s sample %d: %s", split_name, i, e)
            
            log.info("%s split - Unique mask values from first %d samples: %s", 
                    split_name.upper(), samples_to_check, sorted(unique_values_set))
        
        log.info("=" * 80)

    # ------------------------------------------------------------------
    # Data preparation entry point
    # ------------------------------------------------------------------

    @rank_zero_only
    def prepare_data(self) -> None:
        """
        Prepare data for training or inference.

        Behavior depends on test_only flag:

        test_only=False (training):
          - Preprocesses: alignment, TWI, nDSM, stacking
          - Rasterizes labels with AOI masking
          - Tiles inputs and labels
          - Generates CSV with train/val/test splits
          - Computes normalization statistics

        test_only=True (inference preprocessing):
          - Preprocesses: alignment, TWI, nDSM, stacking
          - Skips: label rasterization, tiling, CSV, stats
          - Output: stacked_inputs.tif ready for inference

        Semantics:
        - In inference mode, skips if preprocessed stacks already exist
        - In training mode with input_folders, regenerates stack/tiles/CSV each run
        """
        if self._data_already_exists() and not self.regenerate_csv:
            if self.test_only:
                log.info("[SKIP] Preprocessed stacks already exist.")
            else:
                log.info("[SKIP] Tiles already exist and CSV handling resolved.")
                self._load_or_compute_stats()
            return

        if not self.test_only:
            for path_str in [self.csv_path, self.csv_infer_path]:
                path = Path(path_str)
                if path.exists():
                    path.unlink()
                    log.info("Removed stale CSV: %s", path)

            stats_path = Path(self.output_root) / self.stats_filename
            if stats_path.exists():
                stats_path.unlink()
                log.info("Removed stale stats cache: %s", stats_path)

        # -------------------------------
        # AOI processing (preprocessing + tiling for training)
        # -------------------------------
        if self.input_folders and not self.regenerate_csv:
            for aoi_path in self.input_folders:
                self._process_aoi(aoi_path)
        elif self.regenerate_csv:
            log.info("regenerate_csv=True → skipping AOI processing")

        # -------------------------------
        # CSV generation and stats (training only)
        # -------------------------------
        if not self.test_only:
            log.info("Generating CSV files for training")
            generate_csv_from_tiles(
                root_output_folder=self.output_root,
                csv_tiling_path=self.csv_path,
                csv_inference_path=self.csv_infer_path,
                aoi_names=[Path(aoi_path).name for aoi_path in self.input_folders]
                if self.input_folders
                else None,
                test_ratio=self.test_ratio,
                min_water_pixels=self.min_water_pixels,
                tiles_folder_name=self.tiles_dirname,
                tile_stats_filename=self.tile_stats_filename,
            )
            log.info("Computing normalization statistics")
            self._compute_and_save_stats()
        else:
            log.info("test_only=True → skipping CSV generation and stats computation")

    # ------------------------------------------------------------------
    # Existence checks
    # ------------------------------------------------------------------

    def _data_already_exists(self) -> bool:  # noqa: PLR0911, C901
        """Determine whether data preparation can be skipped."""
        # test_only mode: check if preprocessed stacks exist
        if self.test_only:
            if not self.input_folders:
                return True

            log.info("test_only=True → checking for preprocessed stacks only")
            for aoi_path in self.input_folders:
                aoi_name = Path(aoi_path).name
                stacked_path = (
                    Path(self.output_root) / aoi_name / self.stacked_inputs_filename
                )

                if not stacked_path.exists():
                    log.info("Stacked inputs not found: %s", stacked_path)
                    return False

            log.info("All preprocessed stacks exist → skipping preprocessing")
            return True

        if self.input_folders:
            log.info("Training mode with input_folders → forcing stack/tile/CSV regeneration")
            return False

        # Training mode: check for tiles and CSV
        if self.regenerate_csv:
            log.info("regenerate_csv=True → bypassing CSV existence check")
        elif not Path(self.csv_path).exists():
            return False

        if not self.input_folders:
            return True

        log.info("[DEBUG] self.input_folders = %s", self.input_folders)

        for aoi_path in self.input_folders:
            aoi_name = Path(aoi_path).name
            tiles_root = Path(self.output_root) / aoi_name / self.tiles_dirname

            log.info("[DEBUG] aoi_name = %s", aoi_name)
            log.info("[DEBUG] tiles_root = %s", tiles_root)

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

        stats_path = Path(self.output_root) / self.stats_filename
        if stats_path.exists():
            log.info("Ignoring existing stats cache and recomputing from current CSV: %s", stats_path)
        self._compute_and_save_stats()

    def _compute_and_save_stats(self) -> None:
        """
        Compute normalization statistics strictly from tiles listed in the CSV.

        This guarantees consistency with training data.
        """
        if not Path(self.csv_path).exists():
            log.warning("CSV not found — cannot compute dataset statistics")
            return

        log.info("Computing dataset statistics from CSV tiles")

        tiles_df = pd.read_csv(self.csv_path)

        if "tif" not in tiles_df.columns:
            msg = "CSV must contain a 'tif' column with tile paths"
            raise ValueError(msg)

        tile_paths = tiles_df["tif"].dropna().astype(str).unique().tolist()

        if not tile_paths:
            log.warning("No tiles found in CSV for statistics computation")
            return

        means, stds = compute_dataset_stats_from_list(tile_paths)

        self.norm_stats["mean"] = means
        self.norm_stats["std"] = stds
        self._slice_norm_stats_to_selected_channels(
            source="computed CSV statistics",
        )

        log.info(
            "Computed normalization stats from %d tiles (not cached to %s)",
            len(tile_paths),
            Path(self.output_root) / self.stats_filename,
        )

    def _reset_training_outputs_for_aoi(self, out_dir: Path) -> None:
        """Remove derived training artifacts that must be regenerated on rerun."""
        stack_path = out_dir / self.stacked_inputs_filename
        tiles_root = out_dir / self.tiles_dirname
        tile_stats_path = out_dir / self.tile_stats_filename

        for path in [stack_path, tile_stats_path]:
            if path.exists():
                path.unlink()
                log.info("Removed stale training artifact: %s", path)

        if tiles_root.exists():
            shutil.rmtree(tiles_root)
            log.info("Removed stale tiles directory: %s", tiles_root)

    def _process_aoi(self, aoi_path: str) -> None:  # noqa: C901, PLR0912, PLR0915
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
        extra_raster_inputs = [Path(aoi_path) / raster for raster in self.extra_input_rasters]
        if not self.test_only:
            labels_vector = self._resolve_vector_file(aoi_path, "waterbodies")

        valid_mask_vector = self._resolve_vector_file(
            aoi_path,
            "valid_lidar_mask",
            required=False,
        )

        if not self.test_only:
            self._reset_training_outputs_for_aoi(out_dir)

        # Step 1: Align inputs to DTM
        log.info("Aligning inputs to DTM")
        dsm_aligned = out_dir / "dsm_aligned.tif"
        intensity_aligned = out_dir / "intensity_aligned.tif"
        extra_rasters_aligned = {
            raw_path: out_dir / f"{raw_path.stem}_aligned.tif"
            for raw_path in extra_raster_inputs
        }

        if not dsm_aligned.exists():
            log.info("Aligning DSM: %s", dsm_aligned)
            align_to_reference(str(dtm), str(dsm), str(dsm_aligned))
        else:
            log.info("Skipping DSM alignment (already exists)")

        if self.include_intensity and intensity.exists():
            if not intensity_aligned.exists():
                log.info("Aligning Intensity: %s", intensity_aligned)
                # Align to DTM without cropping
                # Cropping should only happen after inference to remove edge artifacts
                align_to_reference(
                    str(dtm),
                    str(intensity),
                    str(intensity_aligned),
                )
            else:
                log.info("Skipping Intensity alignment (already exists)")

        for raw_path, aligned_path in extra_rasters_aligned.items():
            if not raw_path.exists():
                msg = f"Configured extra input raster not found: {raw_path}"
                raise FileNotFoundError(msg)

            if not aligned_path.exists():
                log.info("Aligning extra raster '%s': %s", raw_path.name, aligned_path)
                align_to_reference(str(dtm), str(raw_path), str(aligned_path))
            else:
                log.info(
                    "Skipping extra raster alignment for '%s' (already exists)",
                    raw_path.name,
                )

        # Seam correction on DTM and DSM
        # Skipped when project_extents_path is not configured.
        log.info(
            "[DEBUG] project_extents_path = %r (type=%s)",
            self.project_extents_path,
            type(self.project_extents_path).__name__,
        )
        if self.project_extents_path is not None and self.project_extents_path != "":
            log.info(
                "Seam correction will be applied using: %s",
                self.project_extents_path,
            )
            dtm_corrected = out_dir / "dtm_corrected.tif"
            dsm_corrected = out_dir / "dsm_corrected.tif"

            if not dtm_corrected.exists():
                log.info("Applying seam correction to DTM: %s", dtm_corrected)
                correct_seams(
                    input_path=str(dtm),
                    output_path=str(dtm_corrected),
                    project_extents_path=self.project_extents_path,
                    gaussian_sigma=self.seam_gaussian_sigma,
                )
            else:
                log.info("Skipping DTM seam correction (already exists)")

            if not dsm_corrected.exists():
                log.info("Applying seam correction to DSM: %s", dsm_corrected)
                correct_seams(
                    input_path=str(dsm_aligned),
                    output_path=str(dsm_corrected),
                    project_extents_path=self.project_extents_path,
                    gaussian_sigma=self.seam_gaussian_sigma,
                )
            else:
                log.info("Skipping DSM seam correction (already exists)")

            dtm_for_deriv = dtm_corrected
            dsm_for_deriv = dsm_corrected
        else:
            log.info("Skipping seam correction (project_extents_path not configured)")
            dtm_for_deriv = dtm
            dsm_for_deriv = dsm_aligned

        # Step 2: Compute derivatives
        twi_path = out_dir / "twi.tif"
        ndsm_path = out_dir / "ndsm.tif"

        if not twi_path.exists():
            log.info("Computing TWI: %s", twi_path)
            compute_twi_whitebox(str(dtm_for_deriv), str(twi_path))
        else:
            log.info("Skipping TWI (already exists at %s)", twi_path)

        if not ndsm_path.exists():
            log.info("Computing nDSM: %s", ndsm_path)
            compute_ndsm(str(dsm_for_deriv), str(dtm_for_deriv), str(ndsm_path))
        else:
            log.info("Skipping nDSM (already exists at %s)", ndsm_path)

        # Step 3: Stack inputs
        stack_path = out_dir / self.stacked_inputs_filename

        log.info("Stacking Inputs")

        stack_inputs = [str(twi_path), str(ndsm_path)]
        if self.include_intensity and intensity_aligned.exists():
            stack_inputs.append(str(intensity_aligned))
            log.info("Adding Intensity")
        for raw_path in extra_raster_inputs:
            aligned_path = extra_rasters_aligned[raw_path]
            stack_inputs.append(str(aligned_path))
            log.info("Adding extra raster: %s", raw_path.name)

        selected_indices = self._resolved_channel_indices()
        if selected_indices != list(range(len(stack_inputs))):
            stack_inputs = [stack_inputs[idx] for idx in selected_indices]
            log.info(
                "Selecting stack channels %s before writing stacked inputs",
                selected_indices,
            )

        log.info("Stacking %d bands: %s", len(stack_inputs), stack_inputs)
        stack_rasters(stack_inputs, str(stack_path))

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

        # Step 4 & 5: Label rasterization and tiling (training only)
        if not self.test_only:
            label_raster = out_dir / "labels_aligned.tif"

            # Rasterize labels
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

            # Tile for training
            log.info("Tiling for training...")
            tile_raster_pair(
                input_path=str(stack_path),
                label_path=str(label_raster),
                output_dir=str(out_dir / self.tiles_dirname),
                patch_size=self.patch_size[0],
                stride=self.stride,
                valid_mask_path=(
                    str(valid_mask_raster) if valid_mask_raster.exists() else None
                ),
                valid_mask_min_ratio=self.valid_mask_min_ratio,
                save_rejected_tiles=self.save_rejected_tiles,
                tile_stats_filename=self.tile_stats_filename,
            )
        else:
            log.info("test_only=True → skipping label rasterization and tiling")

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

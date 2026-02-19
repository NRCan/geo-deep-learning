"""
End-to-end inference script for water extraction from elevation data.

This script performs the complete inference pipeline:
1. Preprocesses raw LiDAR data (alignment, nDSM, TWI, stacking)
2. Runs sliding-window inference with overlap handling
3. Generates georeferenced prediction raster
4. Exports vectorized waterbody polygons

Usage:
    python -m geo_deep_learning.tools.water_extraction.inference \
        --checkpoint path/to/model.ckpt \
        --data_folder path/to/aoi \
        --output_folder path/to/outputs \
        --mean 0.0 0.0 0.0 \
        --std 1.0 1.0 1.0
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import fiona
import numpy as np
import rasterio
import torch
from rasterio.features import rasterize, shapes
from rasterio.windows import Window
from shapely.geometry import mapping
from shapely.geometry import shape as shapely_shape
from tqdm import tqdm

from geo_deep_learning.tools.water_extraction.prepare_inputs import (
    compute_ndsm,
    compute_twi_whitebox,
    stack_rasters,
)
from geo_deep_learning.tools.water_extraction.segmentation_task import (
    WaterExtractionSegmentation,
)
from geo_deep_learning.utils.rasters import align_to_reference

# from geo_deep_learning.utils.tensors import standardization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


# =============================================================================
# 1. PREPROCESSING
# =============================================================================


def preprocess_aoi(
    data_folder: str,
    output_folder: str,
    *,
    nodata_val: float = -32767,
    include_intensity: bool = True,
) -> str:
    """
    Preprocess raw LiDAR data for inference.

    Applies the same preprocessing pipeline as training:
    - Aligns DSM and intensity to DTM grid
    - Computes nDSM (DSM - DTM)
    - Computes TWI (Topographic Wetness Index)
    - Stacks bands into a single multi-band raster

    Args:
        data_folder: Path to folder containing dtm.tif, dsm.tif, intensity.tif
        output_folder: Where to save intermediate and final outputs
        nodata_val: NoData value for output rasters
        include_intensity: Whether to include intensity band in stack

    Returns:
        Path to stacked multi-band input raster

    Raises:
        FileNotFoundError: If required input files are missing

    """
    data_path = Path(data_folder).resolve()
    aoi_name = data_path.name

    # If user passes a parent output directory, we nest AOI_processed inside it
    base_output = Path(output_folder).resolve()
    output_path = base_output / f"preprocessed_{aoi_name}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Check required inputs
    dtm_path = data_path / "dtm.tif"
    dsm_path = data_path / "dsm.tif"

    if not dtm_path.exists():
        msg = f"DTM not found: {dtm_path}"
        raise FileNotFoundError(msg)
    if not dsm_path.exists():
        msg = f"DSM not found: {dsm_path}"
        raise FileNotFoundError(msg)

    log.info("=" * 80)
    log.info("PREPROCESSING: %s", data_folder)
    log.info("=" * 80)

    # Aligned rasters
    dsm_aligned = output_path / "dsm_aligned.tif"
    intensity_aligned = output_path / "intensity_aligned.tif"

    # Align DSM to DTM
    if not dsm_aligned.exists():
        log.info("Aligning DSM to DTM grid...")
        align_to_reference(
            reference_path=str(dtm_path),
            input_path=str(dsm_path),
            output_path=str(dsm_aligned),
            nodata_val=nodata_val,
        )
    else:
        log.info("Skipping DSM alignment (already exists)")

    # Optionally align intensity
    intensity_path = data_path / "intensity.tif"
    has_intensity = False
    if include_intensity and intensity_path.exists():
        if not intensity_aligned.exists():
            log.info("Aligning intensity to DTM grid...")
            align_to_reference(
                reference_path=str(dtm_path),
                input_path=str(intensity_path),
                output_path=str(intensity_aligned),
                nodata_val=nodata_val,
            )
        else:
            log.info("Skipping intensity alignment (already exists)")
        has_intensity = True
    elif include_intensity:
        log.warning(
            "Intensity file not found: %s (proceeding without intensity)",
            intensity_path,
        )

    # Compute nDSM
    ndsm_path = output_path / "ndsm.tif"
    if not ndsm_path.exists():
        log.info("Computing nDSM...")
        compute_ndsm(
            dsm_path=str(dsm_aligned),
            dtm_path=str(dtm_path),
            output_path=str(ndsm_path),
            nodata_val=nodata_val,
        )
    else:
        log.info("Skipping nDSM computation (already exists)")

    # Compute TWI
    twi_path = output_path / "twi.tif"
    if not twi_path.exists():
        log.info("Computing TWI...")
        compute_twi_whitebox(
            dtm_path=str(dtm_path),
            twi_output_path=str(twi_path),
            keep_intermediate=True,
        )
    else:
        log.info("Skipping TWI computation (already exists)")

    # Stack bands (order: TWI, nDSM, [intensity])
    stacked_path = output_path / "stacked_inputs.tif"
    if not stacked_path.exists():
        log.info("Stacking bands...")
        stack_inputs = [str(twi_path), str(ndsm_path)]
        if has_intensity:
            stack_inputs.append(str(intensity_aligned))
            log.info("Band order: [TWI, nDSM, Intensity]")
        else:
            log.info("Band order: [TWI, nDSM]")

        stack_rasters(
            raster_paths=stack_inputs,
            output_path=str(stacked_path),
            nodata_val=nodata_val,
        )
    else:
        log.info("Skipping stacking (already exists)")

    log.info("Preprocessing complete: %s", stacked_path)
    return str(stacked_path)


# =============================================================================
# 2. MODEL LOADING
# =============================================================================


# def load_model(checkpoint_path: str, *, device: str = "cuda") -> WaterExtractionSegmentation:
#     """
#     Load trained model from checkpoint.

#     Args:
#         checkpoint_path: Path to PyTorch Lightning checkpoint (.ckpt)
#         device: Device to load model on ('cuda' or 'cpu')

#     Returns:
#         Loaded model in evaluation mode

#     Raises:
#         FileNotFoundError: If checkpoint doesn't exist
#         RuntimeError: If checkpoint loading fails

#     """
#     ckpt_path = Path(checkpoint_path)
#     if not ckpt_path.exists():
#         msg = f"Checkpoint not found: {checkpoint_path}"
#         raise FileNotFoundError(msg)

#     log.info("Loading model from: %s", checkpoint_path)

#     try:
#         # Load model from checkpoint
#         model = WaterExtractionSegmentation(
#             **WaterExtractionSegmentation.load_from_checkpoint(
#                 checkpoint_path,
#                 map_location=device,
#                 strict=False,
#             ).hparams
#         )
#         model.load_state_dict(
#             torch.load(checkpoint_path, map_location=device)["state_dict"]
#         )
#         model.eval()
#         model.to(device)
#         log.info("Model loaded successfully on %s", device)
#         return model
#     except Exception as e:
#         msg = f"Failed to load checkpoint: {e}"
#         raise RuntimeError(msg) from e

# def load_model(checkpoint_path: str, device: str):
#     ckpt = torch.load(checkpoint_path, map_location=device)

#     hp = ckpt["hyper_parameters"]

#     # Rebuild loss from serialized config
#     # LightningCLI stores the instantiated object, not a dict
#     loss = hp["loss"]

#     model = WaterExtractionSegmentation(
#         encoder=hp["encoder"],
#         in_channels=hp["in_channels"],
#         num_classes=hp["num_classes"],
#         image_size=tuple(hp["image_size"]),
#         max_samples=hp["max_samples"],
#         loss=loss,
#         ignore_index=hp["ignore_index"],
#         weights=hp.get("weights"),
#         class_labels=hp.get("class_labels"),
#         class_colors=hp.get("class_colors"),
#         weights_from_checkpoint_path=None,  # avoid recursion
#     )

#     model.configure_model()

#     # Strip legacy Lightning "model." prefix
#     raw_sd = ckpt["state_dict"]
#     sd = {
#         k.replace("model.", "", 1): v
#         for k, v in raw_sd.items()
#         if k.startswith("model.")
#     }

#     missing, unexpected = model.load_state_dict(sd, strict=False)

#     if missing:
#         print("[WARN] Missing keys (first 10):", missing[:10])
#     if unexpected:
#         print("[WARN] Unexpected keys (first 10):", unexpected[:10])

#     model.eval()
#     model.to(device)
#     return model


# def load_model(checkpoint_path: str, device: str):
#     ckpt = torch.load(checkpoint_path, map_location=device)

#     # Extract the already-instantiated model hyperparameters
#     hp = ckpt["hyper_parameters"]

#     model = WaterExtractionSegmentation(
#         encoder=hp["encoder"],
#         in_channels=hp["in_channels"],
#         num_classes=hp["num_classes"],
#         image_size=tuple(hp["image_size"]),
#         max_samples=hp["max_samples"],
#         loss=hp["loss"],                # already instantiated loss
#         ignore_index=hp["ignore_index"],
#         weights=hp.get("weights"),
#         class_labels=hp.get("class_labels"),
#         class_colors=hp.get("class_colors"),
#         weights_from_checkpoint_path=None,
#     )

#     model.configure_model()

#     # Load full state dict WITHOUT prefix stripping
#     model.load_state_dict(
#         {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()},
#         strict=True,
#     )

#     model.eval()
#     model.to(device)
#     return model


def load_model(checkpoint_path: str, device: str):
    import torch

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["state_dict"]
    hp = ckpt["hyper_parameters"]

    # Instantiate exactly like training
    model = WaterExtractionSegmentation(
        encoder=hp["encoder"],
        in_channels=hp["in_channels"],
        num_classes=hp["num_classes"],
        image_size=tuple(hp["image_size"]),
        max_samples=hp["max_samples"],
        loss=None,
        ignore_index=hp["ignore_index"],
        weights=hp.get("weights"),
        class_labels=hp.get("class_labels"),
        class_colors=hp.get("class_colors"),
        weights_from_checkpoint_path=None,
    )

    # HIS IS THE CRITICAL LINE
    model.configure_model()

    # Load weights AS-IS (checkpoint already uses model.*)
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(device)
    return model


def standardization(
    input_tensor: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    mean = mean.to(input_tensor.device)
    std = std.to(input_tensor.device)
    return (input_tensor - mean) / std


# =============================================================================
# 3. SLIDING WINDOW INFERENCE
# =============================================================================


def sliding_window_inference(  # noqa: PLR0913
    model: torch.nn.Module,
    input_raster_path: str,
    output_raster_path: str,
    mean: list[float],
    std: list[float],
    *,
    model_in_channels: int | None = None,
    window_size: int = 512,
    stride: int = 512,
    batch_size: int = 4,
    device: str = "cuda",
    nodata_val: float = -32767,
) -> None:
    """
    Perform sliding window inference over full AOI raster.

    Uses overlapping windows with weighted averaging to reduce border artifacts.
    Windows are processed in batches for efficiency.

    Args:
        model: Trained PyTorch model in eval mode
        input_raster_path: Path to stacked input raster
        output_raster_path: Where to save prediction raster
        mean: Per-band mean for standardization
        std: Per-band standard deviation for standardization
        model_in_channels: Number of input channels the model expects (if None, uses all bands from raster)
        window_size: Size of inference windows (height and width)
        stride: Stride between window positions (< window_size for overlap)
        batch_size: Number of windows to process in parallel
        device: Device to run inference on
        nodata_val: NoData value to skip in input

    """
    log.info("=" * 80)
    log.info("SLIDING WINDOW INFERENCE")
    log.info("=" * 80)
    log.info("Window size: %d x %d", window_size, window_size)
    log.info("Stride: %d", stride)
    log.info("Batch size: %d", batch_size)
    log.info("Overlap: %.1f%%", (1 - stride / window_size) * 100)

    with rasterio.open(input_raster_path) as src:
        height, width = src.height, src.width
        num_bands = src.count
        profile = src.profile
        transform = src.transform
        crs = src.crs

        ref_shape = src.shape  # (H, W)
        ref_transform = src.transform
        ref_crs = src.crs

        # Determine how many channels to actually load
        channels_to_load = model_in_channels if model_in_channels is not None else num_bands
        
        # Validate that input raster has enough bands
        if channels_to_load > num_bands:
            msg = f"Model expects {channels_to_load} channels but input raster only has {num_bands} bands"
            raise ValueError(msg)
        
        # Log channel information
        log.info("Input raster bands: %d", num_bands)
        log.info("Model expects: %d channels", channels_to_load)
        if channels_to_load < num_bands:
            log.info("Will load only first %d channels (excluding intensity)", channels_to_load)
        
        log.info("Input shape: (%d, %d, %d)", channels_to_load, height, width)

        # Initialize output arrays
        # Prediction accumulator (sum of predictions)
        prediction_sum = np.zeros((height, width), dtype=np.float32)
        # Weight accumulator (count of predictions per pixel)
        weight_sum = np.zeros((height, width), dtype=np.float32)

        # Compute window positions
        windows = []
        for y in range(0, height - window_size + 1, stride):
            for x in range(0, width - window_size + 1, stride):
                windows.append((x, y))

        # Handle edge cases (add windows to cover remaining areas)
        # Right edge
        if width > window_size and (width - window_size) % stride != 0:
            x_last = width - window_size
            for y in range(0, height - window_size + 1, stride):
                if (x_last, y) not in windows:
                    windows.append((x_last, y))

        # Bottom edge
        if height > window_size and (height - window_size) % stride != 0:
            y_last = height - window_size
            for x in range(0, width - window_size + 1, stride):
                if (x, y_last) not in windows:
                    windows.append((x, y_last))

        # Bottom-right corner
        if (width - window_size) % stride != 0 and (height - window_size) % stride != 0:
            x_last = width - window_size
            y_last = height - window_size
            if (x_last, y_last) not in windows:
                windows.append((x_last, y_last))

        log.info("Total windows to process: %d", len(windows))

        # Validate and slice mean/std to match model's expected channels
        if len(mean) < channels_to_load or len(std) < channels_to_load:
            msg = f"Mean/std arrays have {len(mean)}/{len(std)} values but model expects {channels_to_load} channels"
            raise ValueError(msg)
        
        # Slice mean/std if they have more values than needed
        mean_sliced = mean[:channels_to_load]
        std_sliced = std[:channels_to_load]
        
        if len(mean) > channels_to_load:
            log.info("Slicing mean/std from %d to %d values to match model channels", len(mean), channels_to_load)
        
        # Convert stats to tensors
        mean_t = torch.tensor(mean_sliced, dtype=torch.float32, device=device).view(-1, 1, 1)
        std_t = torch.tensor(std_sliced, dtype=torch.float32, device=device).view(-1, 1, 1)

        # Process windows in batches
        batch_windows = []
        batch_positions = []

        valid_lidar_mask_path = "/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/gdl_projects/geo-deep-learning/data/02NB000/valir_lidar_mask.gpkg"

        # Load valid lidar mask GPKG
        with fiona.open(valid_lidar_mask_path) as shp:
            # Reproject check (important)
            if shp.crs and shp.crs != ref_crs:
                raise ValueError("CRS mismatch between valid mask and raster")

            geometries = [feature["geometry"] for feature in shp]

        valid_mask_full = rasterize(
            [(geom, 1) for geom in geometries],
            out_shape=ref_shape,
            transform=ref_transform,
            fill=0,
            dtype="uint8",
        )

        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(windows, desc="Inference")):
                window = Window(x, y, window_size, window_size)

                # Read window data - load only the channels the model expects
                if channels_to_load < num_bands:
                    # Load only first N channels (1-indexed in rasterio)
                    bands_to_read = list(range(1, channels_to_load + 1))
                    window_data = src.read(bands_to_read, window=window).astype(np.float32)  # (C, H, W)
                else:
                    # Load all bands
                    window_data = src.read(window=window).astype(np.float32)  # (C, H, W)

                # Check for nodata
                # valid_mask = window_data[0] != nodata_val
                # valid_mask_window = valid_mask_full[y:y+window_size, x:x+window_size].astype(bool)

                valid_mask_window = valid_mask_full[
                    y : y + window_size,
                    x : x + window_size,
                ].astype(bool)

                valid_mask_window[:] = True

                # Convert to tensor
                window_tensor = torch.from_numpy(window_data).float()  # (C, H, W)

                # Apply standardization
                # window_tensor = standardization(window_tensor.unsqueeze(0), mean_t, std_t)  # (1, C, H, W)

                window_tensor = window_tensor.to(device)
                window_tensor = standardization(
                    window_tensor.unsqueeze(0), mean_t, std_t
                )

                window_tensor[:, :, ~valid_mask_window] = 0.0

                batch_windows.append(window_tensor)
                # batch_positions.append((x, y, valid_mask))
                batch_positions.append((x, y, valid_mask_window))

                # # Convert to tensor
                # window_tensor = torch.from_numpy(window_data).float()  # (C, H, W)

                # # --- DEBUG: first window only ---
                # if len(batch_windows) == 0:
                #     raw = window_tensor

                #     raw_all_nodata = torch.all(raw == nodata_val)
                #     raw_min = raw.min().item()
                #     raw_max = raw.max().item()

                #     print("\n[DEBUG] FIRST WINDOW RAW")
                #     print("  all_nodata:", raw_all_nodata)
                #     print("  min / max:", raw_min, raw_max)

                # # Apply standardization
                # window_tensor = window_tensor.to(device)
                # window_tensor = standardization(window_tensor.unsqueeze(0), mean_t, std_t)

                # # --- DEBUG: after standardization ---
                # if len(batch_windows) == 0:
                #     std_x = window_tensor.squeeze(0)

                #     print("[DEBUG] FIRST WINDOW STD")
                #     print("  finite:", torch.isfinite(std_x).all().item())
                #     print("  min / max:", std_x.min().item(), std_x.max().item())
                #     print(
                #         "  per-channel mean:",
                #         std_x.mean(dim=(1, 2)).detach().cpu().numpy()
                #     )

                # batch_windows.append(window_tensor)
                # batch_positions.append((x, y, valid_mask))

                # Process batch when full or at end
                if len(batch_windows) == batch_size or i == len(windows) - 1:
                    # Stack batch
                    batch_tensor = torch.cat(batch_windows, dim=0).to(
                        device
                    )  # (B, C, H, W)

                    # Run inference
                    logits = model(batch_tensor)  # (B, num_classes, H, W)

                    # Get predictions (class probabilities)
                    if logits.shape[1] == 1:
                        # Binary with sigmoid
                        probs = torch.sigmoid(logits).squeeze(1)  # (B, H, W)
                    else:
                        # Multi-class with softmax (water is class 1)
                        probs = torch.softmax(logits, dim=1)[:, 1, :, :]  # (B, H, W)

                    probs = probs.cpu().numpy()

                    # Accumulate predictions with spatial weighting
                    for j, (bx, by, valid_mask) in enumerate(batch_positions):
                        pred = probs[j]  # (H, W)

                        # Apply Gaussian-like center weighting to reduce edge artifacts
                        weight = _compute_window_weight(window_size)

                        # Mask invalid pixels
                        pred[~valid_mask] = 0
                        weight[~valid_mask] = 0

                        # Accumulate
                        prediction_sum[
                            by : by + window_size, bx : bx + window_size
                        ] += pred * weight
                        weight_sum[by : by + window_size, bx : bx + window_size] += (
                            weight
                        )

                    # Clear batch
                    batch_windows = []
                    batch_positions = []

    # Compute final prediction (weighted average)
    # Avoid division by zero
    final_prediction = np.zeros_like(prediction_sum)
    valid_pixels = weight_sum > 0
    final_prediction[valid_pixels] = (
        prediction_sum[valid_pixels] / weight_sum[valid_pixels]
    )

    # Convert probabilities to binary mask (threshold at 0.5)
    binary_prediction = (final_prediction > 0.5).astype(np.uint8)

    # Save output raster
    log.info("Saving prediction raster: %s", output_raster_path)
    profile.update(
        {
            "dtype": "uint8",
            "count": 1,
            "nodata": 255,
            "compress": "lzw",
        },
    )

    with rasterio.open(output_raster_path, "w", **profile) as dst:
        dst.write(binary_prediction, 1)

    log.info("Prediction complete: %s", output_raster_path)


def _compute_window_weight(window_size: int) -> np.ndarray:
    """
    Compute spatial weight for a window to reduce edge artifacts.

    Uses distance from center with linear falloff. Center pixels have weight 1.0,
    edges have weight that decreases linearly to 0.1.

    Args:
        window_size: Size of the window (assumes square)

    Returns:
        2D weight array of shape (window_size, window_size)

    """
    # Create coordinate grids
    y, x = np.ogrid[:window_size, :window_size]
    center = window_size / 2

    # Compute distance from center (normalized)
    dist_y = np.abs(y - center) / center
    dist_x = np.abs(x - center) / center
    dist = np.maximum(dist_y, dist_x)  # Use max for square weighting

    # Linear falloff: weight = 1.0 at center, 0.1 at edges
    weight = 1.0 - 0.9 * dist
    weight = np.clip(weight, 0.1, 1.0)

    return weight


# =============================================================================
# 4. AOI CROPPING
# =============================================================================


def crop_raster_to_aoi(
    input_raster_path: str,
    output_raster_path: str,
    aoi_vector_path: str,
) -> None:
    """
    Crop a raster to the extent of an AOI polygon.

    Args:
        input_raster_path: Path to input raster to crop
        output_raster_path: Path to save cropped raster
        aoi_vector_path: Path to AOI vector file (gpkg or shp)

    """
    import rasterio.mask

    log.info("Cropping raster to AOI boundary")
    log.info("  Input: %s", input_raster_path)
    log.info("  AOI: %s", aoi_vector_path)
    log.info("  Output: %s", output_raster_path)

    # Read AOI geometries
    with fiona.open(aoi_vector_path, "r") as aoi_src:
        aoi_geoms = [feature["geometry"] for feature in aoi_src]
        aoi_crs = aoi_src.crs

    # Crop raster
    with rasterio.open(input_raster_path) as src:
        # Check CRS compatibility
        if src.crs != aoi_crs:
            log.warning(
                "CRS mismatch between raster (%s) and AOI (%s)", src.crs, aoi_crs
            )

        # Mask and crop
        out_image, out_transform = rasterio.mask.mask(
            src,
            aoi_geoms,
            crop=True,
            nodata=src.nodata,
        )

        # Update metadata - preserve compression and dtype
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw",  # Ensure compression is enabled
            }
        )

        log.info("  Cropped shape: %s", out_image.shape)
        log.info("  Output dtype: %s", out_meta["dtype"])
        log.info("  Compression: %s", out_meta.get("compress", "none"))

    # Write cropped raster
    with rasterio.open(output_raster_path, "w", **out_meta) as dst:
        dst.write(out_image)

    log.info("Cropped raster saved: %s", output_raster_path)


# =============================================================================
# 5. VECTOR EXPORT (POLYGONIZATION)
# =============================================================================


def export_vectors(
    prediction_raster_path: str,
    output_vector_path: str,
    *,
    water_class: int = 1,
    simplify_tolerance: float = 1.0,
) -> None:
    """
    Polygonize prediction raster and export to GeoPackage.

    Extracts water class (value=1) and converts to vector polygons.
    Applies geometry simplification and validation.

    Args:
        prediction_raster_path: Path to binary prediction raster
        output_vector_path: Where to save output GeoPackage
        water_class: Value in raster representing water (default: 1)
        simplify_tolerance: Simplification tolerance in raster units (default: 1.0)

    """
    log.info("=" * 80)
    log.info("VECTOR EXPORT")
    log.info("=" * 80)
    log.info("Input raster: %s", prediction_raster_path)
    log.info("Output vector: %s", output_vector_path)

    with rasterio.open(prediction_raster_path) as src:
        image = src.read(1)
        transform = src.transform
        crs = src.crs

        # Extract shapes for water class
        log.info("Extracting water polygons (class=%d)...", water_class)
        mask = image == water_class

        # Polygonize
        geoms = []
        for geom, value in shapes(image, mask=mask, transform=transform):
            if value == water_class:
                # Convert to shapely geometry
                shp = shapely_shape(geom)

                # Simplify to reduce vertex count
                if simplify_tolerance > 0:
                    shp = shp.simplify(simplify_tolerance, preserve_topology=True)

                # Validate geometry
                if not shp.is_valid:
                    from shapely.validation import make_valid

                    shp = make_valid(shp)

                # Skip empty or very small polygons
                if shp.is_empty or shp.area < 1.0:
                    continue

                geoms.append(shp)

        log.info("Extracted %d water polygons", len(geoms))

        if len(geoms) == 0:
            log.warning("No water polygons found!")
            return

        # Define schema
        schema = {
            "geometry": "Polygon",
            "properties": {
                "class": "int",
                "area": "float",
            },
        }

        # Write to GeoPackage
        log.info("Writing to GeoPackage...")
        with fiona.open(
            output_vector_path,
            "w",
            driver="GPKG",
            crs=crs,
            schema=schema,
        ) as dst:
            for i, geom in enumerate(geoms):
                feature = {
                    "geometry": mapping(geom),
                    "properties": {
                        "class": water_class,
                        "area": geom.area,
                    },
                }
                dst.write(feature)

        log.info("Vector export complete: %s", output_vector_path)


# =============================================================================
# 6. MAIN INFERENCE PIPELINE
# =============================================================================


def _resolve_aoi_vector(data_folder: str) -> Path | None:
    """
    Find the AOI vector file in the data folder.

    Looks for aoi.gpkg or aoi.shp.

    Args:
        data_folder: Path to the data folder

    Returns:
        Path to AOI vector file, or None if not found

    """
    data_path = Path(data_folder)
    candidates = [data_path / "aoi.gpkg", data_path / "aoi.shp"]

    for candidate in candidates:
        if candidate.exists():
            log.info("Found AOI vector: %s", candidate)
            return candidate

    log.warning(
        "No AOI vector found (checked: %s)", ", ".join(str(c) for c in candidates)
    )
    return None


def run_inference(  # noqa: PLR0913
    checkpoint_path: str,
    output_folder: str,
    mean: list[float],
    std: list[float],
    *,
    data_folder: str = None,
    stacked_inputs: str = None,
    window_size: int = 512,
    stride: int = 512,
    batch_size: int = 4,
    device: str = "cuda",
    include_intensity: bool = True,
    export_vector: bool = True,
    simplify_tolerance: float = 1.0,
) -> dict[str, str]:
    """
    Run complete inference pipeline.

    Pipeline steps:
    1. Run sliding window inference over preprocessed inputs (from stacked_inputs)
    2. Crop prediction to AOI boundary (if AOI vector is available via data_folder)
    3. Export prediction raster (cropped)
    4. Polygonize and export vectors (optional)

    Args:
        checkpoint_path: Path to trained model checkpoint
        stacked_inputs: Path to preprocessed stacked_inputs.tif (e.g., preprocessed_02NB000/stacked_inputs.tif)
        data_folder: Path to RAW data folder containing aoi.gpkg/shp (e.g., data/02NB000/)
                    Used ONLY to extract AOI name and find AOI boundary vector.
                    NOT used for inference input files.
        output_folder: Where to save all outputs
        mean: Per-band mean for standardization (must match training)
        std: Per-band standard deviation for standardization (must match training)
        window_size: Inference window size (default: 512)
        stride: Stride between windows (default: 256, 50% overlap)
        batch_size: Batch size for inference (default: 4)
        device: Device for inference ('cuda' or 'cpu', default: 'cuda')
        include_intensity: Whether to include intensity band (default: True)
        export_vector: Whether to export vectorized results (default: True)
        simplify_tolerance: Geometry simplification tolerance (default: 1.0)

    Returns:
        Dictionary with paths to outputs:
        - 'stacked_inputs': preprocessed multi-band raster used for inference
        - 'prediction_raster_full': binary prediction raster (full extent, uncropped)
        - 'prediction_raster': binary prediction raster (cropped to AOI)
        - 'prediction_vector': vectorized waterbodies (if export_vector=True)

    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Start timing
    start_time = time.time()
    log.info("=" * 80)
    log.info("INFERENCE STARTED AT: %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 80)

    # Extract AOI name and find AOI vector
    # data_folder is ONLY used for AOI metadata (name + vector boundary)
    # Actual inference uses stacked_inputs from preprocessed folder
    aoi_name = "unknown_aoi"
    aoi_vector_path = None

    if data_folder is not None:
        # data_folder contains RAW data: dtm.tif, dsm.tif, aoi.gpkg, etc.
        # Used ONLY to extract AOI name and find AOI boundary
        aoi_name = Path(data_folder).name
        aoi_vector_path = _resolve_aoi_vector(data_folder)
        log.info("Using data_folder for AOI metadata: %s", data_folder)

    # If no data_folder provided, try to infer AOI name from stacked_inputs path
    if data_folder is None and stacked_inputs is not None:
        # Expected path: .../preprocessed_{aoi_name}/stacked_inputs.tif
        stacked_path = Path(stacked_inputs)
        parent_name = stacked_path.parent.name
        if parent_name.startswith("preprocessed_"):
            aoi_name = parent_name.replace("preprocessed_", "")
            log.info("Inferred AOI name from stacked_inputs path: %s", aoi_name)

        # Try to find the original data folder for AOI vector
        # Common pattern: data/{aoi_name}/ contains raw data
        possible_data_folder = stacked_path.parent.parent / aoi_name
        if possible_data_folder.exists():
            aoi_vector_path = _resolve_aoi_vector(str(possible_data_folder))
            log.info(
                "Found AOI vector in inferred data folder: %s", possible_data_folder
            )

    log.info("=" * 80)
    log.info("WATER EXTRACTION INFERENCE")
    log.info("=" * 80)
    log.info("AOI name: %s", aoi_name)
    log.info("Output folder: %s", output_folder)
    log.info("Checkpoint: %s", checkpoint_path)
    log.info("Device: %s", device)
    log.info("Mean: %s", mean)
    log.info("Std: %s", std)
    if aoi_vector_path:
        log.info("AOI vector: %s", aoi_vector_path)
    log.info("=" * 80)

    # Step 1: Get stacked inputs (use existing or preprocess from raw data)
    if stacked_inputs is not None:
        # Use already preprocessed inputs (typical for inference after training)
        stacked_inputs = str(Path(stacked_inputs).resolve())
        log.info("Using preprocessed stacked_inputs: %s", stacked_inputs)
    elif data_folder is not None:
        # Preprocess raw data on-the-fly (less common, for ad-hoc inference)
        log.info("Preprocessing raw data from: %s", data_folder)
        stacked_inputs = preprocess_aoi(
            data_folder=data_folder,
            output_folder=output_folder,
            include_intensity=include_intensity,
        )
    else:
        raise ValueError("Must provide either --stacked_inputs or --data_folder")

    # Step 2: Load model
    model = load_model(checkpoint_path, device=device)

    # Get model's expected input channels
    model_in_channels = model.hparams.get("in_channels", None)
    
    print(
        "[DEBUG] Encoder first conv weight mean:",
        model.model.encoder.conv1.weight.mean().item(),
    )

    print("[DEBUG] Model class:", model.__class__)
    print("[DEBUG] In channels:", model_in_channels)
    print("[DEBUG] Num classes:", model.hparams.get("num_classes", "N/A"))
    print("[DEBUG] Encoder:", model.hparams.get("encoder", "N/A"))

    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            print("[DEBUG] Non-finite weights in:", name)
            break

    # Step 3: Run inference (full extent)
    prediction_raster_full = str(output_path / f"water_prediction_{aoi_name}_full.tif")
    sliding_window_inference(
        model=model,
        input_raster_path=stacked_inputs,
        output_raster_path=prediction_raster_full,
        mean=mean,
        std=std,
        model_in_channels=model_in_channels,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size,
        device=device,
    )

    # Step 4: Crop to AOI boundary (if available)
    if aoi_vector_path is not None:
        prediction_raster = str(output_path / f"water_prediction_{aoi_name}.tif")
        crop_raster_to_aoi(
            input_raster_path=prediction_raster_full,
            output_raster_path=prediction_raster,
            aoi_vector_path=str(aoi_vector_path),
        )
        log.info("Using cropped prediction for vector export")
    else:
        log.warning("No AOI vector found - skipping cropping step")
        prediction_raster = prediction_raster_full

    outputs = {
        "stacked_inputs": stacked_inputs,
        "prediction_raster": prediction_raster,
        "prediction_raster_full": prediction_raster_full,
    }

    # Step 5: Export vectors (optional)
    if export_vector:
        prediction_vector = str(output_path / f"water_bodies_{aoi_name}.gpkg")
        export_vectors(
            prediction_raster_path=prediction_raster,
            output_vector_path=prediction_vector,
            simplify_tolerance=simplify_tolerance,
        )
        outputs["prediction_vector"] = prediction_vector

    # Calculate total runtime
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    log.info("=" * 80)
    log.info("INFERENCE COMPLETE")
    log.info("End time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info(
        "Total duration: %02d:%02d:%02d (HH:MM:SS)",
        int(hours),
        int(minutes),
        int(seconds),
    )
    log.info("=" * 80)
    for key, value in outputs.items():
        log.info("%s: %s", key, value)

    return outputs


# =============================================================================
# 7. CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Water extraction inference from LiDAR elevation data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to save outputs",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs="+",
        required=True,
        help="Per-band mean for standardization (e.g., --mean 0.0 0.0 0.0)",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs="+",
        required=True,
        help="Per-band std for standardization (e.g., --std 1.0 1.0 1.0)",
    )

    # Input arguments - at least one is required
    parser.add_argument(
        "--stacked_inputs",
        type=str,
        required=False,
        help="Path to preprocessed stacked_inputs.tif (e.g., preprocessed_02NB000/stacked_inputs.tif). Required for inference.",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        required=False,
        help="Path to RAW data folder with aoi.gpkg/shp (e.g., data/02NB000/). Used only for AOI name and boundary vector. "
        "If not provided, AOI name is inferred from stacked_inputs path.",
    )

    # Optional arguments
    parser.add_argument(
        "--window_size",
        type=int,
        default=512,
        help="Size of inference windows",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Stride between windows (lower = more overlap)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of windows to process in parallel",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference",
    )
    parser.add_argument(
        "--no_intensity",
        action="store_true",
        help="Exclude intensity band from inputs",
    )
    parser.add_argument(
        "--no_vector",
        action="store_true",
        help="Skip vector export (raster only)",
    )
    parser.add_argument(
        "--simplify_tolerance",
        type=float,
        default=1.0,
        help="Geometry simplification tolerance (in raster units)",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    # Validate input lengths
    if len(args.mean) != len(args.std):
        msg = "Mean and std must have the same length"
        raise ValueError(msg)

    # Validate that at least stacked_inputs or data_folder is provided
    if args.stacked_inputs is None and args.data_folder is None:
        msg = "Must provide at least --stacked_inputs (for inference) or --data_folder (for preprocessing)"
        raise ValueError(msg)

    # Run inference
    run_inference(
        checkpoint_path=args.checkpoint,
        output_folder=args.output_folder,
        mean=args.mean,
        std=args.std,
        data_folder=args.data_folder,
        stacked_inputs=args.stacked_inputs,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        device=args.device,
        include_intensity=not args.no_intensity,
        export_vector=not args.no_vector,
        simplify_tolerance=args.simplify_tolerance,
    )


if __name__ == "__main__":
    main()

"""Utility functions related to rasters/arrays."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from tqdm import tqdm


def align_to_reference(
    reference_path: str,
    input_path: str,
    output_path: str,
    resample_alg: str = "bilinear",
    nodata_val: float = -32767,
) -> None:
    """
    Aligns an input raster to match the spatial reference of another raster.

    The process adjusts the input raster's extent, resolution, and CRS to match
    the reference raster, and saves the aligned output.

    Args:
        reference_path: Path to the reference raster to align to.
        input_path: Path to the input raster to be aligned.
        output_path: Path where the aligned raster will be saved.
        resample_alg: Resampling method to use ('nearest', 'bilinear', 'cubic').
        nodata_val: Value to use for nodata if not defined in the source raster.

    """
    resample_methods = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }

    if resample_alg not in resample_methods:
        msg = f"Unsupported resampling method: {resample_alg}"
        raise ValueError(msg)

    with rasterio.open(reference_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_height = ref.height
        dst_width = ref.width

        with rasterio.open(input_path) as src:
            src_data = src.read(1)
            src_nodata = src.nodata if src.nodata is not None else nodata_val

            profile = src.profile
            profile.update(
                {
                    "crs": dst_crs,
                    "transform": dst_transform,
                    "width": dst_width,
                    "height": dst_height,
                    "nodata": src_nodata,
                    "BIGTIFF": "YES",
                    "compress": "lzw",
                },
            )

            with rasterio.open(output_path, "w", **profile) as dst:
                reproject(
                    source=src_data,
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resample_methods[resample_alg],
                    src_nodata=src_nodata,
                    dst_nodata=src_nodata,
                )


def compute_dataset_stats_from_list(
    tile_paths: Sequence[str],
) -> tuple[list[float], list[float]]:
    """
    Compute global per-band mean and standard deviation from a list of raster tiles.

    For each input raster tile, the function reads all bands, excludes pixels matching
    the band-specific nodata value (as defined in the raster metadata), and aggregates
    valid pixels to compute global statistics across all tiles.

    Args:
        tile_paths (list[str]):
            List of file paths to raster tiles. Each raster may contain multiple bands,
            and each band may define its own nodata value in metadata.

    Returns:
        tuple[list[float], list[float]]:
            A tuple containing two lists:
            - The first list contains the mean value for each band.
            - The second list contains the standard deviation for each band.

    Raises:
        AssertionError:
            If 'tile_paths' is empty.

    Example:
        >>> means, stds = compute_dataset_stats_from_list(["tile1.tif", "tile2.tif"])
        >>> print(means, stds)

    """
    if not tile_paths:
        msg = "No input tiles provided for statistics."
        raise ValueError(msg)

    sum_pixels = None
    sum_sq_pixels = None
    total_valid_pixels = None

    for path in tqdm(tile_paths, desc="Computing global dataset stats"):
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)  # shape (C, H, W)
            nodata_vals = src.nodatavals  # tuple of nodata values per band

        valid_pixels = []
        for i in range(img.shape[0]):
            band = img[i]
            nodata_val = nodata_vals[i] if nodata_vals[i] is not None else np.nan
            mask = band != nodata_val
            valid_pixels.append(band[mask])

        if sum_pixels is None:
            sum_pixels = np.zeros(img.shape[0])
            sum_sq_pixels = np.zeros(img.shape[0])
            total_valid_pixels = np.zeros(img.shape[0], dtype=int)

        for i in range(img.shape[0]):
            sum_pixels[i] += np.sum(valid_pixels[i])
            sum_sq_pixels[i] += np.sum(valid_pixels[i] ** 2)
            total_valid_pixels[i] += len(valid_pixels[i])

    means = sum_pixels / total_valid_pixels
    stds = np.sqrt((sum_sq_pixels / total_valid_pixels) - (means**2))

    return means.tolist(), stds.tolist()

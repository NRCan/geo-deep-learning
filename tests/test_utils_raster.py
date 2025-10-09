"""Unit tests for raster utility functions."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geo_deep_learning.utils.rasters import (
    align_to_reference,
    compute_dataset_stats_from_list,
)


@pytest.fixture
def tmp_rasters(tmp_path: Path) -> tuple[Path, Path]:
    """Create small temporary rasters for testing."""
    width, height = 5, 5
    transform = from_origin(0, 5, 1, 1)
    crs = "EPSG:4326"

    # Create reference raster
    ref_data = np.arange(width * height, dtype=np.float32).reshape(height, width)
    ref_path = tmp_path / "ref.tif"
    with rasterio.open(
        ref_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=ref_data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(ref_data, 1)

    # Create input raster (different CRS and transform)
    input_data = np.ones((height, width), dtype=np.float32) * 10
    input_transform = from_origin(0, 10, 2, 2)
    input_path = tmp_path / "input.tif"
    with rasterio.open(
        input_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=input_data.dtype,
        crs="EPSG:3857",
        transform=input_transform,
        nodata=-32767,
    ) as dst:
        dst.write(input_data, 1)

    return ref_path, input_path


def test_align_to_reference(tmp_rasters: Path, tmp_path: Path) -> None:
    """Test that output raster is aligned to the reference raster."""
    ref_path, input_path = tmp_rasters
    output_path = tmp_path / "aligned.tif"
    nodata_value = -32767

    align_to_reference(ref_path, input_path, output_path, resample_alg="nearest")

    assert output_path.exists(), "Aligned raster file was not created."

    with rasterio.open(ref_path) as ref, rasterio.open(output_path) as aligned:
        # Check matching spatial properties
        assert aligned.crs == ref.crs
        assert aligned.transform.almost_equals(ref.transform)
        assert aligned.width == ref.width
        assert aligned.height == ref.height
        assert aligned.nodata == nodata_value
        data = aligned.read(1)
        assert data.shape == (ref.height, ref.width)


def test_align_to_reference_invalid_method(tmp_rasters: Path, tmp_path: Path) -> None:
    """Test invalid resampling algorithm raises ValueError."""
    ref_path, input_path = tmp_rasters
    output_path = tmp_path / "aligned_invalid.tif"

    with pytest.raises(ValueError, match="Unsupported resampling method"):
        align_to_reference(ref_path, input_path, output_path, resample_alg="invalid")


def test_compute_dataset_stats_from_list(tmp_path: Path) -> None:
    """Test mean and std computation across multiple raster tiles."""
    # Create two simple rasters with different constant values
    data1 = np.ones((1, 4, 4), dtype=np.float32) * 5
    data2 = np.ones((1, 4, 4), dtype=np.float32) * 15

    def write_raster(path: Path, arr: np.ndarray) -> None:
        transform = from_origin(0, 4, 1, 1)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=arr.shape[1],
            width=arr.shape[2],
            count=1,
            dtype=arr.dtype,
            transform=transform,
            nodata=-9999,
        ) as dst:
            dst.write(arr)

    path1 = tmp_path / "tile1.tif"
    path2 = tmp_path / "tile2.tif"
    write_raster(path1, data1)
    write_raster(path2, data2)

    means, stds = compute_dataset_stats_from_list([path1, path2])

    expected_mean = 10.0
    expected_std = 5.0

    assert pytest.approx(means[0], rel=1e-6) == expected_mean
    assert pytest.approx(stds[0], rel=1e-6) == expected_std


def test_compute_dataset_stats_empty_list() -> None:
    """Test that empty list raises ValueError."""
    with pytest.raises(ValueError, match="No input tiles provided for statistics"):
        compute_dataset_stats_from_list([])

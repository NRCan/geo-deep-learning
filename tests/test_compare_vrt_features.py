"""Tests for stacked VRT/source feature comparison helpers."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from geo_deep_learning.tools.water_extraction.compare_vrt_features import (
    compare_stack_to_sources,
)


def _write_single_band_raster(
    path: Path,
    data: np.ndarray,
    *,
    description: str,
) -> None:
    """Write a simple single-band test raster."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        transform=from_origin(0, data.shape[0], 1, 1),
        crs="EPSG:4326",
        nodata=-9999,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
        dst.set_band_description(1, description)


def _write_multiband_stack(
    path: Path,
    bands: list[np.ndarray],
    descriptions: list[str],
) -> None:
    """Write a simple multiband stack for testing."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=bands[0].shape[0],
        width=bands[0].shape[1],
        count=len(bands),
        dtype="float32",
        transform=from_origin(0, bands[0].shape[0], 1, 1),
        crs="EPSG:4326",
        nodata=-9999,
    ) as dst:
        for index, (band, description) in enumerate(zip(bands, descriptions), start=1):
            dst.write(band.astype(np.float32), index)
            dst.set_band_description(index, description)


def test_compare_stack_to_sources_reports_matching_bands(tmp_path: Path) -> None:
    """Matching source rasters should compare exactly against stack bands."""
    ndsm = np.array([[1, 2], [3, 4]], dtype=np.float32)
    intensity = np.array([[10, 11], [12, 13]], dtype=np.float32)

    _write_single_band_raster(tmp_path / "ndsm.tif", ndsm, description="ndsm")
    _write_single_band_raster(
        tmp_path / "intensity.tif",
        intensity,
        description="intensity",
    )
    _write_multiband_stack(
        tmp_path / "stacked_inputs.tif",
        [ndsm, intensity],
        ["ndsm", "intensity"],
    )

    report = compare_stack_to_sources(
        tmp_path / "stacked_inputs.tif",
        source_dir=tmp_path,
    )

    assert report["stack_band_descriptions"] == ["ndsm", "intensity"]
    assert report["missing_sources"] == []
    first, second = report["comparisons"]
    assert first["exact_match"] is True
    assert first["best_matching_feature"] == "ndsm"
    assert first["order_matches_description"] is True
    assert second["exact_match"] is True
    assert second["best_matching_feature"] == "intensity"
    assert second["order_matches_description"] is True


def test_compare_stack_to_sources_flags_order_mismatch(tmp_path: Path) -> None:
    """Expected feature order should catch swapped bands even with valid descriptions."""
    ndsm = np.array([[1, 2], [3, 4]], dtype=np.float32)
    intensity = np.array([[10, 11], [12, 13]], dtype=np.float32)

    _write_single_band_raster(tmp_path / "ndsm.tif", ndsm, description="ndsm")
    _write_single_band_raster(
        tmp_path / "intensity.tif",
        intensity,
        description="intensity",
    )
    _write_multiband_stack(
        tmp_path / "stacked_inputs.tif",
        [ndsm, intensity],
        ["ndsm", "intensity"],
    )

    report = compare_stack_to_sources(
        tmp_path / "stacked_inputs.tif",
        source_dir=tmp_path,
        expected_features=["intensity", "ndsm"],
    )

    first, second = report["comparisons"]
    assert first["expected_feature"] == "intensity"
    assert first["order_matches_description"] is False
    assert first["exact_match"] is False
    assert first["best_matching_feature"] == "ndsm"
    assert second["expected_feature"] == "ndsm"
    assert second["order_matches_description"] is False
    assert second["exact_match"] is False
    assert second["best_matching_feature"] == "intensity"

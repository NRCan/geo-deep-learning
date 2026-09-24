"""Tests for AOI partitioning and parent prediction merging."""

from __future__ import annotations

import json
from pathlib import Path

import fiona
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping, shape

from geo_deep_learning.tools.water_extraction.download_elevation import (
    estimate_output_pixels,
)
from geo_deep_learning.tools.water_extraction.inference import (
    crop_raster_to_aoi,
    export_vectors,
    merge_prediction_rasters,
)
from geo_deep_learning.tools.water_extraction.partition_aoi import partition_aoi


def _write_vector(path: Path, geometry: object) -> None:
    """Write a single polygon feature to a GeoPackage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with fiona.open(
        path,
        "w",
        driver="GPKG",
        crs="EPSG:3979",
        schema={
            "geometry": "Polygon",
            "properties": {"name": "str"},
        },
    ) as dst:
        dst.write(
            {
                "geometry": mapping(geometry),
                "properties": {"name": path.stem},
            }
        )


def _read_union_geometry(path: Path):
    with fiona.open(path, "r") as src:
        geometries = [shape(feature["geometry"]) for feature in src]
    assert geometries
    geom = geometries[0]
    for other in geometries[1:]:
        geom = geom.union(other)
    return geom


def _write_binary_raster(
    path: Path,
    data: np.ndarray,
    *,
    west: float,
    north: float,
) -> None:
    """Write a small uint8 raster in EPSG:3979."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="uint8",
        transform=from_origin(west, north, 1, 1),
        crs="EPSG:3979",
        nodata=0,
    ) as dst:
        dst.write(data, 1)


def test_partition_aoi_no_split_at_exact_pixel_budget(tmp_path: Path) -> None:
    """AOIs at the exact budget should remain unsplit."""
    source_dir = tmp_path / "source_aoi"
    aoi_path = source_dir / "aoi.gpkg"
    buffered_path = source_dir / "aoi_buffered.gpkg"
    output_root = tmp_path / "partitions"

    _write_vector(aoi_path, box(0, 0, 10, 10))
    _write_vector(buffered_path, box(0, 0, 10, 10))

    result = partition_aoi(
        input_aoi=str(aoi_path),
        buffered_aoi=str(buffered_path),
        output_root=str(output_root),
        resolution=1,
        max_pixels=100,
        overlap_meters=0,
        child_id_prefix="child",
    )

    assert result["split_required"] is False
    assert result["estimated_parent_pixels"] == 100
    assert result["children"] == []

    manifest = json.loads((output_root / "partition_manifest.json").read_text())
    assert manifest["split_required"] is False
    assert manifest["estimated_parent_pixels"] == 100


def test_partition_aoi_splits_large_aoi_and_skips_empty_cells(tmp_path: Path) -> None:
    """Partitioning should create deterministic child AOIs and skip empty cells."""
    source_dir = tmp_path / "source_aoi"
    aoi_path = source_dir / "aoi.gpkg"
    buffered_path = source_dir / "aoi_buffered.gpkg"
    output_root = tmp_path / "partitions"

    l_shape = box(0, 0, 100, 20).union(box(0, 20, 20, 100))
    _write_vector(aoi_path, l_shape)
    _write_vector(buffered_path, l_shape.buffer(5))

    result = partition_aoi(
        input_aoi=str(aoi_path),
        buffered_aoi=str(buffered_path),
        output_root=str(output_root),
        resolution=1,
        max_pixels=2500,
        overlap_meters=5,
        child_id_prefix="child",
    )

    assert result["split_required"] is True
    children = result["children"]
    assert len(children) > 1
    assert len(children) < 9
    assert [child["child_id"] for child in children] == [
        f"child_{idx:04d}" for idx in range(len(children))
    ]

    unioned_children = None
    for child in children:
        child_geom = _read_union_geometry(Path(child["aoi_path"]))
        unioned_children = (
            child_geom if unioned_children is None else unioned_children.union(child_geom)
        )

    assert unioned_children is not None
    assert unioned_children.symmetric_difference(l_shape).area < 1e-6

    index_path = output_root / "subaoi_index.gpkg"
    assert result["subaoi_index_gpkg"] == str(index_path.resolve())
    assert fiona.listlayers(index_path) == ["subaoi_core", "subaoi_buffered"]
    with fiona.open(index_path, layer="subaoi_core") as src:
        index_features = list(src)

    assert len(index_features) == len(children)
    assert {feature["properties"]["child_id"] for feature in index_features} == {
        child["child_id"] for child in children
    }
    assert {
        Path(feature["properties"]["aoi_path"]).resolve()
        for feature in index_features
    } == {Path(child["aoi_path"]).resolve() for child in children}


def test_partition_aoi_child_buffer_includes_overlap(tmp_path: Path) -> None:
    """Each child buffered AOI should extend beyond its core AOI by the configured overlap."""
    source_dir = tmp_path / "source_aoi"
    aoi_path = source_dir / "aoi.gpkg"
    buffered_path = source_dir / "aoi_buffered.gpkg"
    output_root = tmp_path / "partitions"
    overlap = 10.0

    _write_vector(aoi_path, box(0, 0, 120, 120))
    _write_vector(buffered_path, box(-10, -10, 130, 130))

    result = partition_aoi(
        input_aoi=str(aoi_path),
        buffered_aoi=str(buffered_path),
        output_root=str(output_root),
        resolution=1,
        max_pixels=3600,
        overlap_meters=overlap,
        child_id_prefix="child",
    )

    first_child = result["children"][0]
    core_geom = _read_union_geometry(Path(first_child["aoi_path"]))
    buffered_geom = _read_union_geometry(Path(first_child["buffered_aoi_path"]))

    assert buffered_geom.contains(core_geom)
    assert buffered_geom.area > core_geom.area


def test_estimate_output_pixels_returns_expected_pixel_count() -> None:
    """Shared raster size estimation should expose height, width, and pixels."""
    assert estimate_output_pixels((0.0, 0.0, 10.2, 20.7), 2) == (11, 6, 66)


def test_merge_prediction_rasters_and_crop_to_parent_aoi(tmp_path: Path) -> None:
    """Child prediction rasters should merge, then crop and vectorize against the parent AOI."""
    child_a = tmp_path / "child_a_full.tif"
    child_b = tmp_path / "child_b_full.tif"
    merged = tmp_path / "merged_full.tif"
    cropped = tmp_path / "merged_cropped.tif"
    vectors = tmp_path / "merged_vectors.gpkg"
    parent_aoi = tmp_path / "aoi.gpkg"

    _write_binary_raster(child_a, np.ones((4, 4), dtype=np.uint8), west=0, north=4)
    _write_binary_raster(child_b, np.ones((4, 4), dtype=np.uint8), west=3, north=4)
    _write_vector(parent_aoi, box(1.1, 0.5, 5.9, 3.5))

    merge_prediction_rasters([str(child_a), str(child_b)], str(merged), method="max")
    crop_raster_to_aoi(str(merged), str(cropped), str(parent_aoi))
    export_vectors(str(cropped), str(vectors), simplify_tolerance=0.0)

    with rasterio.open(merged) as src:
        assert src.width == 7
        assert src.height == 4
        merged_bounds = src.bounds
    with rasterio.open(cropped) as src:
        assert src.read(1).max() == 1
        cropped_bounds = src.bounds

    assert cropped_bounds.left > merged_bounds.left
    assert cropped_bounds.right < merged_bounds.right

    assert vectors.exists()

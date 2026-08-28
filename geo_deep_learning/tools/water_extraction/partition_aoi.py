#!/usr/bin/env python3
"""Partition large AOIs into smaller overlapping child AOIs."""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import fiona
import pyproj
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union
from shapely.validation import make_valid

from geo_deep_learning.tools.water_extraction.download_elevation import (
    estimate_output_pixels,
)

log = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@dataclass
class PartitionRecord:
    """Serializable child AOI partition metadata."""

    child_id: str
    child_name: str
    row: int
    col: int
    source_dir: str
    aoi_path: str
    buffered_aoi_path: str
    core_bounds: list[float]
    buffered_bounds: list[float]
    estimated_pixels: int


def _load_union_geometry(
    vector_path: str | Path,
) -> tuple[BaseGeometry, pyproj.CRS]:
    """Load and dissolve a vector layer into one valid geometry."""
    with fiona.open(vector_path, "r") as src:
        crs = pyproj.CRS.from_user_input(src.crs)
        geometries = [
            make_valid(shape(feature["geometry"]))
            for feature in src
            if feature["geometry"] is not None
        ]

    if not geometries:
        msg = f"No geometries found in {vector_path}"
        raise ValueError(msg)

    merged = make_valid(unary_union(geometries))
    if merged.is_empty:
        msg = f"Unioned geometry is empty for {vector_path}"
        raise ValueError(msg)

    return merged, crs


def _local_metric_crs(geometry: BaseGeometry, crs: pyproj.CRS) -> pyproj.CRS:
    """Return a metric CRS suitable for gridding/buffering."""
    if not crs.is_geographic:
        return crs

    centroid = geometry.centroid
    lon, lat = centroid.x, centroid.y
    utm_zone = int((lon + 180) / 6) + 1
    epsg_prefix = 326 if lat >= 0 else 327
    return pyproj.CRS(f"EPSG:{epsg_prefix}{utm_zone}")


def _build_transformer(
    source_crs: pyproj.CRS,
    target_crs: pyproj.CRS,
) -> pyproj.Transformer:
    """Create a reusable geometry transformer."""
    return pyproj.Transformer.from_crs(
        source_crs,
        target_crs,
        always_xy=True,
    )


def _project_geometry(
    geometry: BaseGeometry,
    *,
    source_crs: pyproj.CRS,
    target_crs: pyproj.CRS,
) -> BaseGeometry:
    """Project a geometry between CRSs."""
    if source_crs == target_crs:
        return geometry

    transformer = _build_transformer(source_crs, target_crs)
    return transform(transformer.transform, geometry)


def _write_geometry(
    output_path: Path,
    geometry: BaseGeometry,
    crs: pyproj.CRS,
    *,
    properties: dict[str, object],
) -> None:
    """Write a single geometry feature to GeoPackage."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = {
        "geometry": geometry.geom_type,
        "properties": {
            key: "str" if isinstance(value, str) else "int"
            for key, value in properties.items()
        },
    }
    with fiona.open(
        output_path,
        "w",
        driver="GPKG",
        crs=crs.to_wkt(),
        schema=schema,
    ) as dst:
        dst.write(
            {
                "geometry": mapping(geometry),
                "properties": properties,
            }
        )


def _serialize_bounds(bounds: tuple[float, float, float, float]) -> list[float]:
    """Convert bounds tuple to stable JSON-friendly floats."""
    return [float(value) for value in bounds]


def _target_core_side_meters(
    *,
    max_pixels: int,
    resolution: int,
    overlap_meters: float,
) -> float:
    """Return a core grid side whose buffered envelope stays within budget."""
    full_side = math.sqrt(max_pixels) * resolution
    core_side = full_side - (2.0 * overlap_meters)
    if core_side <= 0:
        msg = (
            "Configured max_pixels and overlap_meters leave no usable core AOI size. "
            f"full_side={full_side:.2f}m, overlap_meters={overlap_meters:.2f}m"
        )
        raise ValueError(msg)
    return core_side


def partition_aoi(
    *,
    input_aoi: str,
    buffered_aoi: str,
    output_root: str,
    resolution: int,
    max_pixels: int,
    overlap_meters: float,
    child_id_prefix: str = "subaoi",
) -> dict[str, object]:
    """Partition an AOI into smaller overlapping children when needed."""
    if max_pixels <= 0:
        msg = "max_pixels must be positive"
        raise ValueError(msg)
    if resolution <= 0:
        msg = "resolution must be positive"
        raise ValueError(msg)
    if overlap_meters < 0:
        msg = "overlap_meters must be >= 0"
        raise ValueError(msg)

    parent_geom, parent_crs = _load_union_geometry(input_aoi)
    buffered_geom, buffered_crs = _load_union_geometry(buffered_aoi)
    if parent_crs != buffered_crs:
        msg = (
            "input_aoi and buffered_aoi must use the same CRS, got "
            f"{parent_crs.to_string()} and {buffered_crs.to_string()}"
        )
        raise ValueError(msg)

    metric_crs = _local_metric_crs(buffered_geom, parent_crs)
    parent_metric = make_valid(
        _project_geometry(parent_geom, source_crs=parent_crs, target_crs=metric_crs)
    )
    buffered_metric = make_valid(
        _project_geometry(buffered_geom, source_crs=parent_crs, target_crs=metric_crs)
    )

    parent_bounds_metric = parent_metric.bounds
    buffered_bounds_metric = buffered_metric.bounds
    _, _, parent_pixel_count = estimate_output_pixels(
        buffered_bounds_metric,
        resolution,
    )

    manifest_path = Path(output_root) / "partition_manifest.json"
    Path(output_root).mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {
        "parent_aoi_name": Path(input_aoi).resolve().parent.name,
        "input_aoi": str(Path(input_aoi).resolve()),
        "buffered_aoi": str(Path(buffered_aoi).resolve()),
        "output_root": str(Path(output_root).resolve()),
        "resolution": resolution,
        "max_pixels": max_pixels,
        "overlap_meters": overlap_meters,
        "estimated_parent_pixels": parent_pixel_count,
        "estimated_parent_buffered_bounds": _serialize_bounds(buffered_bounds_metric),
        "split_required": parent_pixel_count > max_pixels,
        "children": [],
    }

    if parent_pixel_count <= max_pixels:
        manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    core_side = _target_core_side_meters(
        max_pixels=max_pixels,
        resolution=resolution,
        overlap_meters=overlap_meters,
    )
    minx, miny, maxx, maxy = parent_bounds_metric
    n_cols = max(1, math.ceil((maxx - minx) / core_side))
    n_rows = max(1, math.ceil((maxy - miny) / core_side))

    to_parent_crs = _build_transformer(metric_crs, parent_crs)
    children: list[PartitionRecord] = []

    child_index = 0
    for row in range(n_rows):
        for col in range(n_cols):
            cell_minx = minx + (col * core_side)
            cell_miny = miny + (row * core_side)
            cell_maxx = min(cell_minx + core_side, maxx)
            cell_maxy = min(cell_miny + core_side, maxy)
            cell = box(cell_minx, cell_miny, cell_maxx, cell_maxy)

            child_core_metric = make_valid(parent_metric.intersection(cell))
            if child_core_metric.is_empty or child_core_metric.area <= 0:
                continue

            child_buffered_metric = make_valid(
                child_core_metric.buffer(overlap_meters)
                if overlap_meters > 0
                else child_core_metric
            )
            if child_buffered_metric.is_empty:
                continue

            child_core = make_valid(transform(to_parent_crs.transform, child_core_metric))
            child_buffered = make_valid(
                transform(to_parent_crs.transform, child_buffered_metric)
            )

            child_id = f"{child_id_prefix}_{child_index:04d}"
            child_dir = Path(output_root) / child_id
            child_dir.mkdir(parents=True, exist_ok=True)

            aoi_path = child_dir / "aoi.gpkg"
            buffered_path = child_dir / "aoi_buffered.gpkg"
            properties = {
                "child_id": child_id,
                "row": row,
                "col": col,
            }

            _write_geometry(aoi_path, child_core, parent_crs, properties=properties)
            _write_geometry(
                buffered_path,
                child_buffered,
                parent_crs,
                properties=properties,
            )

            _, _, child_pixels = estimate_output_pixels(
                child_buffered_metric.bounds,
                resolution,
            )
            children.append(
                PartitionRecord(
                    child_id=child_id,
                    child_name=child_id,
                    row=row,
                    col=col,
                    source_dir=str(child_dir.resolve()),
                    aoi_path=str(aoi_path.resolve()),
                    buffered_aoi_path=str(buffered_path.resolve()),
                    core_bounds=_serialize_bounds(child_core_metric.bounds),
                    buffered_bounds=_serialize_bounds(child_buffered_metric.bounds),
                    estimated_pixels=child_pixels,
                )
            )
            child_index += 1

    result["children"] = [asdict(child) for child in children]
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info(
        "Partitioned AOI into %d child AOIs under %s",
        len(children),
        output_root,
    )
    return result


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Partition large buffered AOIs into smaller overlapping AOIs",
    )
    parser.add_argument("--input_aoi", required=True, type=str)
    parser.add_argument("--buffered_aoi", required=True, type=str)
    parser.add_argument("--output_root", required=True, type=str)
    parser.add_argument("--resolution", required=True, type=int)
    parser.add_argument("--max_pixels", required=True, type=int)
    parser.add_argument("--overlap_meters", required=True, type=float)
    parser.add_argument("--child_id_prefix", default="subaoi", type=str)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    result = partition_aoi(
        input_aoi=args.input_aoi,
        buffered_aoi=args.buffered_aoi,
        output_root=args.output_root,
        resolution=args.resolution,
        max_pixels=args.max_pixels,
        overlap_meters=args.overlap_meters,
        child_id_prefix=args.child_id_prefix,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

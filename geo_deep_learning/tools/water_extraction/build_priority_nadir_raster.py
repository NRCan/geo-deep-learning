#!/usr/bin/env python3
"""Build a nadir raster using the same project overwrite priority as intensity."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--nadir-tool", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=5_000_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def enabled_projects(config: dict) -> list[dict]:
    projects = [p for p in config.get("projects", []) if p.get("enabled", True) and p.get("input_dir")]
    if not projects:
        raise RuntimeError("The intensity config has no enabled projects with input_dir values.")
    return projects


def run_project_nadir(args: argparse.Namespace, project: dict, ordinal: int) -> tuple[str, Path, Path]:
    project_name = str(project["name"])
    project_dir = args.output_dir / "by_project" / f"{ordinal:02d}_{safe_name(project_name)}"
    project_dir.mkdir(parents=True, exist_ok=True)
    project_id = f"{args.name}__{safe_name(project_name)}"
    weighted = project_dir / f"{project_id}_nadir_weighted_{int(args.resolution)}m.tif"
    counts = project_dir / f"{project_id}_valid_point_count_{int(args.resolution)}m.tif"
    command = [
        sys.executable, str(args.nadir_tool), "--config", str(args.config),
        "--input-dir", str(project["input_dir"]), "--name", project_id,
        "--output-dir", str(project_dir), "--output-mode", "all",
        "--resolution", str(args.resolution), "--workers", str(args.workers),
        "--chunk-size", str(args.chunk_size),
    ]
    if args.overwrite:
        command.append("--overwrite")
    print(f"[{ordinal}] Building nadir for {project_name}", flush=True)
    subprocess.run(command, check=True)
    if not weighted.is_file() or not counts.is_file():
        raise RuntimeError(f"Nadir outputs missing for {project_name}: {project_dir}")
    return project_name, weighted, counts


def mosaic_in_priority_order(inputs: list[tuple[str, Path, Path]], output: Path) -> None:
    sources = [(name, rasterio.open(weighted), rasterio.open(counts)) for name, weighted, counts in inputs]
    try:
        reference = sources[0][1]
        resolution_x, resolution_y = reference.res
        if not np.isclose(resolution_x, resolution_y):
            raise RuntimeError("Nadir inputs must have square pixels.")
        crs = reference.crs
        left = min(src.bounds.left for _, src, _ in sources)
        bottom = min(src.bounds.bottom for _, src, _ in sources)
        right = max(src.bounds.right for _, src, _ in sources)
        top = max(src.bounds.top for _, src, _ in sources)
        transform = rasterio.transform.from_origin(left, top, resolution_x, resolution_y)
        width = round((right - left) / resolution_x)
        height = round((top - bottom) / resolution_y)
        profile = reference.profile.copy()
        profile.update(height=height, width=width, transform=transform, crs=crs, dtype="float32", nodata=-9999.0, count=1, compress="DEFLATE", tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **profile) as dst:
            for _, window in dst.block_windows(1):
                result = np.full((int(window.height), int(window.width)), -9999.0, dtype=np.float32)
                for project_name, weighted, counts in sources:
                    if weighted.crs != crs or not np.isclose(weighted.res[0], resolution_x):
                        raise RuntimeError(f"Grid mismatch for {project_name}: {weighted.name}")
                    col_off = round((weighted.transform.c - transform.c) / resolution_x)
                    row_off = round((transform.f - weighted.transform.f) / resolution_y)
                    source_window = Window(col_off, row_off, weighted.width, weighted.height)
                    if (
                        window.col_off >= source_window.col_off + source_window.width
                        or source_window.col_off >= window.col_off + window.width
                        or window.row_off >= source_window.row_off + source_window.height
                        or source_window.row_off >= window.row_off + window.height
                    ):
                        continue
                    intersection = window.intersection(source_window)
                    read_window = Window(intersection.col_off - col_off, intersection.row_off - row_off, intersection.width, intersection.height)
                    row_start, col_start = int(intersection.row_off - window.row_off), int(intersection.col_off - window.col_off)
                    row_end, col_end = row_start + int(intersection.height), col_start + int(intersection.width)
                    count_data = counts.read(1, window=read_window)
                    valid = count_data > 0
                    if valid.any():
                        target = result[row_start:row_end, col_start:col_end]
                        weighted_data = weighted.read(1, window=read_window)
                        target[valid] = weighted_data[valid]
                dst.write(result, 1, window=window)
    finally:
        for _, weighted, counts in sources:
            weighted.close()
            counts.close()


def main() -> int:
    args = parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    output = args.output_dir / f"{args.name}_nadir_weighted_{int(args.resolution)}m.tif"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output}")
    inputs = [run_project_nadir(args, project, ordinal) for ordinal, project in enumerate(enabled_projects(config), start=1)]
    print("Mosaicking nadir rasters in intensity config order (later projects overwrite earlier ones).", flush=True)
    mosaic_in_priority_order(inputs, output)
    print(f"Priority nadir written: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

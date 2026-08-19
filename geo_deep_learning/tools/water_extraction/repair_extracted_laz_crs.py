#!/usr/bin/env python3
"""Restore missing CRS metadata on extracted LAZ files from project index metadata."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import geopandas as gpd
import laspy
import pandas as pd
from pyproj import CRS

LOG = logging.getLogger("repair_extracted_laz_crs")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore missing LAS/LAZ CRS metadata using the LiDAR project index. "
            "This is required for projects whose extracted files have coordinates "
            "but no embedded CRS, which otherwise causes downstream reprojection "
            "to silently fail."
        )
    )
    parser.add_argument("--metadata-csv", required=True, help="Extraction metadata CSV.")
    parser.add_argument("--project-index", required=True, help="LiDAR project index GPKG.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing CRS metadata instead of only filling missing CRS.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def _load_project_crs_map(project_index: Path) -> pd.DataFrame:
    gdf = gpd.read_file(project_index, columns=["provider", "project_name", "horizontal_ref_sys_lidar"])
    df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
    df["provider"] = df["provider"].astype(str).str.strip()
    df["project_name"] = df["project_name"].astype(str).str.strip()
    df["horizontal_ref_sys_lidar"] = df["horizontal_ref_sys_lidar"].astype(str).str.strip()
    df = df[df["horizontal_ref_sys_lidar"].ne("")]
    return df.drop_duplicates(subset=["provider", "project_name"], keep="first")


def _read_current_crs(laz_path: Path) -> CRS | None:
    # Open the compressed file header only; avoid reading point data unless repair is needed.
    with laspy.open(str(laz_path)) as reader:
        return reader.header.parse_crs()


def _write_with_crs(laz_path: Path, target_crs: CRS, *, replace_existing: bool) -> tuple[bool, str | None, str]:
    current = _read_current_crs(laz_path)

    if current is not None and not replace_existing:
        return False, str(current), "already_set"

    las = laspy.read(str(laz_path))

    if current is not None and replace_existing:
        las.header.vlrs = [vlr for vlr in las.header.vlrs if vlr.user_id != "LASF_Projection"]

    if las.header.point_format.id >= 6:
        las.header.global_encoding.wkt = 1

    las.header.add_crs(target_crs)

    tmp_path = laz_path.with_suffix(laz_path.suffix + ".tmp")
    las.write(str(tmp_path))
    os.replace(tmp_path, laz_path)

    repaired = _read_current_crs(laz_path)
    if repaired is None:
        raise RuntimeError(f"CRS repair did not persist for {laz_path}")

    return True, str(repaired), "repaired"


def main() -> int:
    args = _parse_args()
    _configure_logging(args.verbose)

    metadata_csv = Path(args.metadata_csv)
    project_index = Path(args.project_index)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Extraction metadata not found: {metadata_csv}")
    if not project_index.exists():
        raise FileNotFoundError(f"Project index not found: {project_index}")

    extraction_df = pd.read_csv(metadata_csv)
    required_cols = {"provider", "project", "laz_path"}
    missing_cols = required_cols - set(extraction_df.columns)
    if missing_cols:
        raise ValueError(
            f"Extraction metadata is missing required columns: {sorted(missing_cols)}"
        )

    extraction_df["provider"] = extraction_df["provider"].astype(str).str.strip()
    extraction_df["project"] = extraction_df["project"].astype(str).str.strip()
    extraction_df["laz_path"] = extraction_df["laz_path"].astype(str).str.strip()

    run_local_paths = [Path(path_str) for path_str in extraction_df["laz_path"].drop_duplicates()]
    invalid_targets = [path for path in run_local_paths if not path.exists()]
    if invalid_targets:
        missing_list = ", ".join(str(path) for path in invalid_targets[:5])
        raise FileNotFoundError(
            "Expected run-local extracted LAZ file(s) are missing: "
            f"{missing_list}"
        )

    LOG.info(
        "CRS repair scope: only run-local extracted LAZ copies listed in %s will be checked.",
        metadata_csv,
    )
    LOG.info("Sitestore source files are never modified by this script.")

    project_crs_df = _load_project_crs_map(project_index)
    merged = extraction_df.merge(
        project_crs_df,
        how="left",
        left_on=["provider", "project"],
        right_on=["provider", "project_name"],
    )

    unresolved = merged[merged["horizontal_ref_sys_lidar"].isna()][["provider", "project"]].drop_duplicates()
    if not unresolved.empty:
        details = ", ".join(
            f"{row.provider}/{row.project}" for row in unresolved.itertuples(index=False)
        )
        raise RuntimeError(f"Could not resolve source CRS for extracted projects: {details}")

    repaired_count = 0
    skipped_count = 0
    verified_count = 0

    rows = list(merged[["laz_path", "horizontal_ref_sys_lidar"]].drop_duplicates().itertuples(index=False))
    total_files = len(rows)

    for index, row in enumerate(rows, start=1):
        laz_path = Path(row.laz_path)
        crs_string = str(row.horizontal_ref_sys_lidar).strip()

        target_crs = CRS.from_user_input(crs_string)
        updated, crs_after, status = _write_with_crs(
            laz_path,
            target_crs,
            replace_existing=args.force,
        )
        verified_count += 1
        if updated:
            repaired_count += 1
            LOG.info("Repaired run-local CRS on %s -> %s", laz_path, crs_after)
        else:
            skipped_count += 1
            LOG.debug("Skipped %s (%s)", laz_path, status)

        if index % 500 == 0 or index == total_files:
            LOG.info(
                "CRS verification progress: %d/%d checked, %d repaired, %d already had CRS",
                index,
                total_files,
                repaired_count,
                skipped_count,
            )

    LOG.info(
        "CRS verification complete: %d checked, %d repaired, %d already had CRS",
        verified_count,
        repaired_count,
        skipped_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

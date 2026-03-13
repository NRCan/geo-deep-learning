#!/usr/bin/env python
"""
Create valid_lidar_mask from LiDAR project index.

This script automates the creation of the valid_lidar_mask.gpkg file by:
1. Finding LiDAR project polygons that intersect with the AOI
2. Dissolving them into a single polygon
3. Clipping to the AOI boundary
4. Saving as valid_lidar_mask.gpkg in the data folder

Usage:
    python -m geo_deep_learning.tools.water_extraction.create_valid_lidar_mask \
        --aoi_folder data/02NB000 \
        --lidar_index /path/to/projet_lidar_infos_detaillees_2.gpkg

    # Use default lidar index path
    python -m geo_deep_learning.tools.water_extraction.create_valid_lidar_mask \
        --aoi_folder data/02NB000
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
)
log = logging.getLogger(__name__)

# Default LiDAR project index path
DEFAULT_LIDAR_INDEX = (
    "/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/"
    "lidar/utils/input_data/index_lidar/projet_lidar_infos_detaillees_2.gpkg"
)


def find_aoi_vector(aoi_folder: Path) -> Path:
    """
    Find the AOI vector file in the data folder.

    Looks for aoi.gpkg or aoi.shp.

    Args:
        aoi_folder: Path to the AOI data folder

    Returns:
        Path to AOI vector file

    Raises:
        FileNotFoundError: If no AOI file is found

    """
    candidates = [
        aoi_folder / "aoi.gpkg",
        aoi_folder / "aoi.shp",
    ]

    for candidate in candidates:
        if candidate.exists():
            log.info("Found AOI file: %s", candidate)
            return candidate

    msg = f"No AOI file found. Checked: {', '.join(str(c) for c in candidates)}"
    raise FileNotFoundError(msg)


def create_valid_lidar_mask(  # noqa: PLR0915
    aoi_path: Path,
    lidar_index_path: Path,
    output_path: Path,
    *,
    save_intermediate: bool = False,
) -> None:
    """
    Create valid_lidar_mask from LiDAR project index.

    Args:
        aoi_path: Path to AOI vector file (gpkg or shp)
        lidar_index_path: Path to LiDAR project index geopackage
        output_path: Where to save valid_lidar_mask.gpkg
        save_intermediate: If True, save intermediate selected polygons

    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If no intersecting LiDAR projects are found

    """
    # Validate inputs
    if not aoi_path.exists():
        msg = f"AOI file not found: {aoi_path}"
        raise FileNotFoundError(msg)

    if not lidar_index_path.exists():
        msg = f"LiDAR index file not found: {lidar_index_path}"
        raise FileNotFoundError(msg)

    log.info("=" * 80)
    log.info("CREATING VALID LIDAR MASK")
    log.info("=" * 80)
    log.info("AOI: %s", aoi_path)
    log.info("LiDAR Index: %s", lidar_index_path)
    log.info("Output: %s", output_path)
    log.info("=" * 80)

    # Step 1: Load AOI
    log.info("Loading AOI...")
    aoi_gdf = gpd.read_file(aoi_path)
    log.info("  - CRS: %s", aoi_gdf.crs)
    log.info("  - Features: %d", len(aoi_gdf))
    log.info("  - Bounds: %s", aoi_gdf.total_bounds)

    # Dissolve AOI if multiple polygons (to get single boundary)
    if len(aoi_gdf) > 1:
        log.info("  - Dissolving %d AOI polygons into one", len(aoi_gdf))
        aoi_dissolved = aoi_gdf.dissolve()
    else:
        aoi_dissolved = aoi_gdf

    # Step 2: Load LiDAR project index
    log.info("")
    log.info("Loading LiDAR project index...")
    lidar_index_gdf = gpd.read_file(lidar_index_path)
    log.info("  - CRS: %s", lidar_index_gdf.crs)
    log.info("  - Total projects: %d", len(lidar_index_gdf))

    # Reproject LiDAR index to match AOI CRS if needed
    if lidar_index_gdf.crs != aoi_gdf.crs:
        log.info(
            "  - Reprojecting LiDAR index from %s to %s",
            lidar_index_gdf.crs,
            aoi_gdf.crs,
        )
        lidar_index_gdf = lidar_index_gdf.to_crs(aoi_gdf.crs)

    # Step 3: Select intersecting LiDAR projects
    log.info("")
    log.info("Selecting LiDAR projects that intersect with AOI...")
    intersecting = lidar_index_gdf[
        lidar_index_gdf.intersects(aoi_dissolved.union_all())
    ].copy()

    if len(intersecting) == 0:
        msg = "No LiDAR projects intersect with the AOI. Check inputs."
        raise ValueError(msg)

    log.info("  - Found %d intersecting LiDAR projects", len(intersecting))

    # Optional: Save intermediate selected polygons
    if save_intermediate:
        intermediate_path = (
            output_path.parent / f"{output_path.stem}_selected_projects.gpkg"
        )
        log.info("  - Saving intermediate selected projects to: %s", intermediate_path)
        intersecting.to_file(intermediate_path, driver="GPKG")

    # Step 4: Dissolve intersecting polygons into one
    log.info("")
    log.info("Dissolving %d intersecting polygons into one...", len(intersecting))
    dissolved = intersecting.dissolve()
    log.info("  - Dissolved into %d feature(s)", len(dissolved))

    # Step 5: Clip to AOI boundary
    log.info("")
    log.info("Clipping dissolved polygon to AOI boundary...")
    clipped = gpd.overlay(dissolved, aoi_dissolved, how="intersection")
    log.info("  - Clipped polygon count: %d", len(clipped))

    if len(clipped) == 0:
        log.warning(
            "Clipping resulted in empty geometry. Using dissolved polygon as-is.",
        )
        final_mask = dissolved
    else:
        final_mask = clipped

    # Ensure single polygon (dissolve again if needed)
    if len(final_mask) > 1:
        log.info("  - Dissolving clipped polygons into single feature...")
        final_mask = final_mask.dissolve()

    # Step 6: Save result
    log.info("")
    log.info("Saving valid LiDAR mask...")
    final_mask.to_file(output_path, driver="GPKG")

    log.info("=" * 80)
    log.info("SUCCESS!")
    log.info("Valid LiDAR mask saved to: %s", output_path)
    log.info("  - Features: %d", len(final_mask))
    log.info("  - CRS: %s", final_mask.crs)
    log.info("  - Total area: %.2f sq units", final_mask.area.sum())
    log.info("=" * 80)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create valid_lidar_mask.gpkg from LiDAR project index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--aoi_folder",
        type=str,
        required=True,
        help="Path to AOI data folder containing aoi.gpkg or aoi.shp",
    )
    parser.add_argument(
        "--lidar_index",
        type=str,
        default=DEFAULT_LIDAR_INDEX,
        help="Path to LiDAR project index geopackage",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="valid_lidar_mask.gpkg",
        help="Output filename (saved in aoi_folder)",
    )
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Save intermediate selected LiDAR projects",
    )

    args = parser.parse_args()

    # Resolve paths
    aoi_folder = Path(args.aoi_folder).resolve()
    lidar_index_path = Path(args.lidar_index).resolve()

    if not aoi_folder.exists():
        msg = f"AOI folder not found: {aoi_folder}"
        raise FileNotFoundError(msg)

    # Find AOI vector file
    aoi_path = find_aoi_vector(aoi_folder)

    # Define output path
    output_path = aoi_folder / args.output_name

    # Check if output already exists
    if output_path.exists():
        log.warning("Output file already exists: %s", output_path)
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            log.info("Aborted by user")
            return

    # Create valid lidar mask
    create_valid_lidar_mask(
        aoi_path=aoi_path,
        lidar_index_path=lidar_index_path,
        output_path=output_path,
        save_intermediate=args.save_intermediate,
    )


if __name__ == "__main__":
    main()

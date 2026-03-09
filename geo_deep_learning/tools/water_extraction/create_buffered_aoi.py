#!/usr/bin/env python3
"""
Create a buffered version of an AOI geopackage.

This utility creates an expanded version of your AOI by buffering it outward
by a specified distance. Useful for:
- Downloading DTM/DSM with extra context around boundaries
- Preventing edge effects in processing and predictions

Usage:
    python -m geo_deep_learning.tools.water_extraction.create_buffered_aoi \
        --input_aoi data/02NB000/aoi.gpkg \
        --output_aoi data/02NB000/aoi_buffered.gpkg \
        --buffer_meters 1000

    # Or use short form:
    python -m geo_deep_learning.tools.water_extraction.create_buffered_aoi \
        -i data/02NB000/aoi.gpkg \
        -o data/02NB000/aoi_buffered.gpkg \
        -b 1000
"""

import argparse
import logging
import sys
from pathlib import Path

import fiona
import pyproj
from shapely.geometry import shape
from shapely.ops import transform

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def create_buffered_aoi(  # noqa: PLR0915
    input_aoi_path: str,
    output_aoi_path: str,
    buffer_meters: float,
) -> None:
    """
    Create a buffered version of an AOI geopackage.

    Args:
        input_aoi_path: Path to original AOI geopackage
        output_aoi_path: Path to save buffered AOI geopackage
        buffer_meters: Buffer distance in meters to expand AOI outward

    Raises:
        FileNotFoundError: If input AOI does not exist
        ValueError: If buffer_meters is negative

    """
    if buffer_meters < 0:
        msg = f"Buffer distance must be >= 0, got {buffer_meters}"
        raise ValueError(msg)

    input_path = Path(input_aoi_path)
    if not input_path.exists():
        msg = f"Input AOI file not found: {input_aoi_path}"
        raise FileNotFoundError(msg)

    log.info("=" * 70)
    log.info("Creating Buffered AOI")
    log.info("=" * 70)
    log.info("Input AOI:      %s", input_aoi_path)
    log.info("Output AOI:     %s", output_aoi_path)
    log.info(
        "Buffer distance: %s meters (%.2f km)",
        buffer_meters,
        buffer_meters / 1000,
    )
    log.info("-" * 70)

    # Read original AOI
    with fiona.open(input_aoi_path, "r") as src:
        aoi_crs = src.crs
        schema = src.schema.copy()
        features = list(src)
        num_features = len(features)

    log.info("Read %d feature(s) from input AOI", num_features)
    log.info("CRS: %s", aoi_crs)

    # Buffer geometries
    buffered_features = []
    for idx, feature in enumerate(features, 1):
        geom_dict = feature["geometry"]
        geom = shape(geom_dict)

        log.info("Processing feature %d/%d...", idx, num_features)

        # Check if CRS is geographic (lat/lon)
        if aoi_crs and (
            "epsg:4326" in str(aoi_crs).lower()
            or "+proj=longlat" in str(aoi_crs).lower()
        ):
            # Need to project to metric CRS for buffering
            log.info("  Geographic CRS detected - projecting to UTM for buffering")

            # Use UTM zone based on centroid
            centroid = geom.centroid
            lon, lat = centroid.x, centroid.y
            utm_zone = int((lon + 180) / 6) + 1
            hemisphere = "north" if lat >= 0 else "south"
            epsg_prefix = 326 if hemisphere == "north" else 327
            utm_epsg = f"EPSG:{epsg_prefix}{utm_zone}"
            utm_crs = pyproj.CRS(utm_epsg)

            log.info("  Using %s for metric buffering", utm_epsg)

            # Transform to UTM, buffer, transform back
            project_to_utm = pyproj.Transformer.from_crs(
                aoi_crs,
                utm_crs,
                always_xy=True,
            ).transform
            project_from_utm = pyproj.Transformer.from_crs(
                utm_crs,
                aoi_crs,
                always_xy=True,
            ).transform

            geom_utm = transform(project_to_utm, geom)
            buffered_utm = geom_utm.buffer(buffer_meters)
            buffered_geom = transform(project_from_utm, buffered_utm)

            # Calculate area increase
            original_area_km2 = geom_utm.area / 1_000_000
            buffered_area_km2 = buffered_utm.area / 1_000_000
            area_increase = buffered_area_km2 - original_area_km2

            log.info("  Original area: %.2f km²", original_area_km2)
            log.info("  Buffered area: %.2f km²", buffered_area_km2)
            log.info(
                "  Area increase: %.2f km² (+%.1f%%)",
                area_increase,
                (area_increase / original_area_km2) * 100,
            )

        else:
            # CRS is already in meters (projected), buffer directly
            log.info("  Projected CRS detected - buffering directly")
            buffered_geom = geom.buffer(buffer_meters)

            # Calculate area increase
            original_area_km2 = geom.area / 1_000_000
            buffered_area_km2 = buffered_geom.area / 1_000_000
            area_increase = buffered_area_km2 - original_area_km2

            log.info("  Original area: %.2f km²", original_area_km2)
            log.info("  Buffered area: %.2f km²", buffered_area_km2)
            log.info(
                "  Area increase: %.2f km² (+%.1f%%)",
                area_increase,
                (area_increase / original_area_km2) * 100,
            )

        # Create new feature with buffered geometry
        buffered_feature = dict(feature)
        buffered_feature["geometry"] = buffered_geom.__geo_interface__
        buffered_features.append(buffered_feature)

    # Ensure output directory exists
    output_path = Path(output_aoi_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write buffered AOI
    with fiona.open(
        output_aoi_path,
        "w",
        driver="GPKG",
        crs=aoi_crs,
        schema=schema,
    ) as dst:
        dst.writerecords(buffered_features)

    log.info("-" * 70)
    log.info("✓ Buffered AOI saved: %s", output_aoi_path)
    log.info("=" * 70)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a buffered version of an AOI geopackage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create AOI with 1 km buffer
  python -m geo_deep_learning.tools.water_extraction.create_buffered_aoi \\
      --input_aoi data/02NB000/aoi.gpkg \\
      --output_aoi data/02NB000/aoi_buffered.gpkg \\
      --buffer_meters 1000

  # Create AOI with 2.5 km buffer using short flags
  python -m geo_deep_learning.tools.water_extraction.create_buffered_aoi \\
      -i data/02NB000/aoi.gpkg \\
      -o data/02NB000/aoi_buffered.gpkg \\
      -b 2500

  # Use in download pipeline
  python -m geo_deep_learning.tools.water_extraction.create_buffered_aoi \\
      -i data/02NB000/aoi.gpkg \\
      -o data/02NB000/aoi_buffered.gpkg \\
      -b 1000

  python -m geo_deep_learning.tools.water_extraction.download_elevation \\
      --aoi_path data/02NB000/aoi_buffered.gpkg
        """,
    )
    parser.add_argument(
        "-i",
        "--input_aoi",
        type=str,
        required=True,
        help="Path to input AOI geopackage (.gpkg) or shapefile (.shp)",
    )
    parser.add_argument(
        "-o",
        "--output_aoi",
        type=str,
        required=True,
        help="Path to save buffered AOI geopackage",
    )
    parser.add_argument(
        "-b",
        "--buffer_meters",
        type=float,
        required=True,
        help="Buffer distance in meters (e.g., 1000 for 1km)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    try:
        create_buffered_aoi(
            input_aoi_path=args.input_aoi,
            output_aoi_path=args.output_aoi,
            buffer_meters=args.buffer_meters,
        )
        sys.exit(0)
    except (FileNotFoundError, ValueError, OSError):
        log.exception("Failed to create buffered AOI")
        sys.exit(1)


if __name__ == "__main__":
    main()

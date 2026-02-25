r"""
Seam correction for LiDAR mosaic boundary artifacts in derived rasters.

Usage:
    python -m geo_deep_learning.tools.water_extraction.seam_correction \
        --input path/to/twi.tif \
        --project_extents path/to/project_boundaries.gpkg \
        [--output path/to/twi_seam_corrected.tif] \
        [--buffer_pixels 15] \
        [--seam_width_pixels 3] \
        [--sigma 5.0] \
        [--chunk_rows 512]
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
from pathlib import Path

import fiona
import numpy as np
import rasterio
import rasterio.windows
from fiona.transform import transform_geom
from rasterio.features import rasterize
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import box as shapely_box
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

_MIN_SMOOTH_WEIGHT: float = 1e-6


def _extract_boundary_lines(gpkg_path: str, target_crs: object) -> object | None:
    """
    Extract the boundary edges of all project extent polygons.

    Each polygon boundary represents a lidar acquisition project edge — the
    location where the raster mosaic may have a seam artifact. All polygon
    boundaries that overlap with the raster extent are used as correction
    centerlines, regardless of whether polygons are adjacent to each other.

    Args:
        gpkg_path: Path to GeoPackage containing one polygon per project extent.
        target_crs: Rasterio CRS object; geometries are reprojected to this CRS.

    Returns:
        Unified Shapely geometry of all polygon boundary lines, or None if the
        GeoPackage contains no valid geometries.

    """
    crs_str = str(target_crs)
    boundaries: list = []

    with fiona.open(gpkg_path) as src:
        src_crs = src.crs
        for feat in src:
            if feat["geometry"] is None:
                continue
            geom = shape(feat["geometry"])
            if src_crs != target_crs:
                geom = shape(transform_geom(src_crs, crs_str, mapping(geom)))
            if not geom.is_valid:
                geom = make_valid(geom)
            boundaries.append(geom.boundary)

    if not boundaries:
        log.warning("[SEAM] No valid geometries found in %s.", gpkg_path)
        return None

    log.info("[SEAM] Loaded %d boundary line(s) from %s.", len(boundaries), gpkg_path)
    return unary_union(boundaries)


def _gaussian_smooth_nodata_safe(
    band: np.ndarray,
    valid_mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Apply Gaussian smoothing while treating NoData pixels as absent.

    Uses a normalised weighted Gaussian:
        smoothed = G(band * valid) / G(valid)
    This prevents NoData regions from influencing the output at valid pixels.

    Args:
        band: 2-D float32 array.
        valid_mask: Boolean array, True where data is valid (not NoData).
        sigma: Gaussian standard deviation in pixels.

    Returns:
        Smoothed float32 array.

    """
    weights = valid_mask.astype(np.float64)
    data_w = band.astype(np.float64) * weights
    smooth_data_w = gaussian_filter(data_w, sigma=sigma)
    smooth_weights = gaussian_filter(weights, sigma=sigma)
    # Replace near-zero weights with 1.0 before dividing to avoid RuntimeWarning;
    # np.where selects band values for those pixels so the result is unchanged.
    safe_weights = np.where(smooth_weights > _MIN_SMOOTH_WEIGHT, smooth_weights, 1.0)
    smoothed = np.where(
        smooth_weights > _MIN_SMOOTH_WEIGHT,
        smooth_data_w / safe_weights,
        band.astype(np.float64),
    )
    return smoothed.astype(np.float32)


def correct_seams(  # noqa: PLR0913, PLR0915
    input_path: str,
    output_path: str,
    project_extents_path: str,
    *,
    buffer_pixels: int = 15,
    seam_width_pixels: int = 3,
    gaussian_sigma: float = 5.0,
    chunk_rows: int = 512,
) -> None:
    """
    Apply seam correction to a LiDAR-derived raster using strip-based processing.

    When multiple LiDAR acquisition projects are mosaicked, shared boundaries
    create 2-5 pixel linear artifacts in derived rasters (TWI, slope, etc.)
    that are falsely detected as water. This function removes those artifacts
    by inpainting: pixels within ``seam_width_pixels`` of each seam centerline
    are excluded from the Gaussian source and replaced by values interpolated
    from valid data on both sides. The result is blended back using a linear
    taper over ``buffer_pixels`` so the correction is invisible. nodata pixels
    are never modified.

    Correction procedure:
      1. The boundary edge of every polygon in ``project_extents_path`` is
         extracted. Each edge marks a lidar acquisition project limit where a
         seam artifact may appear in the raster.
      2. The raster is processed in horizontal strips of ``chunk_rows`` rows.
         Each strip is read with a halo of max(buffer_pixels, 3*sigma) rows so
         that the Euclidean distance transform and Gaussian filter are accurate
         at strip boundaries.
      3. For each strip a Euclidean distance map is computed from the seam
         centerline. Pixels within ``seam_width_pixels`` form the inpainting zone.
      4. A nodata-safe Gaussian is applied with the inpainting zone masked out
         so the kernel draws only from clean data on both sides of the seam.
      5. A linear blend weight (1 at the centerline, 0 at ``buffer_pixels``)
         blends inpainted values back:
         output = weight * inpainted + (1 - weight) * original.
      6. Pixels flagged as nodata are written back unchanged.
      7. Strips with no seam lines are copied directly without processing.

    Args:
        input_path: Path to the input raster (single or multi-band GeoTIFF).
        output_path: Path for the corrected output raster.
        project_extents_path: Path to a GeoPackage with one polygon per LiDAR
            project extent. Polygon boundary edges define seam locations.
        buffer_pixels: Half-width of the blend taper zone in pixels (default 15).
        seam_width_pixels: Half-width of the inpainting zone around the seam
            centerline in pixels (default 3). Pixels within this distance are
            treated as missing and filled by Gaussian interpolation.
        gaussian_sigma: Standard deviation of the Gaussian blur kernel in pixels
            (default 5.0). Should be at least ``seam_width_pixels`` so the kernel
            bridges the inpainting zone from both sides.
        chunk_rows: Number of core rows processed per strip (default 512).
            Lower values reduce peak memory at the cost of more I/O passes.

    """
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        crs = src.crs
        transform = src.transform
        nodata_vals = src.nodatavals
        height = src.height
        width = src.width
        band_count = src.count
        bounds = src.bounds

    log.info(
        "[SEAM] Input: %s (%d band(s), %dx%d px)",
        input_path,
        band_count,
        width,
        height,
    )

    # --- Step 1: extract seam geometry ---
    log.info("[SEAM] Extracting polygon boundaries from: %s", project_extents_path)
    seam_geom = _extract_boundary_lines(project_extents_path, crs)

    if seam_geom is None or seam_geom.is_empty:
        log.warning("[SEAM] No boundary lines found. Writing input raster unchanged.")
        shutil.copy2(input_path, output_path)
        return

    # Quick geometric check before touching any raster data
    if not seam_geom.intersects(shapely_box(*bounds)):
        log.warning(
            "[SEAM] No seam lines intersect raster extent. "
            "Writing input raster unchanged.",
        )
        shutil.copy2(input_path, output_path)
        return

    # --- Step 2: set up output and strip parameters ---
    # Halo ensures accurate EDT and Gaussian at strip boundaries.
    # EDT needs buffer_pixels rows of context; Gaussian needs ~3*sigma rows.
    halo = max(buffer_pixels, math.ceil(3.0 * gaussian_sigma))

    profile.update(
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="lzw",
        BIGTIFF="IF_SAFER",
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    n_strips = math.ceil(height / chunk_rows)
    log.info(
        "[SEAM] Processing %d strip(s) of %d core rows each "
        "(halo=%d px, seam_width=%d px, sigma=%.1f).",
        n_strips,
        chunk_rows,
        halo,
        seam_width_pixels,
        gaussian_sigma,
    )

    # --- Step 3: strip-by-strip processing ---
    with (
        rasterio.open(input_path) as src,
        rasterio.open(output_path, "w", **profile) as dst,
    ):
        for strip_idx, core_row_start in enumerate(range(0, height, chunk_rows)):
            core_row_end = min(core_row_start + chunk_rows, height)
            core_rows = core_row_end - core_row_start

            # Halo boundaries, clamped to raster extent
            read_row_start = max(0, core_row_start - halo)
            read_row_end = min(height, core_row_end + halo)
            halo_top = core_row_start - read_row_start
            strip_height = read_row_end - read_row_start

            log.info(
                "[SEAM] Strip %d/%d (rows %d-%d)",
                strip_idx + 1,
                n_strips,
                core_row_start,
                core_row_end,
            )

            strip_window = Window(0, read_row_start, width, strip_height)
            strip_transform = rasterio.windows.transform(strip_window, transform)
            core_window = Window(0, core_row_start, width, core_rows)

            # Rasterize seam lines for this strip only (no full-raster mask)
            seam_strip = rasterize(
                [(mapping(seam_geom), 1)],
                out_shape=(strip_height, width),
                transform=strip_transform,
                fill=0,
                dtype=np.uint8,
            )

            # If no seam falls in this strip, copy core rows directly
            if seam_strip.max() == 0:
                dst.write(src.read(window=core_window), window=core_window)
                continue

            # EDT, inpainting zone, and blend weight for this strip
            dist_strip = distance_transform_edt(seam_strip == 0).astype(np.float32)
            # Pixels within seam_width_pixels of the centerline are inpainted
            seam_zone = dist_strip <= seam_width_pixels
            weight_strip = np.clip(
                1.0 - dist_strip / buffer_pixels,
                0.0,
                1.0,
            ).astype(np.float32)

            # Read band data for this strip (with halo)
            data_strip = src.read(window=strip_window)  # (C, strip_H, W)
            corrected_strip = data_strip.copy()

            for band_idx in range(band_count):
                band = data_strip[band_idx].astype(np.float32)
                nodata = nodata_vals[band_idx]
                if nodata is not None:
                    valid_mask = band != np.float32(nodata)
                else:
                    valid_mask = np.isfinite(band)

                # Exclude seam-zone pixels from the Gaussian source so the kernel
                # fills them by interpolating from clean data on both sides.
                smoothed = _gaussian_smooth_nodata_safe(
                    band,
                    valid_mask & ~seam_zone,
                    sigma=gaussian_sigma,
                )

                w = weight_strip * valid_mask.astype(np.float32)
                blended = w * smoothed + (1.0 - w) * band
                corrected_strip[band_idx] = np.where(
                    valid_mask,
                    blended,
                    band,
                ).astype(data_strip.dtype)

            # Write only the core rows (discard halo)
            dst.write(
                corrected_strip[:, halo_top : halo_top + core_rows, :],
                window=core_window,
            )

    log.info("[SEAM] Seam correction complete. Output: %s", output_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Apply seam correction to a LiDAR-derived raster by feathered "
            "Gaussian smoothing along project boundary seams."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input GeoTIFF (e.g. twi.tif, ndsm.tif, stacked_inputs.tif)",
    )
    parser.add_argument(
        "--project_extents",
        required=True,
        help="Path to GeoPackage with one polygon per LiDAR project extent",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output path. Defaults to <input_stem>_seam_corrected.tif "
            "in the same directory as the input."
        ),
    )
    parser.add_argument(
        "--buffer_pixels",
        type=int,
        default=15,
        help="Half-width of the blend taper zone in pixels",
    )
    parser.add_argument(
        "--seam_width_pixels",
        type=int,
        default=3,
        help=(
            "Half-width of the inpainting zone around the seam centerline in pixels. "
            "Pixels within this distance are excluded from the Gaussian source and "
            "filled by interpolation from both sides."
        ),
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=5.0,
        help=(
            "Gaussian blur sigma in pixels. "
            "Should be >= seam_width_pixels for effective inpainting."
        ),
    )
    parser.add_argument(
        "--chunk_rows",
        type=int,
        default=512,
        help="Rows processed per strip; lower values reduce peak RAM",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        msg = f"Input raster not found: {input_path}"
        raise FileNotFoundError(msg)

    extents_path = Path(args.project_extents).resolve()
    if not extents_path.exists():
        msg = f"Project extents GeoPackage not found: {extents_path}"
        raise FileNotFoundError(msg)

    if args.output is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_seam_corrected{input_path.suffix}",
        )
    else:
        output_path = Path(args.output).resolve()

    correct_seams(
        input_path=str(input_path),
        output_path=str(output_path),
        project_extents_path=str(extents_path),
        buffer_pixels=args.buffer_pixels,
        seam_width_pixels=args.seam_width_pixels,
        gaussian_sigma=args.sigma,
        chunk_rows=args.chunk_rows,
    )


if __name__ == "__main__":
    main()

r"""
Seam correction for LiDAR mosaic boundary artifacts in derived rasters.

Usage:
    python -m geo_deep_learning.tools.water_extraction.seam_correction \
        --input path/to/twi.tif \
        --project_extents path/to/project_boundaries.gpkg \
        [--output path/to/twi_seam_corrected.tif] \
        [--buffer_pixels 15] \
        [--sigma 3.0]
"""

from __future__ import annotations

import argparse
import logging
import shutil
from itertools import combinations
from pathlib import Path

import fiona
import numpy as np
import rasterio
from fiona.transform import transform_geom
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

_MIN_POLYGON_COUNT: int = 2
_MIN_SMOOTH_WEIGHT: float = 1e-6
_LINEAR_GEOM_TYPES: tuple[str, ...] = ("LineString", "MultiLineString")


def _collect_linear_parts(geom: object) -> list:
    """Return a list of linear geometry parts extracted from an intersection result."""
    if geom.geom_type in _LINEAR_GEOM_TYPES:
        return [geom]
    if geom.geom_type == "GeometryCollection":
        return [part for part in geom.geoms if part.geom_type in _LINEAR_GEOM_TYPES]
    return []


def _extract_seam_lines(gpkg_path: str, target_crs: object) -> object | None:
    """
    Extract interior seam lines from touching project boundary polygons.

    Computes pairwise intersections between all polygon pairs. Where adjacent
    polygons share a boundary edge, the intersection is a LineString - the seam.

    Args:
        gpkg_path: Path to GeoPackage containing one polygon per project extent.
        target_crs: Rasterio CRS object; geometries are reprojected to this CRS.

    Returns:
        Unified Shapely geometry of all seam lines, or None if no seams found.

    """
    crs_str = str(target_crs)
    polygons: list = []

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
            polygons.append(geom)

    if len(polygons) < _MIN_POLYGON_COUNT:
        log.warning(
            "[SEAM] Found %d polygon(s) in %s; need at least 2 to form a seam.",
            len(polygons),
            gpkg_path,
        )
        return None

    seam_parts: list = []
    for p1, p2 in combinations(polygons, 2):
        if not p1.intersects(p2):
            continue
        inter = p1.intersection(p2)
        if not inter.is_empty:
            seam_parts.extend(_collect_linear_parts(inter))

    if not seam_parts:
        log.warning("[SEAM] No shared linear boundaries found between polygon pairs.")
        return None

    return unary_union(seam_parts)


def _build_blend_weight(
    seam_geom: object,
    height: int,
    width: int,
    transform: object,
    buffer_pixels: int,
) -> np.ndarray:
    """
    Build a distance-based feathering weight array from seam line geometry.

    Args:
        seam_geom: Shapely geometry of the seam lines.
        height: Raster height in pixels.
        width: Raster width in pixels.
        transform: Rasterio affine transform.
        buffer_pixels: Half-width of correction zone; weight reaches 0 at this
            distance from the seam centerline.

    Returns:
        Float32 array of shape (height, width). Value 1 at the seam centerline,
        tapering linearly to 0 at ``buffer_pixels`` distance, 0 beyond.

    """
    seam_raster = rasterize(
        [(mapping(seam_geom), 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    # Euclidean distance (pixels) from each cell to the nearest seam pixel
    dist = distance_transform_edt(seam_raster == 0).astype(np.float32)
    # Linear taper: 1 at seam, 0 at buffer_pixels away, clipped to [0, 1]
    weight = np.clip(1.0 - dist / buffer_pixels, 0.0, 1.0)
    return weight.astype(np.float32)


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
    smoothed = np.where(
        smooth_weights > _MIN_SMOOTH_WEIGHT,
        smooth_data_w / smooth_weights,
        band.astype(np.float64),
    )
    return smoothed.astype(np.float32)


def correct_seams(
    input_path: str,
    output_path: str,
    project_extents_path: str,
    *,
    buffer_pixels: int = 15,
    gaussian_sigma: float = 3.0,
) -> None:
    """
    Apply seam correction to a LiDAR-derived raster.

    When multiple LiDAR acquisition projects are mosaicked, shared boundaries
    create 2-5 pixel linear artifacts in derived rasters (TWI, slope, etc.)
    that are falsely detected as water. This function suppresses those artifacts
    by applying a spatially feathered Gaussian smoothing pass restricted to a
    buffer around the detected seam centerlines. NoData pixels are never modified.

    Correction procedure:
      1. Shared boundary edges are extracted from adjacent project extent polygons
         in ``project_extents_path``.
      2. A distance-based blend weight is computed for each pixel: weight=1 at
         the seam centerline, tapering linearly to 0 at ``buffer_pixels`` away.
      3. A NoData-safe Gaussian-smoothed copy of each band is blended with the
         original using that weight:
         output = weight * smoothed + (1 - weight) * original.
      4. Pixels flagged as NoData are written back unchanged.

    Args:
        input_path: Path to the input raster (single or multi-band GeoTIFF).
        output_path: Path for the corrected output raster.
        project_extents_path: Path to a GeoPackage with one polygon per LiDAR
            project extent. Touching polygon boundaries define seam locations.
        buffer_pixels: Half-width of the correction zone in pixels (default 15).
        gaussian_sigma: Standard deviation of the Gaussian blur kernel in pixels
            (default 3.0).

    """
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        crs = src.crs
        transform = src.transform
        nodata_vals = src.nodatavals  # per-band tuple, values may be None
        data = src.read()  # (C, H, W), original dtype

    height, width = profile["height"], profile["width"]
    log.info(
        "[SEAM] Input: %s  (%d band(s), %dx%d px)",
        input_path,
        data.shape[0],
        width,
        height,
    )

    # --- Step 1: extract seam geometry from project boundary polygons ---
    log.info("[SEAM] Extracting seam lines from: %s", project_extents_path)
    seam_geom = _extract_seam_lines(project_extents_path, crs)

    if seam_geom is None or seam_geom.is_empty:
        log.warning("[SEAM] No seam lines detected. Writing input raster unchanged.")
        shutil.copy2(input_path, output_path)
        return

    # --- Step 2: build feathering weight mask ---
    log.info(
        "[SEAM] Building blend weight mask (buffer=%d px, sigma=%.1f)",
        buffer_pixels,
        gaussian_sigma,
    )
    blend_weight = _build_blend_weight(
        seam_geom,
        height,
        width,
        transform,
        buffer_pixels,
    )

    if blend_weight.max() == 0.0:
        log.warning(
            "[SEAM] Seam mask is empty after rasterisation "
            "(seam lies outside raster extent). Writing input raster unchanged.",
        )
        shutil.copy2(input_path, output_path)
        return

    log.info(
        "[SEAM] Correction zone covers %.2f%% of raster pixels.",
        100.0 * np.sum(blend_weight > 0) / blend_weight.size,
    )

    # --- Step 3: per-band feathered Gaussian correction ---
    corrected = data.copy()
    for band_idx in range(data.shape[0]):
        band = data[band_idx].astype(np.float32)
        nodata = nodata_vals[band_idx]

        valid_mask = (
            band != np.float32(nodata) if nodata is not None else np.isfinite(band)
        )

        smoothed = _gaussian_smooth_nodata_safe(band, valid_mask, sigma=gaussian_sigma)

        # Blend weight is 0 where nodata, so nodata pixels are preserved
        w = blend_weight * valid_mask.astype(np.float32)
        blended = w * smoothed + (1.0 - w) * band

        # Restore original nodata values exactly
        corrected[band_idx] = np.where(valid_mask, blended, band).astype(data.dtype)

    # --- Step 4: write output ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    log.info("[SEAM] Writing corrected raster: %s", output_path)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(corrected)

    log.info("[SEAM] Seam correction complete.")


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
        help="Half-width of the seam correction zone in pixels",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Gaussian blur sigma in pixels",
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
        gaussian_sigma=args.sigma,
    )


if __name__ == "__main__":
    main()

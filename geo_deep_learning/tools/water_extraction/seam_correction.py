r"""
Seam correction for LiDAR mosaic boundary artifacts in derived rasters.

Usage:
    python -m geo_deep_learning.tools.water_extraction.seam_correction \
        --input path/to/twi.tif \
        --project_extents path/to/project_boundaries.gpkg \
        [--output path/to/twi_seam_corrected.tif] \
        [--buffer_pixels 3] \
        [--seam_width_pixels 1] \
        [--nodata_fill_pixels 5] \
        [--sigma_color 0.05] \
        [--sigma_spatial 2] \
        [--blend_sigma 0.0] \
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
from skimage.restoration import denoise_bilateral

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
        Tuple of (smoothed float32 array, smooth_weights float64 array).
        smooth_weights > _MIN_SMOOTH_WEIGHT indicates pixels where the
        Gaussian had enough valid source data to produce a reliable estimate.

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
    return smoothed.astype(np.float32), smooth_weights


def _bilateral_inpaint_nodata_safe(
    band: np.ndarray,
    valid_mask: np.ndarray,
    seam_zone: np.ndarray,
    sigma_color: float,
    sigma_spatial: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply bilateral filter for seam inpainting, excluding seam-zone and nodata pixels.

    Before filtering, pixels excluded from the source (seam zone + nodata) are
    filled with the value of their nearest valid source neighbour so the bilateral
    range kernel sees plausible values and does not produce edge artefacts from
    extreme nodata fill values.  The filter then smooths across the seam zone
    while preserving real terrain edges.

    Args:
        band: 2-D float32 patch around the seam buffer zone.
        valid_mask: Boolean array, True where data is valid (not nodata).
        seam_zone: Boolean array, True within seam_width_pixels of centerline.
        sigma_color: Bilateral range-kernel sigma in data units.  Controls how
            aggressively the filter smooths across value differences; smaller
            values preserve more edges.
        sigma_spatial: Bilateral spatial-kernel sigma in pixels.  Controls
            neighbourhood reach; must be large enough to bridge the seam zone.

    Returns:
        Tuple of (filtered float32 array, smooth_weights float64 array).
        smooth_weights > _MIN_SMOOTH_WEIGHT indicates pixels with enough valid
        source data nearby to produce a reliable estimate (used for nodata fill).

    """
    source_mask = valid_mask & ~seam_zone

    if source_mask.any():
        # Nearest-neighbour fill: non-source pixels get the value of the
        # closest source pixel so the bilateral range kernel is not confused
        # by extreme nodata sentinel values.
        near_idx = distance_transform_edt(
            ~source_mask,
            return_distances=False,
            return_indices=True,
        )
        work = band[near_idx[0], near_idx[1]].astype(np.float64)
    else:
        work = band.astype(np.float64)

    filtered = denoise_bilateral(
        work,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        channel_axis=None,
    )

    # Gaussian on the source mask serves as a proxy for "enough valid source
    # data nearby" — used downstream to decide whether nodata pixels may be
    # filled.  Bilateral does not expose internal kernel weights directly.
    smooth_weights = gaussian_filter(
        source_mask.astype(np.float64),
        sigma=sigma_spatial,
    )

    return filtered.astype(np.float32), smooth_weights


def correct_seams(  # noqa: PLR0913, PLR0915
    input_path: str,
    output_path: str,
    project_extents_path: str,
    *,
    buffer_pixels: int = 3,
    seam_width_pixels: int = 1,
    nodata_fill_pixels: int = 5,
    sigma_color: float = 0.05,
    sigma_spatial: float = 2.0,
    blend_sigma: float = 0.0,
    chunk_rows: int = 512,
) -> None:
    """
    Apply seam correction to a LiDAR-derived raster using strip-based processing.

    When multiple LiDAR acquisition projects are mosaicked, shared boundaries
    create 2-5 pixel linear artifacts in derived rasters (TWI, slope, etc.)
    that are falsely detected as water. This function removes those artifacts
    by two complementary passes: (1) nodata filling — nodata pixels within
    ``nodata_fill_pixels`` of a seam are replaced by Gaussian interpolation
    from valid neighbors; (2) seamless blending — valid artifact pixels within
    ``seam_width_pixels`` are excluded from the Gaussian source so the kernel
    draws from clean data on both sides, then the result is faded in over
    ``buffer_pixels``. nodata pixels outside the fill zone are never modified.

    Correction procedure:
      1. The boundary edge of every polygon in ``project_extents_path`` is
         extracted. Each edge marks a lidar acquisition project limit where a
         seam artifact may appear in the raster.
      2. The raster is processed in horizontal strips of ``chunk_rows`` rows.
         Each strip is read with a halo of max(buffer_pixels, 3*sigma) rows so
         that the Euclidean distance transform and Gaussian filter are accurate
         at strip boundaries.
      3. For each strip a Euclidean distance map is computed from the seam
         centerline. Two zones are derived:
         - inpainting zone (``seam_width_pixels``): valid artifact pixels excluded
           from the Gaussian source so it interpolates from clean data.
         - fill zone (``nodata_fill_pixels``): nodata pixels replaced by the
           Gaussian estimate when the kernel has enough valid neighbors.
      4. A bilateral filter (``skimage.restoration.denoise_bilateral``) is
         applied to a padded bounding box around the seam buffer zone only.
         Seam-zone and nodata pixels are nearest-neighbour filled before
         filtering so the range kernel is not biased by sentinel values.
      5. A cosine taper weight (1 at the centerline, 0 at ``buffer_pixels``)
         blends values back for valid pixels. Inside the inpainting zone the
         bilateral result is used; outside it a lightly smoothed copy
         (``blend_sigma``) is used to preserve local texture:
         output = weight * source + (1 - weight) * original.
      6. nodata pixels inside the fill zone with a reliable Gaussian estimate
         are written as filled; all other nodata pixels are written unchanged.
      7. Strips with no seam lines are copied directly without processing.

    Args:
        input_path: Path to the input raster (single or multi-band GeoTIFF).
        output_path: Path for the corrected output raster.
        project_extents_path: Path to a GeoPackage with one polygon per LiDAR
            project extent. Polygon boundary edges define seam locations.
        buffer_pixels: Half-width of the blend taper zone in pixels (default 15).
        seam_width_pixels: Half-width of the inpainting zone for valid artifact
            pixels in pixels (default 3). These pixels are excluded from the
            Gaussian source so clean data from both sides fills the gap.
        nodata_fill_pixels: Half-width of the nodata fill zone in pixels
            (default 5). nodata pixels within this distance of the seam
            centerline are filled by Gaussian interpolation from valid neighbors.
            Keep this narrow (a few pixels) to avoid filling legitimate nodata.
        sigma_color: Bilateral range-kernel sigma in data units (default 0.05).
            Smaller values preserve more terrain edges; larger values smooth
            more aggressively across value differences.  Tune to the value
            range of your input raster (e.g. nDSM metres, TWI dimensionless).
        sigma_spatial: Bilateral spatial-kernel sigma in pixels (default 2.0).
            Must be large enough to bridge the ``seam_width_pixels`` gap.
            Also controls the halo size added around each strip.
        blend_sigma: Standard deviation of the Gaussian used only for valid
            pixels in the taper zone outside the inpainting zone (default 1.0).
            Keep small (0-2) to preserve local texture and avoid a visible smudge.
            Set to 0 to skip any smoothing in the taper zone entirely.
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
    # Halo ensures accurate EDT and bilateral at strip boundaries.
    # EDT needs buffer_pixels rows of context; bilateral needs ~3*sigma_spatial rows.
    halo = max(buffer_pixels, math.ceil(3.0 * sigma_spatial))

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
        "[SEAM] Using bilateral filter: sigma_color=%.4f, sigma_spatial=%.1f.",
        sigma_color,
        sigma_spatial,
    )
    log.info(
        "[SEAM] Processing %d strip(s) of %d core rows each "
        "(halo=%d px, seam_width=%d px, nodata_fill=%d px, blend_sigma=%.1f).",
        n_strips,
        chunk_rows,
        halo,
        seam_width_pixels,
        nodata_fill_pixels,
        blend_sigma,
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

            # EDT, inpainting zone, nodata fill zone, and blend weight for this strip
            dist_strip = distance_transform_edt(seam_strip == 0).astype(np.float32)
            # Pixels within seam_width_pixels of the centerline are inpainted
            seam_zone = dist_strip <= seam_width_pixels
            # Thin zone where nodata pixels may be filled (kept narrow on purpose)
            fill_zone = dist_strip <= nodata_fill_pixels
            # Cosine taper: weight=1 at centerline, 0 at buffer_pixels.
            # Smooth S-curve avoids the hard blend boundary of a linear ramp.
            weight_strip = np.where(
                dist_strip <= buffer_pixels,
                0.5 * (1.0 - np.cos(np.pi * (1.0 - dist_strip / buffer_pixels))),
                0.0,
            ).astype(np.float32)

            # Read band data for this strip (with halo)
            data_strip = src.read(window=strip_window)  # (C, strip_H, W)
            corrected_strip = data_strip.copy()

            # Bounding box of the correction zone within this strip, padded by
            # the bilateral kernel radius so the filter has enough context.
            # Bilateral is applied to this subregion only — not the full-width
            # strip — to keep runtime acceptable on wide rasters.
            bil_pad = math.ceil(3.0 * sigma_spatial)
            buf_rows = np.where((weight_strip > 0).any(axis=1))[0]
            buf_cols = np.where((weight_strip > 0).any(axis=0))[0]
            r0 = max(0, int(buf_rows[0]) - bil_pad)
            r1 = min(strip_height, int(buf_rows[-1]) + 1 + bil_pad)
            c0 = max(0, int(buf_cols[0]) - bil_pad)
            c1 = min(width, int(buf_cols[-1]) + 1 + bil_pad)

            for band_idx in range(band_count):
                band = data_strip[band_idx].astype(np.float32)
                nodata = nodata_vals[band_idx]
                if nodata is not None:
                    valid_mask = band != np.float32(nodata)
                else:
                    valid_mask = np.isfinite(band)

                # Bilateral inpainting on the seam bounding box only.
                # smooth_weights indicates where the estimate is reliable enough
                # to fill nodata pixels.
                patch_inpainted, patch_weights = _bilateral_inpaint_nodata_safe(
                    band[r0:r1, c0:c1],
                    valid_mask[r0:r1, c0:c1],
                    seam_zone[r0:r1, c0:c1],
                    sigma_color=sigma_color,
                    sigma_spatial=sigma_spatial,
                )
                inpaint_smoothed = band.copy()
                smooth_weights = np.zeros(band.shape, dtype=np.float64)
                inpaint_smoothed[r0:r1, c0:c1] = patch_inpainted
                smooth_weights[r0:r1, c0:c1] = patch_weights

                # Blend Gaussian: small sigma preserves local texture in the taper
                # zone so the buffer does not produce a visible smudge.
                blend_smoothed, _ = _gaussian_smooth_nodata_safe(
                    band,
                    valid_mask,
                    sigma=blend_sigma,
                )

                # In the inpainting zone use the high-sigma result; outside use
                # the lightly smoothed value so the taper adds no visible blur.
                taper_src = np.where(seam_zone, inpaint_smoothed, blend_smoothed)

                # Pass 1: blend valid pixels within the correction zone
                w = weight_strip * valid_mask.astype(np.float32)
                blended = w * taper_src + (1.0 - w) * band
                result = np.where(valid_mask, blended, band).astype(data_strip.dtype)

                # Pass 2: fill nodata pixels in the thin fill zone where the
                # Gaussian had enough valid neighbors to produce a reliable value
                fill_mask = (
                    ~valid_mask & fill_zone & (smooth_weights > _MIN_SMOOTH_WEIGHT)
                )
                corrected_strip[band_idx] = np.where(
                    fill_mask,
                    inpaint_smoothed.astype(data_strip.dtype),
                    result,
                )

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
        default=3,
        help="Half-width of the cosine taper zone in pixels",
    )
    parser.add_argument(
        "--seam_width_pixels",
        type=int,
        default=1,
        help=(
            "Half-width of the inpainting zone around the seam centerline in pixels. "
            "Valid pixels within this distance are excluded from the Gaussian source "
            "so the kernel interpolates from clean data on both sides."
        ),
    )
    parser.add_argument(
        "--nodata_fill_pixels",
        type=int,
        default=5,
        help=(
            "Half-width of the nodata fill zone in pixels. "
            "nodata pixels within this distance of the seam centerline are filled "
            "by Gaussian interpolation when valid neighbors exist. "
            "Keep narrow (a few pixels) to avoid filling legitimate nodata."
        ),
    )
    parser.add_argument(
        "--sigma_color",
        type=float,
        default=0.05,
        help=(
            "Bilateral range-kernel sigma in data units. "
            "Smaller values preserve more terrain edges. "
            "Tune to the value range of your input raster."
        ),
    )
    parser.add_argument(
        "--sigma_spatial",
        type=float,
        default=2.0,
        help=(
            "Bilateral spatial-kernel sigma in pixels. "
            "Must be >= seam_width_pixels to bridge the inpainting zone."
        ),
    )
    parser.add_argument(
        "--blend_sigma",
        type=float,
        default=0.0,
        help=(
            "Gaussian sigma for the blend taper zone outside the inpainting zone. "
            "Keep small (0-2) to preserve texture and avoid a visible smudge."
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
        nodata_fill_pixels=args.nodata_fill_pixels,
        sigma_color=args.sigma_color,
        sigma_spatial=args.sigma_spatial,
        blend_sigma=args.blend_sigma,
        chunk_rows=args.chunk_rows,
    )


if __name__ == "__main__":
    main()

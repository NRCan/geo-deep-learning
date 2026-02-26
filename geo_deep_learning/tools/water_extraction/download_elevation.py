#!/usr/bin/env python3
"""
Download elevation mosaics from the NRCan STAC API for AOI-based extraction.

Enhancements added:
- Asset download resume/skip (size + GeoTIFF validity check)
- Atomic downloads and outputs using *_temp files then rename
- Resume merge: re-use existing local assets; if final dtm/dsm exists & valid, skip; if invalid/empty, rebuild
- Additional logging + explicit stdout flush for HPC visibility
"""

import logging
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import fiona
import numpy as np
import rasterio
import requests
from lightning.pytorch.cli import LightningArgumentParser
from pystac_client import Client
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

log = logging.getLogger(__name__)

logging.getLogger("rasterio.session").setLevel(logging.WARNING)  # Remove boto warning

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

DEFAULT_STAC_API_URL = "https://datacube.services.geo.ca/stac/api/"
DEFAULT_ELEVATION_COLLECTION = "hrdem-lidar"
# Mapping from legacy identifiers to STAC asset names (adjust if catalog changes)
ASSET_KEYS = {"dtm": "dtm", "dsm": "dsm"}


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string."""
    mins, secs = divmod(seconds, 60)
    if mins:
        return f"{int(mins)}m {secs:0.1f}s"
    return f"{secs:0.1f}s"


def _flush_logs() -> None:
    """Force stdout/stderr flush (useful on HPC where buffering can hide progress)."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass


@dataclass
class AOIDownloadConfig:
    """Configuration for downloading elevation data for an AOI."""

    aoi_path: str
    identifiers: list[str] = field(default_factory=lambda: ["dtm", "dsm"])
    out_folder: str | None = None
    resolution: int = 1
    stac_api_url: str = DEFAULT_STAC_API_URL
    collection_id: str = DEFAULT_ELEVATION_COLLECTION

    def __post_init__(self) -> None:
        # Ensure identifiers is a list of strings
        # jsonargparse with action=append yields a list; if a single str sneaks in, wrap it.
        if isinstance(self.identifiers, str):
            self.identifiers = [self.identifiers]
        self.identifiers = [str(x) for x in self.identifiers] or ["dtm", "dsm"]

    def run(self) -> None:
        """Execute the download for the configured AOI."""
        aoi_file = resolve_aoi_path(self.aoi_path)
        out = self.out_folder or str(Path(aoi_file).parent)
        download_all_for_aoi(
            aoi_path=aoi_file,
            out_folder=out,
            identifiers=self.identifiers,
            resolution=self.resolution,
            stac_api_url=self.stac_api_url,
            collection_id=self.collection_id,
        )


def resolve_aoi_path(aoi_input: str) -> str:
    """
    Resolve the AOI vector path.

    If a directory is provided, looks for ``aoi.gpkg`` inside it.
    Otherwise, assumes the input points directly to the vector file.
    """
    p = Path(aoi_input)
    if p.is_dir():
        candidate = p / "aoi.gpkg"
        if not candidate.exists():
            msg = f"AOI file not found at expected path: {candidate}"
            raise FileNotFoundError(msg)
        return str(candidate)
    if p.exists():
        return str(p)
    msg = f"AOI path does not exist: {aoi_input}"
    raise FileNotFoundError(msg)


def load_aoi(
    aoi_path: str,
) -> tuple[list[dict], CRS, tuple[float, float, float, float]]:
    """Load AOI geometries, CRS and bounds from a vector file."""
    with fiona.open(aoi_path, "r") as src:
        geoms = [feature["geometry"] for feature in src]
        if not geoms:
            msg = f"No geometries found in AOI: {aoi_path}"
            raise ValueError(msg)
        crs = CRS.from_user_input(src.crs)
        bounds = src.bounds  # (minx, miny, maxx, maxy)
    return geoms, crs, bounds


def create_directory(path: str) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path to create

    """
    Path(path).mkdir(parents=True, exist_ok=True)


def is_valid_geotiff(path: str) -> bool:
    """
    Check if a file is a valid GeoTIFF with at least one band.

    Args:
        path: Path to the GeoTIFF file to validate

    Returns:
        True if valid, False otherwise

    """
    try:
        with rasterio.open(path) as src:
            return src.count > 0
    except (OSError, rasterio.errors.RasterioIOError) as e:
        log.warning("Invalid GeoTIFF at %s: %s", path, e)
        return False


def _is_nonempty_file(path: Path, min_bytes: int = 1024) -> bool:
    """Simple size check to avoid treating empty/partial files as complete."""
    try:
        return path.exists() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def find_stac_items(
    stac_api_url: str,
    collection_id: str,
    bbox_wgs84: tuple[float, float, float, float],
) -> list:
    """Query the STAC API for items intersecting the AOI."""
    client = Client.open(stac_api_url)
    search = client.search(collections=[collection_id], bbox=bbox_wgs84)
    items = list(search.get_items())
    if not items:
        msg = (
            f"No STAC items found in collection '{collection_id}' "
            f"for bbox {bbox_wgs84}."
        )
        raise RuntimeError(msg)
    return items


def download_asset(url: str, out_path: Path) -> None:
    """
    Download an asset with resume/skip checks and atomic temp file rename.

    Behavior:
    - If out_path exists and looks valid -> skip.
    - Else download to out_path_temp then rename to out_path on success.
    - If out_path_temp exists from a previous attempt, delete and re-download
      (simple, robust; avoids tricky HTTP range resume).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # CHANGED: previously you skipped only if out_path.exists().
    # Original:
    # if out_path.exists():
    #     return
    # ---------------------------------------------------------------------
    if out_path.exists():
        # If already valid and non-empty, skip.
        if _is_nonempty_file(out_path) and is_valid_geotiff(str(out_path)):
            log.info("[SKIP-ASSET] %s already exists and is valid.", out_path.name)
            _flush_logs()
            return
        # If it exists but invalid/empty, we will replace it.
        log.warning(
            "[RE-DOWNLOAD] %s exists but is invalid/empty (size=%s). Re-downloading.",
            out_path.name,
            out_path.stat().st_size if out_path.exists() else "NA",
        )
        _flush_logs()

    tmp_path = out_path.with_name(out_path.stem + "_temp" + out_path.suffix)

    # If a previous temp exists, remove it to avoid confusion.
    if tmp_path.exists():
        try:
            log.warning("[CLEANUP] Removing leftover temp file: %s", tmp_path.name)
            tmp_path.unlink()
        except OSError:
            # If unlink fails, we will overwrite by opening with "wb" anyway,
            # but keep this log for visibility.
            log.warning("[CLEANUP] Could not remove temp file: %s", tmp_path)

    log.info("[DOWNLOAD] %s", out_path.name)
    _flush_logs()

    # Stream download into temp file (quiet mode)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        t0 = time.time()

        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    log.info(
        "[DONE-DOWNLOAD] %s in %s",
        out_path.name,
        _fmt_duration(time.time() - t0),
    )
    _flush_logs()

    # Validate temp file quickly before rename
    if not _is_nonempty_file(tmp_path):
        raise RuntimeError(f"Downloaded temp file is empty: {tmp_path}")

    # Optional: validate GeoTIFF (can be slightly expensive but worth it)
    if not is_valid_geotiff(str(tmp_path)):
        raise RuntimeError(f"Downloaded temp file is not a valid GeoTIFF: {tmp_path}")

    # Atomic-ish rename (same filesystem). Overwrite if needed.
    try:
        os.replace(tmp_path, out_path)
    except OSError:
        # Fallback: remove destination then rename
        if out_path.exists():
            out_path.unlink()
        tmp_path.rename(out_path)

    log.info(
        "[DONE-DOWNLOAD] %s (%.2f MB) in %s",
        out_path.name,
        out_path.stat().st_size / (1024 * 1024),
        _fmt_duration(time.time() - t0),
    )
    _flush_logs()


def download_stac_asset_mosaic(
    identifier: str,
    geoms: list[dict],
    aoi_bounds: tuple[float, float, float, float],
    aoi_crs: CRS,
    resolution: int,
    output_path: str,
    stac_api_url: str,
    collection_id: str,
) -> None:
    """Mosaic, reproject, and clip a STAC asset over the AOI."""
    log.info("Starting download for %s", identifier)
    _flush_logs()
    t0_total = time.time()

    asset_key = ASSET_KEYS.get(identifier, identifier)

    bbox_wgs84 = transform_bounds(aoi_crs, "EPSG:4326", *aoi_bounds, densify_pts=21)

    items = find_stac_items(stac_api_url, collection_id, bbox_wgs84)
    log.info("Found %d STAC items for %s", len(items), identifier)
    _flush_logs()

    assets_dir = Path(output_path).parent / "_assets" / identifier
    assets_dir.mkdir(parents=True, exist_ok=True)

    local_assets: list[Path] = []

    log.info("Assets found for %s:", identifier)
    _flush_logs()

    # ---------------------------------------------------------------------
    # Resume behavior:
    # - We will (re)download any missing/invalid assets
    # - We will include already-existing valid assets in local_assets
    # ---------------------------------------------------------------------
    for item in items:
        asset = item.assets.get(asset_key)
        if not asset:
            continue

        log.info("  - %s | %s", item.id, asset.href)
        _flush_logs()

        fname = f"{item.id}_{asset_key}.tif"
        local_path = assets_dir / fname

        # Ensure local asset exists and is valid; otherwise download.
        try:
            if (
                local_path.exists()
                and _is_nonempty_file(local_path)
                and is_valid_geotiff(str(local_path))
            ):
                log.info("[SKIP-ASSET] %s already present and valid.", local_path.name)
                _flush_logs()
            else:
                download_asset(asset.href, local_path)
        except Exception:
            log.exception("[ASSET-ERROR] Failed asset %s", local_path.name)
            _flush_logs()
            raise

        local_assets.append(local_path)

    if not local_assets:
        raise RuntimeError(f"No assets '{asset_key}' found for AOI {aoi_bounds}")

    # Filter out any assets that exist but are invalid (defensive)
    valid_assets: list[Path] = []
    for p in local_assets:
        if p.exists() and _is_nonempty_file(p) and is_valid_geotiff(str(p)):
            valid_assets.append(p)
        else:
            log.warning(
                "[SKIP-BAD-ASSET] %s is missing/invalid; excluding from merge.", p.name
            )

    if not valid_assets:
        raise RuntimeError(
            f"All local assets are invalid for '{identifier}' in {assets_dir}"
        )

    log.info("Opening %d local rasters for %s", len(valid_assets), identifier)
    _flush_logs()

    sources = [rasterio.open(p) for p in valid_assets]

    try:
        dtype = sources[0].dtypes[0]
        nodata = sources[0].nodata
        if nodata is None:
            nodata = -9999.0 if np.issubdtype(dtype, np.floating) else 0

        log.info("Merging %d local rasters for %s", len(sources), identifier)
        t0 = time.time()
        _flush_logs()

        log.info("[CRS] AOI CRS     : %s", aoi_crs)
        log.info("[CRS] Raster CRS  : %s", sources[0].crs)
        _flush_logs()

        # merged, merged_transform = merge(
        #     sources,
        #     bounds=aoi_bounds,
        #     res=resolution,
        #     nodata=nodata,
        # )

        # --------------------------------------------------------------
        # CRS handling:
        # AOI CRS is authoritative → warp rasters to AOI CRS
        # --------------------------------------------------------------
        raster_crs = sources[0].crs
        if raster_crs is None:
            raise RuntimeError("Source rasters have no CRS; cannot reproject.")

        log.info("[CRS] AOI CRS     : %s", aoi_crs)
        log.info("[CRS] Raster CRS  : %s", raster_crs)
        _flush_logs()

        # Build WarpedVRTs if CRS differs
        if raster_crs != aoi_crs:
            log.info("[CRS] Reprojecting rasters to AOI CRS via WarpedVRT")
            _flush_logs()

            vrt_sources = [
                WarpedVRT(
                    src,
                    crs=aoi_crs,
                    resampling=rasterio.enums.Resampling.nearest,
                )
                for src in sources
            ]
        else:
            vrt_sources = sources

        log.info("[MERGE] Using AOI bounds (AOI CRS): %s", aoi_bounds)
        _flush_logs()

        merged, merged_transform = merge(
            vrt_sources,
            bounds=aoi_bounds,  # AOI bounds in AOI CRS (correct)
            res=resolution,  # resolution in AOI CRS units (meters)
            nodata=nodata,
        )

        log.info(
            "[MERGE] Result for %s: shape=%s dtype=%s",
            identifier,
            merged.shape,
            merged.dtype,
        )
        _flush_logs()

        log.info("Merge completed in %s", _fmt_duration(time.time() - t0))
        _flush_logs()

        log.info("Starting mask + write for %s", identifier)
        t0 = time.time()
        _flush_logs()

        # -----------------------------------------------------------------
        # CHANGED: We now write to output_temp.tif and rename at the end.
        # This prevents 0-byte "final" outputs if the job dies mid-write.
        # -----------------------------------------------------------------
        final_out = Path(output_path)
        tmp_out = final_out.with_name(final_out.stem + "_temp" + final_out.suffix)

        # If leftover temp exists, remove it.
        if tmp_out.exists():
            try:
                log.warning("[CLEANUP] Removing leftover temp output: %s", tmp_out.name)
                tmp_out.unlink()
            except OSError:
                log.warning("[CLEANUP] Could not remove temp output: %s", tmp_out)

        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                height=merged.shape[1],
                width=merged.shape[2],
                count=1,
                dtype=merged.dtype,
                crs=aoi_crs,
                transform=merged_transform,
                nodata=nodata,
                compress="DEFLATE",
                BIGTIFF="YES",
            ) as tmp:
                tmp.write(merged)

                masked, masked_transform = mask(
                    tmp,
                    geoms,
                    crop=True,
                    filled=True,
                    nodata=nodata,
                )

                profile = tmp.profile
                profile.update(
                    height=masked.shape[1],
                    width=masked.shape[2],
                    transform=masked_transform,
                    BIGTIFF="YES",
                )

        log.info("[WRITE] Writing %s to temp output %s", identifier, tmp_out.name)
        _flush_logs()

        with rasterio.open(tmp_out, "w", **profile) as dst:
            dst.write(masked.astype(dtype, copy=False))

        # Validate written temp output before renaming
        if not _is_nonempty_file(tmp_out) or not is_valid_geotiff(str(tmp_out)):
            raise RuntimeError(f"Temp output is invalid after write: {tmp_out}")

        # Rename temp -> final atomically
        try:
            os.replace(tmp_out, final_out)
        except OSError:
            if final_out.exists():
                final_out.unlink()
            tmp_out.rename(final_out)

        log.info(
            "Finished %s → %s in %s (write+mask phase %s)",
            identifier,
            str(final_out),
            _fmt_duration(time.time() - t0_total),
            _fmt_duration(time.time() - t0),
        )
        _flush_logs()

    finally:
        for src in sources:
            src.close()


def download_all_for_aoi(
    aoi_path: str,
    out_folder: str,
    identifiers: Sequence[str],
    resolution: int,
    stac_api_url: str,
    collection_id: str,
) -> None:
    create_directory(out_folder)
    geoms, aoi_crs, bounds = load_aoi(aoi_path)

    for identifier in identifiers:
        out_path = Path(out_folder) / f"{identifier}.tif"

        # -----------------------------------------------------------------
        # Treat empty/invalid outputs as "needs rebuild"
        #
        # Original:
        # -----------------------------------------------------------------
        if out_path.exists():
            if _is_nonempty_file(out_path) and is_valid_geotiff(str(out_path)):
                log.info("[SKIP] %s already exists and is valid.", identifier)
                _flush_logs()
                continue
            log.warning(
                "[REBUILD] %s exists but is invalid/empty (size=%s). Will rebuild.",
                out_path.name,
                out_path.stat().st_size if out_path.exists() else "NA",
            )
            _flush_logs()

        try:
            download_stac_asset_mosaic(
                identifier=identifier,
                geoms=geoms,
                aoi_bounds=bounds,
                aoi_crs=aoi_crs,
                resolution=resolution,
                output_path=str(out_path),
                stac_api_url=stac_api_url,
                collection_id=collection_id,
            )
        except Exception:
            log.exception("Failed to download %s", identifier)
            _flush_logs()
            raise


if __name__ == "__main__":
    parser = LightningArgumentParser()
    parser.add_class_arguments(AOIDownloadConfig, nested_key=None)
    args = parser.parse_args()
    config = AOIDownloadConfig(**vars(args))
    config.run()

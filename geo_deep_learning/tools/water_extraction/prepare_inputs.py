"""Data preparation for water extraction: TWI, tiling, preprocessing."""
from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import fiona
import numpy as np
import pandas as pd
import psutil
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.windows import Window
from shapely.geometry import mapping, shape
from shapely.validation import make_valid
import whitebox_workflows as wbw

import math
import os
import subprocess


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def compute_ndsm(
    dsm_path: str,
    dtm_path: str,
    output_path: str,
    nodata_val: float = -32767,
    block_size: int = 2048,
) -> None:
    """
    Compute nDSM (DSM - DTM) using chunked processing.

    Args:
        dsm_path (str): Path to aligned DSM raster.
        dtm_path (str): Path to aligned DTM raster.
        output_path (str): Where to write the nDSM raster.
        nodata_val (float): Value to assign to NoData areas in the output.
        block_size (int): Size of processing window (both width and height).

    """
    with rasterio.open(dsm_path) as dsm_src, rasterio.open(dtm_path) as dtm_src:
        if dsm_src.shape != dtm_src.shape:
            msg = "DSM and DTM must have the same shape"
            raise ValueError(msg)
        if dsm_src.transform != dtm_src.transform:
            msg = "Transforms do not match"
            raise ValueError(msg)
        if dsm_src.crs != dtm_src.crs:
            msg = "CRS does not match"
            raise ValueError(msg)

        profile = dsm_src.profile
        profile.update(
            {
                "dtype": rasterio.float32,
                "nodata": nodata_val,
                "BIGTIFF": "YES",
                "compress": "lzw",
                "tiled": True,
                "blockxsize": block_size,
                "blockysize": block_size,
            },
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            for y in range(0, dsm_src.height, block_size):
                for x in range(0, dsm_src.width, block_size):
                    window = Window(
                        x,
                        y,
                        min(block_size, dsm_src.width - x),
                        min(block_size, dsm_src.height - y),
                    )

                    dsm = dsm_src.read(1, window=window).astype(np.float32)
                    dtm = dtm_src.read(1, window=window).astype(np.float32)

                    valid_mask = (dsm != dsm_src.nodata) & (dtm != dtm_src.nodata)
                    ndsm = np.full(dsm.shape, nodata_val, dtype=np.float32)
                    ndsm[valid_mask] = dsm[valid_mask] - dtm[valid_mask]

                    dst.write(ndsm, 1, window=window)


def log_system_resources(tag: str = "") -> None:
    """Log current RAM and temporary disk usage for debugging."""
    ram = psutil.virtual_memory()
    tmp = shutil.disk_usage(tempfile.gettempdir())
    log.info(
        "[RESOURCES %s] RAM used: %.1f GB / %.1f GB",
        tag,
        (ram.total - ram.available) / 1e9,
        ram.total / 1e9,
    )
    log.info(
        "[RESOURCES %s] /tmp used: %.1f GB / %.1f GB",
        tag,
        tmp.used / 1e9,
        tmp.total / 1e9,
    )


# def compute_twi_whitebox(  # noqa: C901
#     dtm_path: str,
#     twi_output_path: str,
#     temp_dir: str | None = None,
# ) -> None:
#     """
#     Compute the Topographic Wetness Index (TWI) from a DTM using WhiteboxTools.

#     Args:
#         dtm_path (str): Path to the input DTM GeoTIFF.
#         twi_output_path (str): Path where the computed TWI raster will be saved.
#         temp_dir (str): Folder to store intermediate files. Defaults to TWI's folder.

#     """
#     # Ensure per-AOI temp dir
#     if temp_dir is None:
#         temp_dir = Path(twi_output_path).parent / "temp_whitebox"
#     temp_dir_path = Path(temp_dir)
#     if temp_dir_path.exists():
#         shutil.rmtree(temp_dir_path)
#     temp_dir_path.mkdir(parents=True, exist_ok=True)
#     temp_dir = str(temp_dir_path.resolve())

#     # Init Whitebox
#     wbt = WhiteboxTools()
#     wbt.verbose = True
#     wbt.set_working_dir(temp_dir)

#     # Absolute paths
#     dtm_path = str(Path(dtm_path).resolve())
#     twi_output_path = str(Path(twi_output_path).resolve())

#     dtm_breached = str(temp_dir_path / "dtm_breached.tif")
#     slope_path = str(temp_dir_path / "slope.tif")
#     flow_acc_path = str(temp_dir_path / "flow_acc.tif")

#     log.info("[TWI] Working dir: %s", temp_dir)
#     log.info("[TWI] DTM: %s", dtm_path)
#     log.info("[TWI] Output: %s", twi_output_path)

#     # Step 1: Breach depressions
#     if not Path(dtm_breached).exists():
#         log.info("[TWI] Breaching depressions...")
#         log_system_resources("before breach_depressions")
#         ret = wbt.breach_depressions(dem=dtm_path, output=dtm_breached)
#         log_system_resources("after breach_depressions")
#         if ret != 0:
#             error_msg = f"[Whitebox] breach_depressions failed with code {ret}"
#             raise RuntimeError(error_msg)

#     # Step 2: Slope
#     if not Path(slope_path).exists():
#         log.info("[TWI] Computing slope...")
#         log_system_resources("before slope")
#         ret = wbt.slope(dem=dtm_breached, output=slope_path, zfactor=1.0)
#         log_system_resources("after slope")
#         if ret != 0:
#             error_msg = f"[Whitebox] slope failed with code {ret}"
#             raise RuntimeError(error_msg)

#     # Step 3: Flow accumulation
#     if not Path(flow_acc_path).exists():
#         log.info("[TWI] Computing flow accumulation...")
#         ret = wbt.d8_flow_accumulation(
#             i=dtm_breached,
#             output=flow_acc_path,
#             out_type="sca",
#         )
#         if ret != 0:
#             error_msg = f"[Whitebox] d8_flow_accumulation failed with code {ret}"
#             raise RuntimeError(error_msg)

#     if not Path(slope_path).exists() or not Path(flow_acc_path).exists():
#         error_msg = "[Whitebox] One or more required intermediate files are missing."
#         raise FileNotFoundError(error_msg)

#     # Step 4: Wetness index
#     log.info("[TWI] Running wetness_index...")
#     ret = wbt.wetness_index(sca=flow_acc_path, slope=slope_path, output=twi_output_path)
#     if ret != 0 or not Path(twi_output_path).exists():
#         error_msg = f"[Whitebox] wetness_index failed. Code: {ret}"
#         raise RuntimeError(error_msg)

#     log.info("[TWI] TWI created: %s", twi_output_path)

# def compute_twi_whitebox(
#     dtm_path: str,
#     twi_output_path: str,
#     temp_dir: str | None = None,
#     keep_intermediate: bool = False,
# ) -> None:
#     """
#     Compute the Topographic Wetness Index (TWI) from a DTM using whitebox_workflows.
#     """
#     dtm_path = str(Path(dtm_path).resolve())
#     twi_output_path = str(Path(twi_output_path).resolve())
#     if temp_dir is not None:
#         temp_dir = str(Path(temp_dir).resolve())
#     else:
#         temp_dir = str(Path(twi_output_path).parent / "temp_whitebox")
#     Path(temp_dir).mkdir(parents=True, exist_ok=True)

#     log.info("[TWI] Running whitebox_workflows TWI pipeline")
#     log.info("[TWI] DTM: %s", dtm_path)
#     log.info("[TWI] Output: %s", twi_output_path)

#     wbe = wbw.WbEnvironment()
#     wbe.working_directory = temp_dir
#     log.info("[TWI] Loading DTM raster: %s", dtm_path)
#     dem = wbe.read_raster(dtm_path)

#     breached_path = Path(temp_dir) / "dem_breached.tif"
#     slope_path = Path(temp_dir) / "slope.tif"
#     sca_path = Path(temp_dir) / "flow_acc_sca.tif"

#     # Step 1: Breach depressions
#     if keep_intermediate and breached_path.exists():
#         log.info(f"[TWI] Using existing breached DEM: {breached_path}")
#         dem_breached = wbe.read_raster(str(breached_path))
#     else:
#         log.info("[TWI] Breaching depressions (least cost)...")
#         dem_breached = wbe.breach_depressions_least_cost(dem)
#         if keep_intermediate:
#             log.info(f"[TWI] Saving breached DEM to {breached_path}")
#             wbe.write_raster(dem_breached, str(breached_path))
#     del dem

#     # Step 2: Slope
#     if keep_intermediate and slope_path.exists():
#         log.info(f"[TWI] Using existing slope raster: {slope_path}")
#         slope = wbe.read_raster(str(slope_path))
#     else:
#         log.info("[TWI] Computing slope (radians)...")
#         slope = wbe.slope(dem_breached, z_factor=1.0)
#         if keep_intermediate:
#             log.info(f"[TWI] Saving slope raster to {slope_path}")
#             wbe.write_raster(slope, str(slope_path))

#     # Step 3: Flow accumulation
#     if keep_intermediate and sca_path.exists():
#         log.info(f"[TWI] Using existing flow accumulation raster: {sca_path}")
#         sca = wbe.read_raster(str(sca_path))
#     else:
#         log.info("[TWI] Computing D8 flow accumulation (sca)...")
#         sca = wbe.qin_flow_accumulation(dem_breached, out_type="sca")
#         if keep_intermediate:
#             log.info(f"[TWI] Saving flow accumulation raster to {sca_path}")
#             wbe.write_raster(sca, str(sca_path))
#     del dem_breached

#     # Step 4: Wetness index
#     log.info("[TWI] Computing wetness index...")
#     twi = wbe.wetness_index(sca, slope)
#     del sca, slope

#     log.info("[TWI] Writing TWI raster to: %s", twi_output_path)
#     wbe.write_raster(twi, twi_output_path)

#     if not Path(twi_output_path).exists():
#         raise RuntimeError("[whitebox_workflows] TWI output was not created")

#     log.info("[TWI] TWI created: %s", twi_output_path)

def get_raster_size(path: str) -> tuple[int, int]:
    with rasterio.open(path) as ds:
        height = ds.height  # rows
        width = ds.width    # cols
    return height, width

def write_raster_window(
    src_path: str,
    dst_path: str,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
) -> None:
    with rasterio.open(src_path) as src:
        window = Window(
            col_off=col0,
            row_off=row0,
            width=col1 - col0,
            height=row1 - row0,
        )
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            height=window.height,
            width=window.width,
            transform=transform,
        )

        data = src.read(1, window=window)

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data, 1)

def compute_twi_whitebox(
    dtm_path: str,
    twi_output_path: str,
    temp_dir: str | None = None,
    keep_intermediate: bool = False,
    *,
    # With your AOI size, 2048–4096 is typically safe on 256 GB
    tile_size: int = 4096,
    # halo for slope correctness at tile boundaries (1 px is sufficient for slope)
    halo: int = 1,
    # keep OMP modest; higher can increase memory pressure
    max_threads: int = 8,
) -> None:
    """
    Compute Topographic Wetness Index (TWI) from a DTM using whitebox_workflows,
    optimized for very large AOIs by tiling only the slope + TWI stages.

    Global (once):
      1) breach_depressions_least_cost (checkpoint)
      2) d8_pointer (checkpoint)
      3) d8_flow_accum from pointer -> SCA (checkpoint)

    Tiled (many):
      4) slope on breached DEM (tiled, with 1px halo)
      5) wetness_index using (SCA tile, slope tile)
      6) mosaic tiles to final TWI
    """

    dtm_path = Path(dtm_path).resolve()
    twi_output_path = Path(twi_output_path).resolve()

    if temp_dir is None:
        temp_dir = twi_output_path.parent / "temp_whitebox"
    else:
        temp_dir = Path(temp_dir).resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Cap OpenMP threads (critical)
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))

    log.info("[TWI] DTM: %s", dtm_path)
    log.info("[TWI] Output: %s", twi_output_path)
    log.info("[TWI] Temp dir: %s", temp_dir)
    log.info("[TWI] tile_size=%d halo=%d OMP_NUM_THREADS=%s", tile_size, halo, os.environ.get("OMP_NUM_THREADS"))

    wbe = wbw.WbEnvironment()
    wbe.working_directory = str(temp_dir)
    wbe.verbose = False

    breached_path = temp_dir / "dem_breached.tif"
    pointer_path = temp_dir / "d8_pointer.tif"
    sca_path = temp_dir / "sca_d8.tif"

    tiles_dir = temp_dir / "twi_tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Step 1 — breach depressions (global)
    # ---------------------------------------------------------------------
    if breached_path.exists():
        log.info("[TWI] Using existing breached DEM: %s", breached_path)
    else:
        log.info("[TWI] Loading DTM")
        dem = wbe.read_raster(str(dtm_path))

        log.info("[TWI] Breaching depressions (least cost)")
        dem_breached = wbe.breach_depressions_least_cost(dem)
        del dem

        log.info("[TWI] Writing breached DEM: %s", breached_path)
        wbe.write_raster(dem_breached, str(breached_path))
        del dem_breached

    # ---------------------------------------------------------------------
    # Step 2 — D8 pointer (global)
    # ---------------------------------------------------------------------
    if pointer_path.exists():
        log.info("[TWI] Using existing D8 pointer: %s", pointer_path)
    else:
        log.info("[TWI] Computing D8 pointer")
        dem_breached = wbe.read_raster(str(breached_path))
        pntr = wbe.d8_pointer(dem_breached)
        del dem_breached

        log.info("[TWI] Writing D8 pointer: %s", pointer_path)
        wbe.write_raster(pntr, str(pointer_path))
        del pntr

    # ---------------------------------------------------------------------
    # Step 3 — SCA from pointer (global)
    # ---------------------------------------------------------------------
    if sca_path.exists():
        log.info("[TWI] Using existing SCA: %s", sca_path)
    else:
        log.info("[TWI] Computing D8 flow accumulation from pointer (SCA)")
        pntr = wbe.read_raster(str(pointer_path))
        sca = wbe.d8_flow_accum(
            pntr,
            out_type="sca",
            input_is_pointer=True,
            log_transform=False,
        )
        del pntr

        log.info("[TWI] Writing SCA: %s", sca_path)
        wbe.write_raster(sca, str(sca_path))
        del sca

    # ---------------------------------------------------------------------
    # Determine raster dimensions (use breached DEM as reference)
    # ---------------------------------------------------------------------
    nrows, ncols = get_raster_size(str(breached_path))

    log.info("[TWI] Raster size: %d cols × %d rows", ncols, nrows)

    n_tiles_x = math.ceil(ncols / tile_size)
    n_tiles_y = math.ceil(nrows / tile_size)
    log.info("[TWI] Tiling grid: %d × %d tiles", n_tiles_x, n_tiles_y)

    tile_paths: list[Path] = []

    # ---------------------------------------------------------------------
    # Step 4/5 — tile slope + twi (local; memory-bounded)
    # ---------------------------------------------------------------------
    total_tiles = n_tiles_x * n_tiles_y
    done_tiles = 0
    log_every_tiles = max(1, total_tiles // 100)  # ~1% increments
    
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            core_row0 = ty * tile_size
            core_col0 = tx * tile_size
            core_row1 = min(core_row0 + tile_size, nrows)
            core_col1 = min(core_col0 + tile_size, ncols)

            # Halo window for slope
            buf_row0 = max(core_row0 - halo, 0)
            buf_col0 = max(core_col0 - halo, 0)
            buf_row1 = min(core_row1 + halo, nrows)
            buf_col1 = min(core_col1 + halo, ncols)
            
            tile_id = f"twi_{ty:03d}_{tx:03d}"
            tile_dir = tiles_dir / tile_id
            tile_dir.mkdir(parents=True, exist_ok=True)

            if tx == 0:
                pct = 100.0 * (ty * n_tiles_x) / total_tiles
                log.info("[TWI] Row %d/%d (%.1f%%) tiles %d–%d of %d",
                        ty + 1, n_tiles_y, pct,
                        ty * n_tiles_x + 1, min((ty + 1) * n_tiles_x, total_tiles),
                        total_tiles)

            dem_tile_path = tile_dir / "dem_buf.tif"
            sca_tile_path = tile_dir / "sca_core.tif"
            slope_buf_path = tile_dir / "slope_buf.tif"
            slope_core_path = tile_dir / "slope_core.tif"
            twi_tile_path = tile_dir / f"{tile_id}.tif"

            # --- write DEM buffered window (with halo) ---
            write_raster_window(
                str(breached_path),
                str(dem_tile_path),
                buf_row0,
                buf_row1,
                buf_col0,
                buf_col1,
            )

            # --- write SCA core window ---
            write_raster_window(
                str(sca_path),
                str(sca_tile_path),
                core_row0,
                core_row1,
                core_col0,
                core_col1,
            )

            # --- read tiles into Whitebox ---
            dem_buf = wbe.read_raster(str(dem_tile_path))
            sca_core = wbe.read_raster(str(sca_tile_path))

            # --- slope on buffered DEM ---
            slope_buf = wbe.slope(dem_buf, z_factor=1.0)
            del dem_buf

            # write slope buffer so we can crop with rasterio
            wbe.write_raster(slope_buf, str(slope_buf_path))
            del slope_buf

            # --- crop slope buffer back to core window ---
            write_raster_window(
                str(slope_buf_path),
                str(slope_core_path),
                core_row0 - buf_row0,
                core_row1 - buf_row0,
                core_col0 - buf_col0,
                core_col1 - buf_col0,
            )

            slope_core = wbe.read_raster(str(slope_core_path))

            # --- TWI ---
            twi_core = wbe.wetness_index(sca_core, slope_core)
            del sca_core, slope_core

            wbe.write_raster(twi_core, str(twi_tile_path))
            del twi_core

            tile_paths.append(twi_tile_path)

            # done_tiles += 1
            # if done_tiles % log_every_tiles == 0 or done_tiles == total_tiles:
            #     pct = 100.0 * done_tiles / total_tiles
            #     log.info("[TWI] Progress: %d/%d tiles (%.1f%%)", done_tiles, total_tiles, pct)
                
    # ---------------------------------------------------------------------
    # Step 6 — mosaic tiles to final output using GDAL (robust)
    # ---------------------------------------------------------------------
    vrt_path = temp_dir / "twi_tiles.vrt"
    tmp_tif = temp_dir / "twi_mosaic_tmp.tif"

    # Build VRT
    log.info("[TWI] Building VRT mosaic: %s", vrt_path)
    subprocess.run(
        ["gdalbuildvrt", "-overwrite", str(vrt_path)] + [str(p) for p in tile_paths],
        check=True,
    )

    # Translate VRT to GeoTIFF (compressed)
    log.info("[TWI] Writing final GeoTIFF: %s", twi_output_path)
    subprocess.run(
        [
            "gdal_translate",
            "-of", "GTiff",
            "-co", "COMPRESS=DEFLATE",
            "-co", "TILED=YES",
            "-co", "BIGTIFF=YES",
            str(vrt_path),
            str(tmp_tif),
        ],
        check=True,
    )

    # Move into place atomically
    tmp_tif.replace(twi_output_path)

    if not twi_output_path.exists():
        raise RuntimeError("[TWI] Final TWI output was not created")

    log.info("[TWI] TWI created: %s", twi_output_path)

    # Optional cleanup
    if not keep_intermediate:
        # Keep the big global checkpoints unless you explicitly want them gone
        # (they’re useful for restart). Remove tiles + vrt.
        shutil.rmtree(tiles_dir, ignore_errors=True)
        if vrt_path.exists():
            vrt_path.unlink(missing_ok=True)

# def stack_rasters(
#     raster_paths: "Sequence[str]",
#     output_path: str,
#     nodata_val: float = -32767,
# ) -> None:
#     """
#     Stack multiple single-band rasters into a multi-band raster.

#     Args:
#         raster_paths: Ordered list of raster file paths (TWI, nDSM, Intensity).
#         output_path: Path to save the stacked raster.
#         nodata_val: Value to use for NoData in the output bands.
#     """
#     sources = [rasterio.open(path) for path in raster_paths]

#     # Check alignment
#     ref_shape = sources[0].shape
#     ref_transform = sources[0].transform
#     ref_crs = sources[0].crs

#     for src in sources[1:]:
#         if src.shape != ref_shape:
#             error_msg = f"Shape mismatch: {src.name}"
#             raise ValueError(error_msg)
#         if src.transform != ref_transform:
#             error_msg = f"Transform mismatch: {src.name}"
#             raise ValueError(error_msg)
#         if src.crs != ref_crs:
#             error_msg = f"CRS mismatch: {src.name}"
#             raise ValueError(error_msg)

#     profile = sources[0].profile
#     profile.update(
#         {
#             "count": len(sources),
#             "dtype": "float32",  # Use float32 to accommodate all input types
#             "nodata": nodata_val,
#             "BIGTIFF": "YES",
#             "compress": "lzw",
#         },
#     )

#     with rasterio.open(output_path, "w", **profile) as dst:
#         for i, src in enumerate(sources, start=1):
#             band = src.read(1).astype(np.float32)
#             # Replace source nodata with output nodata
#             if src.nodata is not None:
#                 band[band == src.nodata] = nodata_val
#             dst.write(band, i)

#     for src in sources:
#         src.close()

def stack_rasters(
    input_rasters: Sequence[str],
    output_path: str,
    nodata_val: float = -32767,
) -> None:
    """
    Stack multiple single-band rasters into a multiband GeoTIFF.

    - Preserves per-band dtype
    - Preserves per-band NoData when compatible
    - Avoids applying float NoData to integer rasters (e.g. intensity)
    - Assumes all rasters are already aligned (same grid, CRS, transform)

    Args:
        input_rasters: Ordered list of single-band raster paths
        output_path: Output stacked GeoTIFF path
        nodata_val: Default NoData for floating-point rasters
    """

    if not input_rasters:
        raise ValueError("No input rasters provided for stacking")

    # ------------------------------------------------------------------
    # Open reference raster
    # ------------------------------------------------------------------
    try:
        with rasterio.open(input_rasters[0]) as ref:
            profile = ref.profile.copy()
    except RasterioIOError as e:
        raise RuntimeError(f"Failed to open reference raster: {input_rasters[0]}") from e

    profile.update(
        count=len(input_rasters),
        compress="lzw",
        interleave="band",
        tiled=True,
        BIGTIFF="IF_SAFER",
        nodata=None,  # CRITICAL: prevent global nodata
    )

    # ------------------------------------------------------------------
    # Create output
    # ------------------------------------------------------------------
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        for band_idx, raster_path in enumerate(input_rasters, start=1):
            with rasterio.open(raster_path) as src:
                data = src.read(1)

                # Write raw data
                dst.write(data, band_idx)

                # ------------------------------------------------------------------
                # Per-band NoData handling
                # ------------------------------------------------------------------
                src_dtype = src.dtypes[0]
                src_nodata = src.nodata

                if np.issubdtype(np.dtype(src_dtype), np.integer):
                    # Integer rasters (e.g. intensity)
                    if src_nodata is not None:
                        dst.update_tags(band_idx, NODATA_VALUE=src_nodata)
                else:
                    # Floating-point rasters (TWI, nDSM)
                    dst.update_tags(band_idx, NODATA_VALUE=nodata_val)

                # Optional: band description for readability
                dst.set_band_description(
                    band_idx, Path(raster_path).stem
                )

    return None

def rasterize_labels_binary_aoi_mask(  # noqa: PLR0913
    label_vector_path: str,
    aoi_vector_path: str,
    reference_raster_path: str,
    output_path: str,
    *,
    burn_value: int = 1,
    fill_value: int = 0,
    ignore_value: int = -1,
) -> None:
    """
    Rasterize label vector and apply AOI mask.

    Args:
        label_vector_path: Path to label vector file
        aoi_vector_path: Path to AOI vector file
        reference_raster_path: Raster to match for shape, transform, CRS
        output_path: Where to save the output raster
        burn_value: Value to burn for label geometries
        fill_value: Value for background pixels
        ignore_value: Value for pixels outside AOI

    """
    # Skip if output already exists
    if Path(output_path).exists():
        log.info("Skipping label rasterization (already exists): %s", output_path)
        return

    # Open reference raster
    with rasterio.open(reference_raster_path) as ref:
        ref_profile = ref.profile
        shape_hw = (ref.height, ref.width)
        transform = ref.transform
        crs = ref.crs

    # Rasterize labels (binary)
    with fiona.open(label_vector_path, "r") as src:
        crs_vector = src.crs
        shapes = []
        for feat in src:
            if feat["geometry"] is None:
                continue
            geom = shape(feat["geometry"])
            if not geom.is_valid:
                geom = make_valid(geom)
            if crs_vector != crs:
                from fiona.transform import (
                    transform_geom,
                )

                geom = shape(transform_geom(crs_vector, crs, mapping(geom)))
            shapes.append((mapping(geom), burn_value))

    label_raster = rasterize(
        shapes=shapes,
        out_shape=shape_hw,
        transform=transform,
        fill=fill_value,
        dtype=np.int16,  # allow for -1
    )

    # Rasterize AOI mask
    with fiona.open(aoi_vector_path, "r") as src:
        crs_aoi = src.crs
        aoi_shapes = []
        for feat in src:
            if feat["geometry"] is None:
                continue
            geom = shape(feat["geometry"])
            if not geom.is_valid:
                geom = make_valid(geom)
            if crs_aoi != crs:
                from fiona.transform import (
                    transform_geom,
                )

                geom = shape(transform_geom(crs_aoi, crs, mapping(geom)))
            aoi_shapes.append((mapping(geom), 1))

    aoi_mask = rasterize(
        shapes=aoi_shapes,
        out_shape=shape_hw,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    # Apply AOI mask: set everything outside AOI to ignore_value
    label_raster[aoi_mask == 0] = ignore_value

    log.info(
        "Final rasterized unique values (after AOI masking): %s",
        np.unique(label_raster),
    )

    # Save result
    ref_profile.update(
        {
            "count": 1,
            "dtype": "int16",
            "nodata": ignore_value,
            "BIGTIFF": "YES",
            "compress": "lzw",
        },
    )
    with rasterio.open(output_path, "w", **ref_profile) as dst:
        dst.write(label_raster, 1)

    log.info("Saved AOI-masked label raster to: %s", output_path)


def rasterize_labels_binary(
    vector_path: str,
    output_path: str,
    ref_profile: dict,
    *,
    burn_value: int = 1,
    fill_value: int = 0,
) -> None:
    """
    Rasterize a vector file to a binary raster matching a reference profile.

    Args:
        vector_path: Path to vector file to rasterize
        output_path: Where to save the output raster
        ref_profile: Reference raster profile (shape, transform, CRS)
        burn_value: Value to burn for geometries
        fill_value: Value for background pixels

    """
    with fiona.open(vector_path, "r") as src:
        crs_vector = src.crs
        shapes = []
        for feat in src:
            if feat["geometry"] is None:
                continue
            geom = shape(feat["geometry"])
            if not geom.is_valid:
                geom = make_valid(geom)
            if crs_vector != ref_profile["crs"]:
                from fiona.transform import (
                    transform_geom,
                )

                geom = shape(
                    transform_geom(crs_vector, ref_profile["crs"], mapping(geom)),
                )
            shapes.append((mapping(geom), burn_value))

    label_raster = rasterize(
        shapes=shapes,
        out_shape=(ref_profile["height"], ref_profile["width"]),
        transform=ref_profile["transform"],
        fill=fill_value,
        dtype="uint8",
    )

    log.info("Rasterized unique values: %s", np.unique(label_raster))

    ref_profile.update(count=1, dtype="uint8", nodata=255)
    with rasterio.open(output_path, "w", **ref_profile) as dst:
        dst.write(label_raster, 1)


def rasterize_valid_lidar_mask(
    vector_path: str,
    reference_raster_path: str,
    output_path: str,
    *,
    burn_value: int = 1,
    fill_value: int = 0,
) -> None:
    """
    Rasterize a LiDAR validity polygon to a binary mask matching a reference raster.

    Args:
        vector_path: Path to vector file describing valid LiDAR coverage.
        reference_raster_path: Raster to match for shape, transform, CRS.
        output_path: Where to save the output raster.
        burn_value: Value to burn for valid geometries (defaults to 1).
        fill_value: Value for background pixels (defaults to 0).

    """
    with rasterio.open(reference_raster_path) as ref:
        ref_profile = ref.profile
        shape_hw = (ref.height, ref.width)
        transform = ref.transform
        crs = ref.crs

    with fiona.open(vector_path, "r") as src:
        crs_vector = src.crs
        shapes = []
        for feat in src:
            if feat["geometry"] is None:
                continue
            geom = shape(feat["geometry"])
            if not geom.is_valid:
                geom = make_valid(geom)
            if crs_vector != crs:
                from fiona.transform import transform_geom

                geom = shape(transform_geom(crs_vector, crs, mapping(geom)))
            shapes.append((mapping(geom), burn_value))

    valid_mask = rasterize(
        shapes=shapes,
        out_shape=shape_hw,
        transform=transform,
        fill=fill_value,
        dtype=np.uint8,
    )

    log.info(
        "Rasterized valid mask unique values: %s (fill value=%s)",
        np.unique(valid_mask),
        fill_value,
    )

    ref_profile.update(
        {
            "count": 1,
            "dtype": "uint8",
            "nodata": fill_value,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        },
    )

    with rasterio.open(output_path, "w", **ref_profile) as dst:
        dst.write(valid_mask, 1)


def tile_raster_pair(  # noqa: PLR0913
    input_path: str,
    label_path: str,
    output_dir: str,
    *,
    patch_size: int = 512,
    stride: int = 512,
    nodata_val: float = -32767,
    min_valid_ratio: float = 0.9,
    valid_mask_path: str | None = None,
    valid_mask_min_ratio: float | None = 0.9,
    save_rejected_tiles: bool = False,
    rejected_dir: str | None = None,
) -> None:
    """
    Tile input/label rasters into patches, apply all validity filtering once,
    and compute authoritative tile-level statistics.
    """

    from pathlib import Path
    import numpy as np
    import pandas as pd
    import rasterio
    from rasterio.windows import Window

    Path(output_dir, "inputs").mkdir(parents=True, exist_ok=True)
    Path(output_dir, "labels").mkdir(parents=True, exist_ok=True)

    rejected_root = Path(rejected_dir) if rejected_dir else Path(output_dir) / "rejected_tiles"
    rejected_inputs = rejected_root / "inputs"
    rejected_labels = rejected_root / "labels"
    rejected_masks = rejected_root / "valid_masks"

    tile_stats: list[dict] = []

    def _save_rejected(
        x: int,
        y: int,
        input_patch: np.ndarray,
        label_patch: np.ndarray,
        input_meta: dict,
        label_meta: dict,
        mask_patch: np.ndarray | None = None,
    ) -> None:
        if not save_rejected_tiles:
            return

        rejected_inputs.mkdir(parents=True, exist_ok=True)
        rejected_labels.mkdir(parents=True, exist_ok=True)

        name = f"reject_x{x:05d}_y{y:05d}.tif"
        with rasterio.open(rejected_inputs / name, "w", **input_meta) as dst:
            dst.write(input_patch)
        with rasterio.open(rejected_labels / name, "w", **label_meta) as dst:
            dst.write(label_patch, 1)

        if mask_patch is not None:
            rejected_masks.mkdir(parents=True, exist_ok=True)
            mask_meta = label_meta | {"dtype": "uint8", "nodata": 0}
            with rasterio.open(rejected_masks / name, "w", **mask_meta) as dst:
                dst.write(mask_patch.astype("uint8"), 1)

    with rasterio.open(input_path) as src_input, rasterio.open(label_path) as src_label:
        if src_input.transform != src_label.transform or src_input.shape != src_label.shape:
            raise ValueError("Input and label rasters must be aligned")

        valid_mask_src = rasterio.open(valid_mask_path) if valid_mask_path else None

        tile_id = 0
        total_candidates = 0
        filtered = 0

        for y in range(0, src_input.height - patch_size + 1, stride):
            for x in range(0, src_input.width - patch_size + 1, stride):
                total_candidates += 1
                window = Window(x, y, patch_size, patch_size)

                input_patch = src_input.read(window=window)
                label_patch = src_label.read(1, window=window)

                #valid_pixels = np.sum(label_patch != nodata_val)
                valid_pixels = np.sum(label_patch != -1)
                if valid_pixels == 0:
                    filtered += 1
                    _save_rejected(...)
                    continue
                
                valid_ratio = valid_pixels / label_patch.size

                if valid_ratio < min_valid_ratio:
                    filtered += 1
                    _save_rejected(x, y, input_patch, label_patch, src_input.meta, src_label.meta)
                    continue

                if valid_mask_src and valid_mask_min_ratio is not None:
                    mask_patch = valid_mask_src.read(1, window=window)
                    if np.mean(mask_patch == 1) < valid_mask_min_ratio:
                        filtered += 1
                        _save_rejected(
                            x, y, input_patch, label_patch,
                            src_input.meta, src_label.meta,
                            mask_patch=mask_patch,
                        )
                        continue

                water_pixels = int(np.sum(label_patch == 1))
                water_ratio = water_pixels / max(valid_pixels, 1)

                # Save tiles
                input_meta = src_input.meta | {
                    "height": patch_size,
                    "width": patch_size,
                    "transform": rasterio.windows.transform(window, src_input.transform),
                }
                label_meta = src_label.meta | {
                    "count": 1,
                    "height": patch_size,
                    "width": patch_size,
                    "transform": rasterio.windows.transform(window, src_label.transform),
                }

                with rasterio.open(
                    Path(output_dir) / "inputs" / f"tile_{tile_id:05d}.tif", "w", **input_meta
                ) as dst:
                    dst.write(input_patch)

                with rasterio.open(
                    Path(output_dir) / "labels" / f"tile_{tile_id:05d}_label.tif", "w", **label_meta
                ) as dst:
                    dst.write(label_patch, 1)

                tile_stats.append({
                    "tile_id": tile_id,
                    "x": x,
                    "y": y,
                    "valid_pixels": valid_pixels,
                    "water_pixels": water_pixels,
                    "water_ratio": water_ratio,
                })

                tile_id += 1

        if valid_mask_src:
            valid_mask_src.close()

    pd.DataFrame(tile_stats).to_csv(
    Path(output_dir).parent / "tile_stats.csv",
    index=False,
    )

    log.info(
        "Tiling complete: %d candidates, %d kept, %d filtered → %s",
        total_candidates,
        tile_id,
        filtered,
        output_dir,
    )

def generate_csv_from_tiles(
    root_output_folder: str,
    csv_tiling_path: str,
    csv_inference_path: str,
    *,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    water_ratio_bins: tuple[float, ...] = (0.001, 0.01, 0.05),
    min_water_pixels: int = 1,  # NEW (1 = remove zero-water tiles)
) -> None:
    """
    Generate stratified train/val/test CSVs using tile-level statistics.

    Filtering policy:
      - tiles with water_pixels < min_water_pixels are removed
      - filtering is applied BEFORE split
    """

    import pandas as pd
    import numpy as np
    from pathlib import Path

    rows: list[dict] = []

    log.info(f"[DEBUG] Path(root_output_folder) = {Path(root_output_folder)}")
    log.info(f"[DEBUG] Path(root_output_folder) = {Path(root_output_folder)}")

    for aoi_dir in Path(root_output_folder).iterdir():
        
        log.info(f"[DEBUG] aoi_dir inside loop = {aoi_dir}")
        
        if not aoi_dir.is_dir():
            continue
        
        tiles_root = aoi_dir / "tiles"
        if not tiles_root.exists():
            raise FileNotFoundError(f"Missing tiles directory in {aoi_dir}")

        stats_path = aoi_dir / "tile_stats.csv"
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing tile_stats.csv in {aoi_dir}")

        stats_df = pd.read_csv(stats_path)

        for _, row in stats_df.iterrows():
            tid = int(row["tile_id"])
            rows.append({
                "tif": str(aoi_dir / "tiles" / "inputs" / f"tile_{tid:05d}.tif"),
                "gpkg": str(aoi_dir / "tiles" / "labels" / f"tile_{tid:05d}_label.tif"),
                "aoi": aoi_dir.name,
                "water_pixels": int(row["water_pixels"]),
                "water_ratio": float(row["water_ratio"]),
            })

    df = pd.DataFrame(rows)

    # -------------------------------------------------
    # CSV-level water filtering (CLEAN & EXPLICIT)
    # -------------------------------------------------
    before = len(df)
    df = df[df["water_pixels"] >= min_water_pixels].reset_index(drop=True)
    after = len(df)

    log.info(
        "CSV filtering: removed %d tiles with water_pixels < %d (%d → %d)",
        before - after,
        min_water_pixels,
        before,
        after,
    )

    # -------------------------------------------------
    # Stratification by water ratio
    # -------------------------------------------------
    df["water_bin"] = pd.cut(
        df["water_ratio"],
        bins=(0.0, *water_ratio_bins, 1.0),
        labels=False,
        include_lowest=True,
    )

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    splits: list[str] = []

    for _, bin_df in df.groupby("water_bin"):
        n = len(bin_df)
        n_val = int(np.floor(n * val_ratio))
        n_test = int(np.floor(n * test_ratio))

        splits.extend(
            ["val"] * n_val +
            ["tst"] * n_test +
            ["trn"] * (n - n_val - n_test)
        )

    if test_ratio == 1.0:
        df["split"] = "tst"
    else:
        df["split"] = splits
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # -------------------------------------------------
    # Save outputs
    # -------------------------------------------------
    Path(csv_tiling_path).parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["water_bin"]).to_csv(csv_tiling_path, index=False)

    infer_df = df[df["split"] == "tst"].copy()
    infer_df["split"] = "inference"
    Path(csv_inference_path).parent.mkdir(parents=True, exist_ok=True)
    infer_df.drop(columns=["water_bin"]).to_csv(csv_inference_path, index=False)

    log.info(
        "CSV generation complete: %d tiles (%d train / %d val / %d test)",
        len(df),
        (df["split"] == "trn").sum(),
        (df["split"] == "val").sum(),
        (df["split"] == "tst").sum(),
    )
    
def align_to_reference(
    ref_path: str,
    src_path: str,
    dst_path: str,
    resampling: Resampling = Resampling.bilinear,
) -> None:
    """
    Align a raster to the grid (extent, resolution, CRS) of a reference raster.

    Args:
        ref_path (str): Path to reference raster (e.g., DTM).
        src_path (str): Path to source raster (e.g., DSM, intensity).
        dst_path (str): Where to save the aligned raster.
        resampling (rasterio.warp.Resampling): Resampling method.

    """
    with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs

        # Prepare output profile
        ref_profile.update(
            {
                "count": src.count,
                "dtype": src.dtypes[0],
                "nodata": src.nodata,  # Use source nodata to match dtype
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "BIGTIFF": "YES",
            },
        )

        with rasterio.open(dst_path, "w", **ref_profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=resampling,
                )

    log.info("[ALIGN] Saved aligned raster: %s", dst_path)


def prepare_inference_dataset(
    aoi_folder: str,
    output_folder: str | None = None,
    nodata_val: float = -32767,
) -> str:
    """
    Prepare stacked inputs for inference from an AOI folder.

    Given an AOI folder with dtm.tif, dsm.tif, and intensity.tif inside.

    Workflow:
      1. Align DSM and Intensity to DTM
      2. Compute nDSM
      3. Compute TWI
      4. Stack all rasters into stacked_inputs.tif

    Args:
        aoi_folder: Path to AOI folder with elevation data
        output_folder: Where to save processed outputs
        nodata_val: NoData value for output rasters

    Returns:
        Path to the stacked inputs raster

    """
    if output_folder is None:
        output_folder = str(Path(aoi_folder) / "processed")
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Input files (expected names in AOI folder)
    dtm = str(Path(aoi_folder) / "dtm.tif")
    dsm = str(Path(aoi_folder) / "dsm.tif")
    intensity = str(Path(aoi_folder) / "intensity.tif")

    # Aligned versions
    dsm_aligned = str(Path(output_folder) / "dsm_aligned.tif")
    intensity_aligned = str(Path(output_folder) / "intensity_aligned.tif")

    log.info("Aligning inputs to DTM")
    if not Path(dsm_aligned).exists():
        align_to_reference(dtm, dsm, dsm_aligned)
    else:
        log.info("Skipping DSM alignment (already exists)")

    if Path(intensity).exists():
        if not Path(intensity_aligned).exists():
            align_to_reference(dtm, intensity, intensity_aligned)
        else:
            log.info("Skipping Intensity alignment (already exists)")
    else:
        intensity_aligned = None
        log.warning("No intensity.tif found, skipping intensity")

    # Derived rasters
    ndsm_path = str(Path(output_folder) / "ndsm.tif")
    twi_path = str(Path(output_folder) / "twi.tif")
    stacked_path = str(Path(output_folder) / "stacked_inputs.tif")

    # Step 1: nDSM
    if not Path(ndsm_path).exists():
        compute_ndsm(dsm_aligned, dtm, ndsm_path, nodata_val=nodata_val)

    # Step 2: TWI
    if not Path(twi_path).exists():
        compute_twi_whitebox(dtm, twi_path)

    # Step 3: Stack (only include available bands)
    stack_inputs = [twi_path, ndsm_path]
    if intensity_aligned:
        stack_inputs.append(intensity_aligned)

    stack_rasters(stack_inputs, stacked_path, nodata_val=nodata_val)

    log.info("[INFO] Finished stacking: %s", stacked_path)
    return stacked_path


if __name__ == "__main__":
    aoi_folder = (
        "/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/"
        "deep_learning/gdl_projects/gdl-refactor-water-temp/"
        "data/mb_aoi_05OJ001"
    )
    outdir = (
        "/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/"
        "deep_learning/gdl_projects/gdl-refactor-water-temp/"
        "geo_deep_learning/outputs/mb_aoi_05OJ001"
    )

    prepare_inference_dataset(aoi_folder, outdir)

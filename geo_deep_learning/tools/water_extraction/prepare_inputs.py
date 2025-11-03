import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

import fiona
import numpy as np
import pandas as pd
import psutil
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from shapely.geometry import mapping, shape
from shapely.validation import make_valid
from whitebox.whitebox_tools import WhiteboxTools

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
    Compute normalized DSM (nDSM = DSM - DTM) using chunked processing and save as GeoTIFF.

    Args:
        dsm_path (str): Path to aligned DSM raster.
        dtm_path (str): Path to aligned DTM raster.
        output_path (str): Where to write the nDSM raster.
        nodata_val (float): Value to assign to NoData areas in the output.
        block_size (int): Size of processing window (both width and height).

    """
    with rasterio.open(dsm_path) as dsm_src, rasterio.open(dtm_path) as dtm_src:
        assert dsm_src.shape == dtm_src.shape, "DSM and DTM must have the same shape"
        assert dsm_src.transform == dtm_src.transform, "Transforms do not match"
        assert dsm_src.crs == dtm_src.crs, "CRS does not match"

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
    # Debug function
    ram = psutil.virtual_memory()
    tmp = shutil.disk_usage("/tmp")
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


def compute_twi_whitebox(
    dtm_path: str,
    twi_output_path: str,
    temp_dir: str | None = None,
) -> None:
    """
    Compute the Topographic Wetness Index (TWI) from a DTM using WhiteboxTools.

    Args:
        dtm_path (str): Path to the input DTM GeoTIFF.
        twi_output_path (str): Path where the computed TWI raster will be saved.
        temp_dir (str): Folder to store intermediate files. Defaults to TWI's folder.

    """
    # Ensure per-AOI temp dir
    if temp_dir is None:
        temp_dir = Path(twi_output_path).parent / "temp_whitebox"
    temp_dir_path = Path(temp_dir)
    if temp_dir_path.exists():
        shutil.rmtree(temp_dir_path)
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    temp_dir = str(temp_dir_path.resolve())

    # Init Whitebox
    wbt = WhiteboxTools()
    wbt.verbose = True
    wbt.set_working_dir(temp_dir)

    # Absolute paths
    dtm_path = str(Path(dtm_path).resolve())
    twi_output_path = str(Path(twi_output_path).resolve())

    dtm_breached = str(temp_dir_path / "dtm_breached.tif")
    slope_path = str(temp_dir_path / "slope.tif")
    flow_acc_path = str(temp_dir_path / "flow_acc.tif")

    log.info("[TWI] Working dir: %s", temp_dir)
    log.info("[TWI] DTM: %s", dtm_path)
    log.info("[TWI] Output: %s", twi_output_path)

    # Step 1: Breach depressions
    if not Path(dtm_breached).exists():
        log.info("[TWI] Breaching depressions...")
        log_system_resources("before breach_depressions")
        ret = wbt.breach_depressions(dem=dtm_path, output=dtm_breached)
        log_system_resources("after breach_depressions")
        if ret != 0:
            raise RuntimeError(f"[Whitebox] breach_depressions failed with code {ret}")

    # Step 2: Slope
    if not Path(slope_path).exists():
        log.info("[TWI] Computing slope...")
        log_system_resources("before slope")
        ret = wbt.slope(dem=dtm_breached, output=slope_path, zfactor=1.0)
        log_system_resources("after slope")
        if ret != 0:
            raise RuntimeError(f"[Whitebox] slope failed with code {ret}")

    # Step 3: Flow accumulation
    if not Path(flow_acc_path).exists():
        log.info("[TWI] Computing flow accumulation...")
        ret = wbt.d8_flow_accumulation(
            i=dtm_breached,
            output=flow_acc_path,
            out_type="sca",
        )
        if ret != 0:
            raise RuntimeError(
                f"[Whitebox] d8_flow_accumulation failed with code {ret}",
            )

    if not Path(slope_path).exists() or not Path(flow_acc_path).exists():
        raise FileNotFoundError(
            "[Whitebox] One or more required intermediate files are missing.",
        )

    # Step 4: Wetness index
    log.info("[TWI] Running wetness_index...")
    ret = wbt.wetness_index(sca=flow_acc_path, slope=slope_path, output=twi_output_path)
    if ret != 0 or not Path(twi_output_path).exists():
        raise RuntimeError(f"[Whitebox] wetness_index failed. Code: {ret}")

    log.info("[TWI] TWI created: %s", twi_output_path)


def stack_rasters(
    raster_paths: "Sequence[str]",
    output_path: str,
    nodata_val: float = -32767,
) -> None:
    """
    Stack multiple single-band rasters into a multi-band raster.

    Args:
        raster_paths (list of str): Ordered list of raster file paths (TWI, nDSM, Intensity, etc.).
        output_path (str): Path to save the stacked raster.
        nodata_val (float): Value to use for NoData in the output bands.

    """
    sources = [rasterio.open(path) for path in raster_paths]

    # Check alignment
    ref_shape = sources[0].shape
    ref_transform = sources[0].transform
    ref_crs = sources[0].crs
    dtype = sources[0].dtypes[0]

    for src in sources[1:]:
        assert src.shape == ref_shape, f"Shape mismatch: {src.name}"
        assert src.transform == ref_transform, f"Transform mismatch: {src.name}"
        assert src.crs == ref_crs, f"CRS mismatch: {src.name}"

    profile = sources[0].profile
    profile.update(
        {
            "count": len(sources),
            "nodata": nodata_val,
            "BIGTIFF": "YES",
            "compress": "lzw",
        },
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        for i, src in enumerate(sources, start=1):
            band = src.read(1)
            band[band == src.nodata] = nodata_val
            dst.write(band, i)

    for src in sources:
        src.close()


def rasterize_labels_binary_aoi_mask(
    label_vector_path: str,
    aoi_vector_path: str,
    reference_raster_path: str,
    output_path: str,
    burn_value: int = 1,
    fill_value: int = 0,
    ignore_value: int = -1,
) -> None:
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
                from fiona.transform import transform_geom

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
                from fiona.transform import transform_geom

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


# TODO revise if need to be kept LR
def rasterize_labels_binary(
    vector_path: str,
    output_path: str,
    ref_profile: dict,
    burn_value: int = 1,
    fill_value: int = 0,
) -> None:
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
                from fiona.transform import transform_geom

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


def tile_raster_pair(
    input_path: str,
    label_path: str,
    output_dir: str,
    patch_size: int = 256,
    stride: int = 256,
    nodata_val: float = -32767,
    min_valid_ratio: float = 0.9,
) -> None:
    """
    Tile a multi-band input raster and its corresponding single-band label raster into patches.

    Args:
        input_path (str): Path to multi-band raster (e.g., stacked_inputs.tif).
        label_path (str): Path to label raster (e.g., labels.tif).
        output_dir (str): Output folder to save tiled patches.
        patch_size (int): Size of square tiles (patch_size x patch_size).
        stride (int): Step size between tiles.
        nodata_val (float): NoData value to skip invalid tiles.
        min_valid_ratio (float): Minimum ratio of valid pixels required to keep a tile.

    """
    Path(output_dir, "inputs").mkdir(parents=True, exist_ok=True)
    Path(output_dir, "labels").mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src_input, rasterio.open(label_path) as src_label:
        assert (
            src_input.width == src_label.width and src_input.height == src_label.height
        ), "Input and label size mismatch"
        assert src_input.transform == src_label.transform, (
            "Input and label geotransform mismatch"
        )

        tile_id = 0
        for y in range(0, src_input.height - patch_size + 1, stride):
            for x in range(0, src_input.width - patch_size + 1, stride):
                window = Window(x, y, patch_size, patch_size)

                input_patch = src_input.read(window=window)
                label_patch = src_label.read(1, window=window)

                # Check for too much NoData
                valid_mask = label_patch != nodata_val
                if np.mean(valid_mask) < min_valid_ratio:
                    continue

                # Save input tile
                input_meta = src_input.meta.copy()
                input_meta.update(
                    {
                        "height": patch_size,
                        "width": patch_size,
                        "transform": rasterio.windows.transform(
                            window,
                            src_input.transform,
                        ),
                    },
                )
                input_tile_path = (
                    Path(output_dir) / "inputs" / f"tile_{tile_id:04d}.tif"
                )
                with rasterio.open(input_tile_path, "w", **input_meta) as dst:
                    dst.write(input_patch)

                # Save label tile
                label_meta = src_label.meta.copy()
                label_meta.update(
                    {
                        "count": 1,
                        "height": patch_size,
                        "width": patch_size,
                        "transform": rasterio.windows.transform(
                            window,
                            src_label.transform,
                        ),
                    },
                )
                label_tile_path = (
                    Path(output_dir) / "labels" / f"tile_{tile_id:04d}_label.tif"
                )
                with rasterio.open(label_tile_path, "w", **label_meta) as dst:
                    dst.write(label_patch, 1)

                tile_id += 1

    log.info("Tiling complete. %d tiles created in %s", tile_id, output_dir)


def generate_csv_from_tiles(
    root_output_folder: str,
    csv_tiling_path: str,
    csv_inference_path: str,
    test_ratio: float = 0.2,
    remove_empty_labels: bool = False,
) -> None:
    """
    Generate training/validation and inference CSVs from pre-tiled input/label folders.

    Args:
        root_output_folder (str): Path to folder containing AOI subfolders (each with `tiles/inputs/` and `tiles/labels/`).
        csv_tiling_path (str): Where to save the full tile CSV.
        csv_inference_path (str): Where to save the inference subset CSV.
        test_ratio (float): Ratio of tiles to allocate to the test split.

    """
    all_rows = []

    aoi_folders = sorted(
        [str(p) for p in Path(root_output_folder).iterdir() if p.is_dir()],
    )

    log.info("Aoi_folders %s", aoi_folders)
    for aoi_folder in aoi_folders:
        log.info("Aoi_folder %s", aoi_folder)
        aoi_name = Path(aoi_folder).name
        inputs_folder = Path(aoi_folder) / "tiles" / "inputs"
        labels_folder = Path(aoi_folder) / "tiles" / "labels"

        input_tiles = sorted(str(p) for p in inputs_folder.glob("*.tif"))

        log.info("Filtering tiles with 0 labeled pixel")
        filetered_count = 0
        for input_path in input_tiles:
            tile_id = Path(input_path).stem.split("_")[1]
            label_path = labels_folder / f"tile_{tile_id}_label.tif"

            if not label_path.exists():
                log.warning("No matching label for: %s", input_path)
                continue

            # Filter out tiles that have no valid labels (only -1 or 0)
            with rasterio.open(label_path) as src:
                label_data = src.read(1)
                valid_mask = label_data >= 1

                if not np.any(valid_mask):
                    filetered_count += 1
                    if remove_empty_labels:
                        Path(input_path).unlink()
                        label_path.unlink()
                    continue

            all_rows.append({"tif": input_path, "gpkg": label_path, "aoi": aoi_name})

        if remove_empty_labels:
            log.info("Removed %d empty-labeled tile(s).", filetered_count)

    if not all_rows:
        raise ValueError("No labeled tiles found. Check data or label filtering logic.")

    df = pd.DataFrame(all_rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Assign split
    # Assign split with trn/val/tst
    val_ratio = test_ratio  # Same size as test set; customize if needed
    train_ratio = 1 - test_ratio - val_ratio

    num_tiles = len(df)
    num_train = int(np.floor(num_tiles * train_ratio))
    num_val = int(np.floor(num_tiles * val_ratio))
    num_test = num_tiles - num_train - num_val

    split_list = ["trn"] * num_train + ["val"] * num_val + ["tst"] * num_test
    df["split"] = split_list
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # DEBUG
    # df['split'] = 'trn'
    # test_every = int(1 / test_ratio)
    # df.loc[::test_every, 'split'] = 'tst'

    # Save full CSV
    Path(csv_tiling_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_tiling_path, index=False)
    log.info("Saved full tiling CSV to: %s", csv_tiling_path)

    # Save inference subset
    df_infer = df[df["split"] == "tst"].copy()
    df_infer["split"] = "inference"
    Path(csv_inference_path).parent.mkdir(parents=True, exist_ok=True)
    df_infer.to_csv(csv_inference_path, index=False)
    log.info("Saved inference CSV to: %s", csv_inference_path)


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
    Prepare stacked inputs for inference given an AOI folder
    with dtm.tif, dsm.tif, and intensity.tif inside.

    Workflow:
      1. Align DSM and Intensity to DTM
      2. Compute nDSM
      3. Compute TWI
      4. Stack all rasters into stacked_inputs.tif
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
    aoi_folder = "/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/gdl_projects/gdl-refactor-water-temp/data/mb_aoi_05OJ001"
    outdir = "/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/gdl_projects/gdl-refactor-water-temp/geo_deep_learning/outputs/mb_aoi_05OJ001"

    prepare_inference_dataset(aoi_folder, outdir)

#!/bin/bash
# Usage:
#   sbatch slurm/run_full_aoi_pipeline.sh /path/to/aoi_folder
#   WORKFLOW=inference sbatch slurm/run_full_aoi_pipeline.sh /path/to/aoi_folder

#SBATCH --job-name=water_full_aoi
#SBATCH --partition=standard
#SBATCH --account=nrcan_geobase
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%j_full_aoi_pipeline.out
#SBATCH --error=slurm/logs/%j_full_aoi_pipeline.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=luca.romanini@nrcan-rncan.gc.ca
#SBATCH --comment="image=registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu24.04,tmpfs_size=250G"
#SBATCH --qos=low

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch slurm/run_full_aoi_pipeline.sh /path/to/aoi_folder" >&2
    echo "       WORKFLOW=inference sbatch slurm/run_full_aoi_pipeline.sh /path/to/aoi_folder" >&2
    exit 1
fi

SOURCE_AOI_DIR="$(realpath "$1")"
if [[ ! -d "$SOURCE_AOI_DIR" ]]; then
    echo "AOI folder not found: $SOURCE_AOI_DIR" >&2
    exit 1
fi

mkdir -p slurm/logs

# ── Environment ───────────────────────────────────────────────
export https_proxy="${https_proxy:-http://webproxy.science.gc.ca:8888/}"
export http_proxy="${http_proxy:-http://webproxy.science.gc.ca:8888/}"
export PYTHONUNBUFFERED=1
export TMPDIR="${TMPDIR:-/gpfs/fs5/nrcan/nrcan_geobase/gdl_tmp}"

# ── Core paths ────────────────────────────────────────────────
GDL_REPO="${GDL_REPO:-/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/gdl_projects/geo-deep-learning}"
INTENSITY_REPO="${INTENSITY_REPO:-/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/lidar/utils/intensity_pipeline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$GDL_REPO/data}"
WORKFLOW="${WORKFLOW:-training}"
AOI_VECTOR_STEM="${AOI_VECTOR_STEM:-}"

GDL_CONDA_SH="${GDL_CONDA_SH:-/space/partner/nrcan/geobase/work/opt/miniconda-gdl-ops/etc/profile.d/conda.sh}"
GDL_ENV="${GDL_ENV:-gdl_env}"
INTENSITY_CONDA_SH="${INTENSITY_CONDA_SH:-/space/partner/nrcan/geobase/work/opt/miniconda-geoai/etc/profile.d/conda.sh}"
INTENSITY_ENV="${INTENSITY_ENV:-lidar_utils}"

# ── Input dependencies ────────────────────────────────────────
REFERENCE_STANDARDS="${REFERENCE_STANDARDS:-$INTENSITY_REPO/config/reference_standards.json}"
SITESTORE_BASE="${SITESTORE_BASE:-}"
PROJECT_INDEX="${PROJECT_INDEX:-/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/lidar/utils/input_data/index_lidar/projet_lidar_infos_detaillees_2.gpkg}"
TILES_INDEX="${TILES_INDEX:-/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/lidar/utils/input_data/index_lidar/all_project_tiles_lidar.gpkg}"

# ── Processing parameters ─────────────────────────────────────
RESOLUTION="${RESOLUTION:-1}"
TARGET_CRS="${TARGET_CRS:-EPSG:3979}"
NADIR_RESOLUTION="${NADIR_RESOLUTION:-10}"
NADIR_CHUNK_SIZE="${NADIR_CHUNK_SIZE:-5000000}"
INTENSITY_PROCESSING_MODE="${INTENSITY_PROCESSING_MODE:-baseline}"
INTENSITY_DARK_ZONE_MODE="${INTENSITY_DARK_ZONE_MODE:-false}"
INTENSITY_OVERLAP_MODE="${INTENSITY_OVERLAP_MODE:-false}"
PATCH_SIZE="${PATCH_SIZE:-512}"
STRIDE="${STRIDE:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-16}"
TEST_RATIO="${TEST_RATIO:-0.2}"
MIN_WATER_PIXELS="${MIN_WATER_PIXELS:-1}"
ZERO_MIN_COMPONENT_SIZE="${ZERO_MIN_COMPONENT_SIZE:-10}"
ZERO_MAX_HOLE_SIZE="${ZERO_MAX_HOLE_SIZE:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

if [[ -z "$AOI_VECTOR_STEM" ]]; then
    if [[ "$WORKFLOW" == "inference" ]]; then
        AOI_VECTOR_STEM="aoi_buffered"
    else
        AOI_VECTOR_STEM="aoi"
    fi
fi

AOI_NAME="$(basename "$SOURCE_AOI_DIR")"
WORKFLOW_ROOT="$OUTPUT_ROOT/$WORKFLOW"
RAW_ROOT="$WORKFLOW_ROOT/raw"
PREP_ROOT="$WORKFLOW_ROOT/preprocessed"
RUN_ROOT="$WORKFLOW_ROOT/runs"
RAW_AOI_DIR="$RAW_ROOT/$AOI_NAME"
RUN_DIR="$RUN_ROOT/$AOI_NAME"
EXTRACT_DIR="$RUN_DIR/extracted_laz"
INTENSITY_RESULTS="$RUN_DIR/intensity_results"
NADIR_DIR="$RUN_DIR/nadir_outputs"
INTENSITY_CONFIG="$RUN_DIR/intensity_pipeline_config.yaml"
PREP_CONFIG="$RUN_DIR/prepare_data_config.yaml"
ZERO_SOURCE="$RUN_DIR/zero_intensity_source.tif"

mkdir -p "$RAW_AOI_DIR" "$PREP_ROOT" "$RUN_DIR" "$NADIR_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

is_valid_geotiff() {
    local raster_path="$1"
    [[ -f "$raster_path" ]] || return 1
    python - "$raster_path" <<'PY' >/dev/null 2>&1
import sys
import rasterio

path = sys.argv[1]
try:
    with rasterio.open(path) as src:
        ok = src.count > 0 and src.width > 0 and src.height > 0
        sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PY
}

promote_file() {
    local temp_path="$1"
    local final_path="$2"
    mv -f "$temp_path" "$final_path"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file not found: $1" >&2
        exit 1
    fi
}

resolve_sitestore_base() {
    local candidates=()

    if [[ -n "$SITESTORE_BASE" ]]; then
        candidates+=("$SITESTORE_BASE")
    fi

    candidates+=(
        "/space/partner/nrcan/geobase/sitestore/elevation/lidar"
        "/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/sitestore/data/lidar"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -d "$candidate" && -r "$candidate" && -x "$candidate" ]]; then
            SITESTORE_BASE="$candidate"
            return 0
        fi
    done

    echo "Could not resolve a readable sitestore base. Tried:" >&2
    printf '  %s\n' "${candidates[@]}" >&2
    exit 1
}

find_vector() {
    local folder="$1"
    local stem="$2"
    if [[ -f "$folder/$stem.gpkg" ]]; then
        echo "$folder/$stem.gpkg"
        return 0
    fi
    if [[ -f "$folder/$stem.shp" ]]; then
        echo "$folder/$stem.shp"
        return 0
    fi
    return 1
}

copy_vector_family() {
    local source="$1"
    local target_dir="$2"
    local stem ext source_dir target_dir_real
    stem="$(basename "${source%.*}")"
    ext="${source##*.}"
    source_dir="$(realpath "$(dirname "$source")")"
    target_dir_real="$(realpath "$target_dir")"

    if [[ "$source_dir" == "$target_dir_real" ]]; then
        return 0
    fi

    if [[ "$ext" == "shp" ]]; then
        cp -f "$(dirname "$source")/$stem".* "$target_dir"/
    else
        cp -f "$source" "$target_dir"/
    fi
}

activate_gdl() {
    # shellcheck disable=SC1090
    set +u
    source "$GDL_CONDA_SH"
    conda activate "$GDL_ENV"
    set -u
    cd "$GDL_REPO"
    export PYTHONPATH="$GDL_REPO"
}

activate_intensity() {
    # shellcheck disable=SC1090
    set +u
    source "$INTENSITY_CONDA_SH"
    conda activate "$INTENSITY_ENV"
    set -u
    cd "$INTENSITY_REPO"
    export PYTHONPATH="$INTENSITY_REPO"
}

AOI_VECTOR="$(find_vector "$SOURCE_AOI_DIR" "$AOI_VECTOR_STEM" || true)"
if [[ -z "${AOI_VECTOR:-}" ]]; then
    echo "Missing AOI vector in $SOURCE_AOI_DIR (expected $AOI_VECTOR_STEM.gpkg or $AOI_VECTOR_STEM.shp)" >&2
    exit 1
fi

WATERBODIES_VECTOR=""
if [[ "$WORKFLOW" == "training" ]]; then
    WATERBODIES_VECTOR="$(find_vector "$SOURCE_AOI_DIR" waterbodies || true)"
    if [[ -z "${WATERBODIES_VECTOR:-}" ]]; then
        echo "Missing waterbodies vector in $SOURCE_AOI_DIR (expected waterbodies.gpkg or waterbodies.shp)" >&2
        exit 1
    fi
fi

require_file "$REFERENCE_STANDARDS"
require_file "$PROJECT_INDEX"
require_file "$TILES_INDEX"
require_file "$GDL_CONDA_SH"
require_file "$INTENSITY_CONDA_SH"
resolve_sitestore_base

copy_vector_family "$AOI_VECTOR" "$RAW_AOI_DIR"
if [[ -n "$WATERBODIES_VECTOR" ]]; then
    copy_vector_family "$WATERBODIES_VECTOR" "$RAW_AOI_DIR"
fi

RAW_AOI_VECTOR="$(find_vector "$RAW_AOI_DIR" "$AOI_VECTOR_STEM")"

echo "============================================"
echo "Job ID     : ${SLURM_JOB_ID:-N/A}"
echo "Node       : ${SLURM_NODELIST:-N/A}"
echo "Started    : $(date)"
echo "GDL Repo   : $GDL_REPO"
echo "Intensity  : $INTENSITY_REPO"
echo "AOI Source : $SOURCE_AOI_DIR"
echo "AOI Name   : $AOI_NAME"
echo "AOI Stem   : $AOI_VECTOR_STEM"
echo "Raw AOI    : $RAW_AOI_DIR"
echo "Prep Root  : $PREP_ROOT"
echo "Run Dir    : $RUN_DIR"
echo "Sitestore  : $SITESTORE_BASE"
echo "============================================"

log "Step 1/7: Download DTM and DSM"
activate_gdl
if [[ "$FORCE_RERUN" != "1" && $(is_valid_geotiff "$RAW_AOI_DIR/dtm.tif"; echo $?) -eq 0 && $(is_valid_geotiff "$RAW_AOI_DIR/dsm.tif"; echo $?) -eq 0 ]]; then
    log "Skipping DTM/DSM download: existing rasters validated"
else
    ELEVATION_STAGE_DIR="$RUN_DIR/elevation_temp"
    rm -rf "$ELEVATION_STAGE_DIR"
    mkdir -p "$ELEVATION_STAGE_DIR"

    python -m geo_deep_learning.tools.water_extraction.download_elevation \
        --aoi_path "$RAW_AOI_VECTOR" \
        --out_folder "$ELEVATION_STAGE_DIR" \
        --resolution "$RESOLUTION"

    if ! is_valid_geotiff "$ELEVATION_STAGE_DIR/dtm.tif"; then
        echo "Downloaded DTM is missing or invalid: $ELEVATION_STAGE_DIR/dtm.tif" >&2
        exit 1
    fi
    if ! is_valid_geotiff "$ELEVATION_STAGE_DIR/dsm.tif"; then
        echo "Downloaded DSM is missing or invalid: $ELEVATION_STAGE_DIR/dsm.tif" >&2
        exit 1
    fi

    promote_file "$ELEVATION_STAGE_DIR/dtm.tif" "$RAW_AOI_DIR/dtm.tif"
    promote_file "$ELEVATION_STAGE_DIR/dsm.tif" "$RAW_AOI_DIR/dsm.tif"
    rm -rf "$ELEVATION_STAGE_DIR"
fi

log "Step 2/7: Create valid_lidar_mask.gpkg"
if [[ "$FORCE_RERUN" != "1" && -s "$RAW_AOI_DIR/valid_lidar_mask.gpkg" ]]; then
    log "Skipping valid_lidar_mask creation: existing file found"
else
    rm -f "$RAW_AOI_DIR"/valid_lidar_mask.gpkg "$RAW_AOI_DIR"/valid_lidar_mask.gpkg-shm "$RAW_AOI_DIR"/valid_lidar_mask.gpkg-wal
    python -m geo_deep_learning.tools.water_extraction.create_valid_lidar_mask \
        --aoi_folder "$RAW_AOI_DIR" \
        --lidar_index "$PROJECT_INDEX"
fi

log "Step 3/7: Extract intersecting LAZ files"
activate_intensity
if [[ "$FORCE_RERUN" != "1" && -s "$EXTRACT_DIR/extraction_metadata.csv" ]]; then
    log "Skipping LAZ extraction: existing extraction metadata found"
else
    rm -rf "$EXTRACT_DIR"
    python extract_laz_by_aoi.py \
        "$RAW_AOI_VECTOR" \
        --sitestore-base "$SITESTORE_BASE" \
        --project-index "$PROJECT_INDEX" \
        --tiles-index "$TILES_INDEX" \
        --output-dir "$RUN_DIR" \
        --folder-name "$(basename "$EXTRACT_DIR")" \
        --mosaic-name "$AOI_NAME"
fi

mapfile -t PROJECT_DIRS < <(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ ${#PROJECT_DIRS[@]} -eq 0 ]]; then
    echo "No extracted LiDAR project folders found under $EXTRACT_DIR" >&2
    exit 1
fi

log "Step 3b/7: Detect and repair missing CRS metadata on run-local extracted LAZ copies only"
log "Step 3b/7: Sitestore source LAZ files are never modified"
python "$GDL_REPO/scripts/repair_extracted_laz_crs.py" \
    --metadata-csv "$EXTRACT_DIR/extraction_metadata.csv" \
    --project-index "$PROJECT_INDEX"

log "Step 4/7: Build and run intensity pipeline config"
cat > "$INTENSITY_CONFIG" <<EOF
reference:
  standards_file: "$REFERENCE_STANDARDS"

preprocessing:
  enabled: false
  sitestore_base_dir: "$SITESTORE_BASE"
  project_index_gpkg: "$PROJECT_INDEX"
  tiles_index_gpkg: "$TILES_INDEX"
  output_dir: "$RUN_DIR"

global:
  resolution: ${RESOLUTION}.0
  target_crs: "$TARGET_CRS"
  workers: $NUM_WORKERS
  temp_dir: "/tmp/lidar_pipeline"
  nodata_intermediate: -9999.0
  nodata_final: 255

projects:
EOF

for project_dir in "${PROJECT_DIRS[@]}"; do
    project_name="$(basename "$project_dir")"
    cat >> "$INTENSITY_CONFIG" <<EOF
  - name: "$project_name"
    input_dir: "$project_dir"
    processing_mode: "$INTENSITY_PROCESSING_MODE"
    aoi: "$RAW_AOI_VECTOR"
    enabled: true
EOF
done

cat >> "$INTENSITY_CONFIG" <<EOF

output:
  results: "$INTENSITY_RESULTS"
  mosaic_name: "$AOI_NAME"
  create_vrt: true
  create_physical_mosaic: true
  keep_intermediate: false
  compression: "DEFLATE"

processing:
  save_intermediate: true
  dark_zone: ${INTENSITY_DARK_ZONE_MODE}
  dark_zone_config:
    gamma_boost: 0.70
    offset_multiplier: 1.0
    dark_percentile: 30
  overlap: ${INTENSITY_OVERLAP_MODE}
  overlap_config:
    scan_angle_threshold: 30.0
    aggregation: "median"
  baseline:
    aggregation: "mean"

qa:
  min_coverage_percent: 80
  max_edge_gradient: 50
  generate_thumbnails: true
  thumbnail_resolution: 0.1
EOF

if [[ "$FORCE_RERUN" != "1" && $(is_valid_geotiff "$RAW_AOI_DIR/intensity.tif"; echo $?) -eq 0 ]]; then
    log "Skipping intensity pipeline: existing intensity.tif validated"
else
    python main.py --config "$INTENSITY_CONFIG"

    FINAL_INTENSITY="$(ls -t "$INTENSITY_RESULTS"/mosaics/"${AOI_NAME}"*.tif 2>/dev/null | head -n 1 || true)"
    if [[ -z "${FINAL_INTENSITY:-}" ]]; then
        echo "Could not find final intensity mosaic under $INTENSITY_RESULTS/mosaics" >&2
        exit 1
    fi
    cp -f "$FINAL_INTENSITY" "$RAW_AOI_DIR/intensity_temp.tif"
    if ! is_valid_geotiff "$RAW_AOI_DIR/intensity_temp.tif"; then
        echo "Copied intensity raster is invalid: $RAW_AOI_DIR/intensity_temp.tif" >&2
        exit 1
    fi
    promote_file "$RAW_AOI_DIR/intensity_temp.tif" "$RAW_AOI_DIR/intensity.tif"
fi

log "Step 5/7: Create zero-intensity and nadir-weighted rasters"
if [[ "$FORCE_RERUN" != "1" && $(is_valid_geotiff "$RAW_AOI_DIR/zero_intensity.tif"; echo $?) -eq 0 ]]; then
    log "Skipping zero-intensity generation: existing zero_intensity.tif validated"
else
    activate_gdl
    python "$GDL_REPO/geo_deep_learning/tools/water_extraction/extract_zero_intensity_mask.py" \
        --input-raster "$RAW_AOI_DIR/intensity.tif" \
        --output-raster "$ZERO_SOURCE" \
        --min-component-size "$ZERO_MIN_COMPONENT_SIZE" \
        --max-hole-size "$ZERO_MAX_HOLE_SIZE" \
        --overwrite
fi

if [[ "$FORCE_RERUN" != "1" && $(is_valid_geotiff "$RAW_AOI_DIR/nadir_weighted.tif"; echo $?) -eq 0 ]]; then
    log "Skipping nadir-weighted generation: existing nadir_weighted.tif validated"
else
    activate_intensity
    python make_nadir_risk_raster.py \
        --config "$INTENSITY_CONFIG" \
        --name "$AOI_NAME" \
        --output-dir "$NADIR_DIR" \
        --output-mode weighted_only \
        --resolution "$NADIR_RESOLUTION" \
        --workers "$NUM_WORKERS" \
        --chunk-size "$NADIR_CHUNK_SIZE" \
        --overwrite
fi

NADIR_SOURCE="$NADIR_DIR/${AOI_NAME}_nadir_weighted_${NADIR_RESOLUTION}m.tif"
if [[ ! -f "$RAW_AOI_DIR/nadir_weighted.tif" ]]; then
    require_file "$NADIR_SOURCE"
fi

log "Step 6/7: Align extra rasters to DTM in gdl_env"
activate_gdl
ZERO_VALID=0
NADIR_VALID=0
if is_valid_geotiff "$RAW_AOI_DIR/zero_intensity.tif"; then
    ZERO_VALID=1
fi
if is_valid_geotiff "$RAW_AOI_DIR/nadir_weighted.tif"; then
    NADIR_VALID=1
fi

if [[ "$FORCE_RERUN" != "1" && "$ZERO_VALID" -eq 1 && "$NADIR_VALID" -eq 1 ]]; then
    log "Skipping raster alignment: existing aligned extra rasters validated"
else
    if [[ "$FORCE_RERUN" == "1" || "$ZERO_VALID" -ne 1 ]]; then
python - <<PY
from rasterio.enums import Resampling
from geo_deep_learning.tools.water_extraction.prepare_inputs import align_to_reference

align_to_reference(
    r"$RAW_AOI_DIR/dtm.tif",
    r"$ZERO_SOURCE",
    r"$RAW_AOI_DIR/zero_intensity_temp.tif",
    resampling=Resampling.nearest,
)
PY
        if ! is_valid_geotiff "$RAW_AOI_DIR/zero_intensity_temp.tif"; then
            echo "Aligned zero-intensity raster is invalid: $RAW_AOI_DIR/zero_intensity_temp.tif" >&2
            exit 1
        fi
        promote_file "$RAW_AOI_DIR/zero_intensity_temp.tif" "$RAW_AOI_DIR/zero_intensity.tif"
    else
        log "Skipping zero-intensity alignment: existing aligned raster validated"
    fi

    if [[ "$FORCE_RERUN" == "1" || "$NADIR_VALID" -ne 1 ]]; then
python - <<PY
from rasterio.enums import Resampling
from geo_deep_learning.tools.water_extraction.prepare_inputs import align_to_reference

align_to_reference(
    r"$RAW_AOI_DIR/dtm.tif",
    r"$NADIR_SOURCE",
    r"$RAW_AOI_DIR/nadir_weighted_temp.tif",
    resampling=Resampling.bilinear,
)
PY
        if ! is_valid_geotiff "$RAW_AOI_DIR/nadir_weighted_temp.tif"; then
            echo "Aligned nadir-weighted raster is invalid: $RAW_AOI_DIR/nadir_weighted_temp.tif" >&2
            exit 1
        fi
        promote_file "$RAW_AOI_DIR/nadir_weighted_temp.tif" "$RAW_AOI_DIR/nadir_weighted.tif"
    else
        log "Skipping nadir-weighted alignment: existing aligned raster validated"
    fi
fi

log "Step 7/7: Run existing water-extraction preprocessing"
TEST_ONLY=false
if [[ "$WORKFLOW" == "inference" ]]; then
    TEST_ONLY=true
fi

cat > "$PREP_CONFIG" <<EOF
data:
  class_path: geo_deep_learning.tools.water_extraction.elevation_stack_datamodule.ElevationStackDataModule
  init_args:
    csv_root_folder: "$PREP_ROOT"
    patches_root_folder: "$PREP_ROOT"
    input_folders:
      - "$RAW_AOI_DIR"
    output_root: "$PREP_ROOT"
    csv_path: "$PREP_ROOT/${AOI_NAME}_water_extraction.csv"
    csv_infer_path: "$PREP_ROOT/${AOI_NAME}_water_extraction_infer.csv"
    workflow: "$WORKFLOW"
    include_intensity: true
    extra_input_rasters:
      - "nadir_weighted.tif"
      - "zero_intensity.tif"
    stride: $STRIDE
    patch_size: [$PATCH_SIZE, $PATCH_SIZE]
    batch_size: $BATCH_SIZE
    num_workers: $NUM_WORKERS
    regenerate_csv: false
    min_water_pixels: $MIN_WATER_PIXELS
    test_ratio: $TEST_RATIO
    test_only: $TEST_ONLY
EOF

STACKED_OUTPUT="$PREP_ROOT/$AOI_NAME/stacked_inputs.tif"
python -m geo_deep_learning.tools.water_extraction.prepare_data \
    --config "$PREP_CONFIG"

echo "============================================"
echo "Finished : $(date)"
echo "Intensity raster      : $RAW_AOI_DIR/intensity.tif"
echo "Zero-intensity raster : $RAW_AOI_DIR/zero_intensity.tif"
echo "Nadir-weighted raster : $RAW_AOI_DIR/nadir_weighted.tif"
echo "Preprocessed outputs  : $PREP_ROOT/$AOI_NAME"
echo "============================================"

#!/bin/bash
# Usage:
#   WORKFLOW=inference sbatch slurm/run_full_aoi_pipeline_registry_highqos.sh /path/to/aoi_folder
# Optional:
#   MODEL_IDS="model_1 model_3 model_7" sbatch slurm/run_full_aoi_pipeline_registry_highqos.sh /path/to/aoi_folder

#SBATCH --job-name=water_full_aoi_registry
#SBATCH --partition=gpu_a100
#SBATCH --account=nrcan_geobase__gpu_a100
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%j_full_aoi_pipeline_registry.out
#SBATCH --error=slurm/logs/%j_full_aoi_pipeline_registry.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=luca.romanini@nrcan-rncan.gc.ca
#SBATCH --comment="image=registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu24.04,tmpfs_size=250G"
#SBATCH --qos=nrcan__gpu_a100_gpsc7_allocation

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch slurm/run_full_aoi_pipeline_registry_highqos.sh /path/to/aoi_folder" >&2
    echo "       WORKFLOW=inference sbatch slurm/run_full_aoi_pipeline_registry_highqos.sh /path/to/aoi_folder" >&2
    exit 1
fi

SOURCE_AOI_DIR="$(realpath "$1")"
if [[ ! -d "$SOURCE_AOI_DIR" ]]; then
    echo "AOI folder not found: $SOURCE_AOI_DIR" >&2
    exit 1
fi

mkdir -p slurm/logs

export https_proxy="${https_proxy:-http://webproxy.science.gc.ca:8888/}"
export http_proxy="${http_proxy:-http://webproxy.science.gc.ca:8888/}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export no_proxy="${no_proxy:+$no_proxy,}datacube.services.geo.ca"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}datacube.services.geo.ca"
export PYTHONUNBUFFERED=1
export TMPDIR="${TMPDIR:-/gpfs/fs5/nrcan/nrcan_geobase/gdl_tmp}"

GDL_REPO="${GDL_REPO:-/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/gdl_projects/geo-deep-learning}"
INTENSITY_REPO="${INTENSITY_REPO:-/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/lidar/utils/intensity_pipeline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$GDL_REPO/data}"
RESULTS_ROOT="${RESULTS_ROOT:-$GDL_REPO/results/inference_registry}"
WORKFLOW="${WORKFLOW:-inference}"
AOI_VECTOR_STEM="${AOI_VECTOR_STEM:-aoi}"
BUFFERED_AOI_STEM="${BUFFERED_AOI_STEM:-aoi_buffered}"
BUFFER_METERS="${BUFFER_METERS:-1000}"
RUN_REGISTERED_INFERENCE="${RUN_REGISTERED_INFERENCE:-1}"
MODEL_IDS="${MODEL_IDS:-}"
MODEL_REGISTRY="${MODEL_REGISTRY:-$GDL_REPO/configs/inference_models.yaml}"

GDL_CONDA_SH="${GDL_CONDA_SH:-/space/partner/nrcan/geobase/work/opt/miniconda-gdl-ops/etc/profile.d/conda.sh}"
GDL_ENV="${GDL_ENV:-gdl_env}"
INTENSITY_CONDA_SH="${INTENSITY_CONDA_SH:-/space/partner/nrcan/geobase/work/opt/miniconda-geoai/etc/profile.d/conda.sh}"
INTENSITY_ENV="${INTENSITY_ENV:-lidar_utils}"

REFERENCE_STANDARDS="${REFERENCE_STANDARDS:-$INTENSITY_REPO/config/reference_standards.json}"
SITESTORE_BASE="${SITESTORE_BASE:-}"
PROJECT_INDEX="${PROJECT_INDEX:-/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/lidar/utils/input_data/index_lidar/projet_lidar_infos_detaillees_2.gpkg}"
TILES_INDEX="${TILES_INDEX:-/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/lidar/utils/input_data/index_lidar/all_project_tiles_lidar.gpkg}"

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
INFERENCE_DEVICE="${INFERENCE_DEVICE:-cuda}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
INFERENCE_WINDOW_SIZE="${INFERENCE_WINDOW_SIZE:-512}"
INFERENCE_STRIDE="${INFERENCE_STRIDE:-256}"

AOI_NAME="$(basename "$SOURCE_AOI_DIR")"
WORKFLOW_ROOT="$OUTPUT_ROOT/$WORKFLOW"
RAW_ROOT="$WORKFLOW_ROOT/raw"
PREP_ROOT="$WORKFLOW_ROOT/preprocessed"
RUN_ROOT="$WORKFLOW_ROOT/runs"
RAW_AOI_DIR="$RAW_ROOT/$AOI_NAME"
RUN_DIR="$RUN_ROOT/$AOI_NAME"
ELEVATION_CACHE_DIR="$RUN_DIR/elevation_cache"
EXTRACT_DIR="$RUN_DIR/extracted_laz"
INTENSITY_RESULTS="$RUN_DIR/intensity_results"
NADIR_DIR="$RUN_DIR/nadir_outputs"
INTENSITY_CONFIG="$RUN_DIR/intensity_pipeline_config.yaml"
PREP_CONFIG="$RUN_DIR/prepare_data_config.yaml"
ZERO_SOURCE="$RUN_DIR/zero_intensity_source.tif"
MODEL_RESULTS_ROOT="$RESULTS_ROOT/$AOI_NAME"

mkdir -p "$RAW_AOI_DIR" "$PREP_ROOT" "$RUN_DIR" "$NADIR_DIR" "$MODEL_RESULTS_ROOT"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

sanitize_tag() {
    local raw_value="${1:-}"
    local fallback="${2:-unknown}"
    local sanitized

    sanitized="$(printf '%s' "$raw_value" | tr '[:space:]/:' '___' | tr -cd '[:alnum:]_.-')"
    sanitized="${sanitized#[-._]}"
    sanitized="${sanitized%[-._]}"

    if [[ -z "$sanitized" ]]; then
        sanitized="$fallback"
    fi

    printf '%.80s\n' "$sanitized"
}

setup_dynamic_slurm_log() {
    local aoi_tag model_tag job_suffix job_name log_dir dynamic_log

    aoi_tag="$(sanitize_tag "$AOI_NAME" "unknown_aoi")"
    model_tag="$(sanitize_tag "${MODEL_IDS:-all_models}" "all_models")"
    job_suffix="${aoi_tag}__${model_tag}"
    job_name="water_full_aoi_registry__${job_suffix}"
    log_dir="$GDL_REPO/slurm/logs"
    dynamic_log="$log_dir/${SLURM_JOB_ID:-nojob}_${job_suffix}_full_aoi_pipeline_registry.out"

    mkdir -p "$log_dir"

    if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
        scontrol update JobId="$SLURM_JOB_ID" JobName="$job_name" >/dev/null 2>&1 || true
        scontrol update JobId="$SLURM_JOB_ID" StdOut="$dynamic_log" StdErr="$dynamic_log" >/dev/null 2>&1 || true
    fi

    export SLURM_DYNAMIC_LOG_PATH="$dynamic_log"
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
    set +u
    # shellcheck disable=SC1090
    source "$GDL_CONDA_SH"
    conda activate "$GDL_ENV"
    set -u
    cd "$GDL_REPO"
    export PYTHONPATH="$GDL_REPO"
}

activate_intensity() {
    set +u
    # shellcheck disable=SC1090
    source "$INTENSITY_CONDA_SH"
    conda activate "$INTENSITY_ENV"
    set -u
    cd "$INTENSITY_REPO"
    export PYTHONPATH="$INTENSITY_REPO"
}

resolve_model_ids() {
    if [[ -n "$MODEL_IDS" ]]; then
        printf '%s\n' $MODEL_IDS
        return 0
    fi

    python - "$MODEL_REGISTRY" <<'PY'
import sys
from pathlib import Path
import yaml

registry = Path(sys.argv[1])
with registry.open(encoding="utf-8") as file:
    payload = yaml.safe_load(file) or {}

models = payload.get("models", {})
for model_id in sorted(models):
    print(model_id)
PY
}

resolve_requested_features() {
    python - "$MODEL_REGISTRY" ${MODEL_IDS:-} <<'PY'
import sys
from pathlib import Path
import yaml

registry = Path(sys.argv[1])
requested_model_ids = sys.argv[2:]
with registry.open(encoding="utf-8") as file:
    payload = yaml.safe_load(file) or {}

models = payload.get("models", {})
if requested_model_ids:
    model_ids = requested_model_ids
else:
    model_ids = sorted(models)

feature_order = ["twi", "ndsm", "dtm", "intensity", "nadir_weighted", "zero_intensity"]
selected = []
for model_id in model_ids:
    model_cfg = models.get(model_id)
    if not isinstance(model_cfg, dict):
        continue
    for feature in model_cfg.get("features", []):
        if feature in feature_order and feature not in selected:
            selected.append(feature)

for feature in selected:
    print(feature)
PY
}

resolve_model_features() {
    local model_id="$1"
    python - "$MODEL_REGISTRY" "$model_id" <<'PY'
import sys
from pathlib import Path
import yaml

registry = Path(sys.argv[1])
model_id = sys.argv[2]
with registry.open(encoding="utf-8") as file:
    payload = yaml.safe_load(file) or {}

models = payload.get("models", {})
model_cfg = models.get(model_id)
if not isinstance(model_cfg, dict):
    msg = f"Model '{model_id}' not found in registry {registry}"
    raise SystemExit(msg)

for feature in model_cfg.get("features", []):
    print(feature)
PY
}

write_prepare_data_config() {
    local config_path="$1"
    local csv_path="$2"
    local csv_infer_path="$3"
    local stacked_inputs_filename="$4"
    local tiles_dirname="$5"
    local tile_stats_filename="$6"
    shift 6
    local requested_features=("$@")

    cat > "$config_path" <<EOF
data:
  class_path: geo_deep_learning.tools.water_extraction.elevation_stack_datamodule.ElevationStackDataModule
  init_args:
    csv_root_folder: "$PREP_ROOT"
    patches_root_folder: "$PREP_ROOT"
    input_folders:
      - "$RAW_AOI_DIR"
    output_root: "$PREP_ROOT"
    csv_path: "$csv_path"
    csv_infer_path: "$csv_infer_path"
    workflow: "$WORKFLOW"
    include_intensity: true
    stack_as_vrt: true
    extra_input_rasters:
      - "nadir_weighted.tif"
      - "zero_intensity.tif"
    stacked_inputs_filename: "$stacked_inputs_filename"
    tiles_dirname: "$tiles_dirname"
    tile_stats_filename: "$tile_stats_filename"
EOF

    if [[ ${#requested_features[@]} -gt 0 ]]; then
        {
            echo "    requested_features:"
            local feature
            for feature in "${requested_features[@]}"; do
                echo "      - \"$feature\""
            done
        } >> "$config_path"
    fi

    cat >> "$config_path" <<EOF
    stride: $STRIDE
    patch_size: [$PATCH_SIZE, $PATCH_SIZE]
    batch_size: $BATCH_SIZE
    num_workers: $NUM_WORKERS
    regenerate_csv: false
    min_water_pixels: $MIN_WATER_PIXELS
    test_ratio: $TEST_RATIO
    test_only: $TEST_ONLY
EOF
}

SOURCE_AOI_VECTOR="$(find_vector "$SOURCE_AOI_DIR" "$AOI_VECTOR_STEM" || true)"
if [[ -z "${SOURCE_AOI_VECTOR:-}" ]]; then
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
require_file "$MODEL_REGISTRY"
resolve_sitestore_base

REQUESTED_FEATURES=()
if [[ "$RUN_REGISTERED_INFERENCE" == "1" && "$WORKFLOW" == "inference" ]]; then
    mapfile -t REQUESTED_FEATURES < <(resolve_requested_features)
fi

setup_dynamic_slurm_log

copy_vector_family "$SOURCE_AOI_VECTOR" "$RAW_AOI_DIR"
if [[ -n "$WATERBODIES_VECTOR" ]]; then
    copy_vector_family "$WATERBODIES_VECTOR" "$RAW_AOI_DIR"
fi

RAW_AOI_VECTOR="$(find_vector "$RAW_AOI_DIR" "$AOI_VECTOR_STEM")"
RAW_BUFFERED_AOI="$RAW_AOI_DIR/$BUFFERED_AOI_STEM.gpkg"
DOWNLOAD_AOI_VECTOR="$RAW_BUFFERED_AOI"

echo "============================================"
echo "Job ID         : ${SLURM_JOB_ID:-N/A}"
echo "Node           : ${SLURM_NODELIST:-N/A}"
echo "Started        : $(date)"
echo "GDL Repo       : $GDL_REPO"
echo "Intensity Repo : $INTENSITY_REPO"
echo "AOI Source     : $SOURCE_AOI_DIR"
echo "AOI Name       : $AOI_NAME"
echo "Raw AOI        : $RAW_AOI_DIR"
echo "Prep Root      : $PREP_ROOT"
echo "Run Dir        : $RUN_DIR"
echo "Registry       : $MODEL_REGISTRY"
echo "Models         : ${MODEL_IDS:-<all registry models>}"
echo "Dynamic Log    : ${SLURM_DYNAMIC_LOG_PATH:-<default slurm log>}"
echo "============================================"

log "Step 1/9: Create buffered AOI before download"
activate_gdl
if [[ "$FORCE_RERUN" != "1" && -s "$RAW_BUFFERED_AOI" ]]; then
    log "Skipping buffered AOI creation: existing $RAW_BUFFERED_AOI found"
else
    rm -f "$RAW_AOI_DIR/$BUFFERED_AOI_STEM.gpkg" \
          "$RAW_AOI_DIR/$BUFFERED_AOI_STEM.gpkg-shm" \
          "$RAW_AOI_DIR/$BUFFERED_AOI_STEM.gpkg-wal"
    python -m geo_deep_learning.tools.water_extraction.create_buffered_aoi \
        --input_aoi "$RAW_AOI_VECTOR" \
        --output_aoi "$RAW_BUFFERED_AOI" \
        --buffer_meters "$BUFFER_METERS"
fi
require_file "$RAW_BUFFERED_AOI"

log "Step 2/9: Download DTM and DSM using buffered AOI"
if [[ "$FORCE_RERUN" != "1" && $(is_valid_geotiff "$RAW_AOI_DIR/dtm.tif"; echo $?) -eq 0 && $(is_valid_geotiff "$RAW_AOI_DIR/dsm.tif"; echo $?) -eq 0 ]]; then
    log "Skipping DTM/DSM download: existing rasters validated"
else
    mkdir -p "$ELEVATION_CACHE_DIR"

    python -m geo_deep_learning.tools.water_extraction.download_elevation \
        --aoi_path "$DOWNLOAD_AOI_VECTOR" \
        --out_folder "$ELEVATION_CACHE_DIR" \
        --resolution "$RESOLUTION"

    if ! is_valid_geotiff "$ELEVATION_CACHE_DIR/dtm.tif"; then
        echo "Downloaded DTM is missing or invalid: $ELEVATION_CACHE_DIR/dtm.tif" >&2
        exit 1
    fi
    if ! is_valid_geotiff "$ELEVATION_CACHE_DIR/dsm.tif"; then
        echo "Downloaded DSM is missing or invalid: $ELEVATION_CACHE_DIR/dsm.tif" >&2
        exit 1
    fi

    promote_file "$ELEVATION_CACHE_DIR/dtm.tif" "$RAW_AOI_DIR/dtm.tif"
    promote_file "$ELEVATION_CACHE_DIR/dsm.tif" "$RAW_AOI_DIR/dsm.tif"
fi

log "Step 3/9: Create valid_lidar_mask.gpkg"
if [[ "$FORCE_RERUN" != "1" && -s "$RAW_AOI_DIR/valid_lidar_mask.gpkg" ]]; then
    log "Skipping valid_lidar_mask creation: existing file found"
else
    rm -f "$RAW_AOI_DIR"/valid_lidar_mask.gpkg \
          "$RAW_AOI_DIR"/valid_lidar_mask.gpkg-shm \
          "$RAW_AOI_DIR"/valid_lidar_mask.gpkg-wal
    python -m geo_deep_learning.tools.water_extraction.create_valid_lidar_mask \
        --aoi_folder "$RAW_AOI_DIR" \
        --lidar_index "$PROJECT_INDEX"
fi

log "Step 4/9: Extract intersecting LAZ files"
activate_intensity
if [[ "$FORCE_RERUN" != "1" && -s "$EXTRACT_DIR/extraction_metadata.csv" ]]; then
    log "Skipping LAZ extraction: existing extraction metadata found"
else
    rm -rf "$EXTRACT_DIR"
    python extract_laz_by_aoi.py \
        "$DOWNLOAD_AOI_VECTOR" \
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

log "Step 4b/9: Detect and repair missing CRS metadata on extracted LAZ copies only"
python "$GDL_REPO/geo_deep_learning/tools/water_extraction/repair_extracted_laz_crs.py" \
    --metadata-csv "$EXTRACT_DIR/extraction_metadata.csv" \
    --project-index "$PROJECT_INDEX"

log "Step 5/9: Build and run intensity pipeline config"
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
    aoi: "$DOWNLOAD_AOI_VECTOR"
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

log "Step 6/9: Create zero-intensity and nadir-weighted rasters"
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

log "Step 7/9: Align extra rasters to DTM in gdl_env"
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

log "Step 8/9: Run water-extraction preprocessing"
TEST_ONLY=false
if [[ "$WORKFLOW" == "inference" ]]; then
    TEST_ONLY=true
fi

declare -a RESOLVED_MODEL_IDS=()
if [[ "$RUN_REGISTERED_INFERENCE" == "1" && "$WORKFLOW" == "inference" ]]; then
    mapfile -t RESOLVED_MODEL_IDS < <(resolve_model_ids)
    if [[ ${#RESOLVED_MODEL_IDS[@]} -eq 0 ]]; then
        echo "No model ids resolved from MODEL_IDS or $MODEL_REGISTRY" >&2
        exit 1
    fi

    for model_id in "${RESOLVED_MODEL_IDS[@]}"; do
        if [[ -z "$model_id" ]]; then
            continue
        fi
        mapfile -t MODEL_FEATURES < <(resolve_model_features "$model_id")
        if [[ ${#MODEL_FEATURES[@]} -eq 0 ]]; then
            echo "No features resolved for model '$model_id' from $MODEL_REGISTRY" >&2
            exit 1
        fi

        model_prep_config="$RUN_DIR/prepare_data_config__${model_id}.yaml"
        model_csv_infer="$PREP_ROOT/${AOI_NAME}_water_extraction_infer__${model_id}.csv"

        log "Preparing model-specific inputs for $model_id with features: ${MODEL_FEATURES[*]}"
        write_prepare_data_config \
            "$model_prep_config" \
            "$PREP_ROOT/${AOI_NAME}_water_extraction.csv" \
            "$model_csv_infer" \
            "stacked_inputs__${model_id}.tif" \
            "tiles__${model_id}" \
            "tile_stats__${model_id}.csv" \
            "${MODEL_FEATURES[@]}"

        python -m geo_deep_learning.tools.water_extraction.prepare_data \
            --config "$model_prep_config"
    done
else
    write_prepare_data_config \
        "$PREP_CONFIG" \
        "$PREP_ROOT/${AOI_NAME}_water_extraction.csv" \
        "$PREP_ROOT/${AOI_NAME}_water_extraction_infer.csv" \
        "stacked_inputs.tif" \
        "tiles" \
        "tile_stats.csv" \
        "${REQUESTED_FEATURES[@]}"

    python -m geo_deep_learning.tools.water_extraction.prepare_data \
        --config "$PREP_CONFIG"
fi

if [[ "$RUN_REGISTERED_INFERENCE" == "1" && "$WORKFLOW" == "inference" ]]; then
    log "Step 9/9: Run registry-driven inference"
    activate_gdl

    for model_id in "${RESOLVED_MODEL_IDS[@]}"; do
        if [[ -z "$model_id" ]]; then
            continue
        fi
        model_output_dir="$MODEL_RESULTS_ROOT/$model_id"
        model_stacked_inputs="$PREP_ROOT/$AOI_NAME/stacked_inputs__${model_id}.vrt"
        if [[ ! -f "$model_stacked_inputs" ]]; then
            fallback_stacked_inputs="$PREP_ROOT/$AOI_NAME/stacked_inputs__${model_id}.tif"
            if [[ -f "$fallback_stacked_inputs" ]]; then
                model_stacked_inputs="$fallback_stacked_inputs"
            fi
        fi
        require_file "$model_stacked_inputs"
        mkdir -p "$model_output_dir"
        log "Running inference for $model_id -> $model_output_dir using $model_stacked_inputs"

        python -m geo_deep_learning.tools.water_extraction.inference \
            --model_id "$model_id" \
            --registry "$MODEL_REGISTRY" \
            --stacked_inputs "$model_stacked_inputs" \
            --data_folder "$RAW_AOI_DIR" \
            --output_folder "$model_output_dir" \
            --window_size "$INFERENCE_WINDOW_SIZE" \
            --stride "$INFERENCE_STRIDE" \
            --batch_size "$INFERENCE_BATCH_SIZE" \
            --device "$INFERENCE_DEVICE"
    done
else
    log "Step 9/9: Skipping registry-driven inference"
fi

echo "============================================"
echo "Finished : $(date)"
echo "Buffered AOI          : $RAW_BUFFERED_AOI"
echo "Intensity raster      : $RAW_AOI_DIR/intensity.tif"
echo "Zero-intensity raster : $RAW_AOI_DIR/zero_intensity.tif"
echo "Nadir-weighted raster : $RAW_AOI_DIR/nadir_weighted.tif"
echo "Preprocessed outputs  : $PREP_ROOT/$AOI_NAME"
echo "Inference outputs     : $MODEL_RESULTS_ROOT"
echo "============================================"

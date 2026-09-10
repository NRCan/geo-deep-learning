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
ENABLE_AOI_PARTITIONING="${ENABLE_AOI_PARTITIONING:-1}"
MAX_BUFFERED_AOI_PIXELS="${MAX_BUFFERED_AOI_PIXELS:-0}"
SUBAOI_OVERLAP_METERS="${SUBAOI_OVERLAP_METERS:-$BUFFER_METERS}"
RUN_REGISTERED_INFERENCE="${RUN_REGISTERED_INFERENCE:-1}"
MODEL_IDS="${MODEL_IDS:-}"
MODEL_REGISTRY="${MODEL_REGISTRY:-$GDL_REPO/configs/inference_models.yaml}"
SKIP_DYNAMIC_SLURM_LOG="${SKIP_DYNAMIC_SLURM_LOG:-0}"

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
SIMPLIFY_TOLERANCE="${SIMPLIFY_TOLERANCE:-1.0}"

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
PARTITION_ROOT="$RUN_DIR/partitions"
PARTITION_MANIFEST="$PARTITION_ROOT/partition_manifest.json"
RUN_OUTPUTS_MANIFEST="$RUN_DIR/run_outputs_manifest.json"
PARTITION_SHARED_ELEVATION_DIR="$PARTITION_ROOT/_shared_elevation"

mkdir -p "$RAW_AOI_DIR" "$PREP_ROOT" "$RUN_DIR" "$NADIR_DIR" "$MODEL_RESULTS_ROOT"

export GDL_REPO INTENSITY_REPO OUTPUT_ROOT RESULTS_ROOT WORKFLOW AOI_VECTOR_STEM
export BUFFERED_AOI_STEM BUFFER_METERS ENABLE_AOI_PARTITIONING
export MAX_BUFFERED_AOI_PIXELS SUBAOI_OVERLAP_METERS RUN_REGISTERED_INFERENCE
export MODEL_IDS MODEL_REGISTRY GDL_CONDA_SH GDL_ENV INTENSITY_CONDA_SH
export INTENSITY_ENV REFERENCE_STANDARDS SITESTORE_BASE PROJECT_INDEX TILES_INDEX
export RESOLUTION TARGET_CRS NADIR_RESOLUTION NADIR_CHUNK_SIZE
export INTENSITY_PROCESSING_MODE INTENSITY_DARK_ZONE_MODE INTENSITY_OVERLAP_MODE
export PATCH_SIZE STRIDE BATCH_SIZE NUM_WORKERS TEST_RATIO MIN_WATER_PIXELS
export ZERO_MIN_COMPONENT_SIZE ZERO_MAX_HOLE_SIZE FORCE_RERUN INFERENCE_DEVICE
export INFERENCE_BATCH_SIZE INFERENCE_WINDOW_SIZE INFERENCE_STRIDE
export SIMPLIFY_TOLERANCE TMPDIR
export https_proxy http_proxy
export PYTHONUNBUFFERED SKIP_DYNAMIC_SLURM_LOG

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
    if [[ "$SKIP_DYNAMIC_SLURM_LOG" == "1" ]]; then
        return 0
    fi

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

should_partition_parent_aoi() {
    [[ "$WORKFLOW" == "inference" ]] || return 1
    [[ "$RUN_REGISTERED_INFERENCE" == "1" ]] || return 1
    [[ "$ENABLE_AOI_PARTITIONING" == "1" ]] || return 1
    [[ "$MAX_BUFFERED_AOI_PIXELS" =~ ^[0-9]+$ ]] || return 1
    [[ "$MAX_BUFFERED_AOI_PIXELS" -gt 0 ]] || return 1
    return 0
}

partition_manifest_complete() {
    python - "$PARTITION_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
if not manifest_path.exists():
    raise SystemExit(1)

with manifest_path.open(encoding="utf-8") as file:
    payload = json.load(file)

children = payload.get("children", [])
if payload.get("split_required"):
    if not children:
        raise SystemExit(1)
    for child in children:
        source_dir = child.get("source_dir")
        aoi_path = child.get("aoi_path")
        buffered_aoi_path = child.get("buffered_aoi_path")
        if not source_dir or not aoi_path or not buffered_aoi_path:
            raise SystemExit(1)
        if not Path(source_dir).is_dir():
            raise SystemExit(1)
        if not Path(aoi_path).is_file():
            raise SystemExit(1)
        if not Path(buffered_aoi_path).is_file():
            raise SystemExit(1)

raise SystemExit(0)
PY
}

partition_parent_aoi() {
    mkdir -p "$PARTITION_ROOT"
    if [[ "$FORCE_RERUN" != "1" ]] && partition_manifest_complete; then
        log "Reusing existing partition manifest and child AOIs from $PARTITION_MANIFEST"
        return 0
    fi
    python -m geo_deep_learning.tools.water_extraction.partition_aoi \
        --input_aoi "$RAW_AOI_VECTOR" \
        --buffered_aoi "$RAW_BUFFERED_AOI" \
        --output_root "$PARTITION_ROOT" \
        --resolution "$RESOLUTION" \
        --max_pixels "$MAX_BUFFERED_AOI_PIXELS" \
        --overlap_meters "$SUBAOI_OVERLAP_METERS" \
        --child_id_prefix "${AOI_NAME}__part"
}

partition_required() {
    python - "$PARTITION_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
if not manifest_path.exists():
    raise SystemExit(1)

with manifest_path.open(encoding="utf-8") as file:
    payload = json.load(file)

print("1" if payload.get("split_required") else "0")
PY
}

partition_child_dirs() {
    python - "$PARTITION_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

with Path(sys.argv[1]).open(encoding="utf-8") as file:
    payload = json.load(file)

for child in payload.get("children", []):
    source_dir = child.get("source_dir")
    if source_dir:
        print(source_dir)
PY
}

is_partition_child_aoi() {
    [[ "$(basename "$(dirname "$SOURCE_AOI_DIR")")" == "partitions" ]]
}

resolve_parent_run_dir_for_child() {
    if ! is_partition_child_aoi; then
        return 1
    fi
    realpath "$(dirname "$(dirname "$SOURCE_AOI_DIR")")"
}

crop_parent_elevation_to_child() {
    local parent_raster="$1"
    local child_aoi="$2"
    local output_raster="$3"

    python - "$parent_raster" "$child_aoi" "$output_raster" <<'PY'
from pathlib import Path
import sys

import fiona
import rasterio
import rasterio.mask

parent_raster = Path(sys.argv[1])
child_aoi = Path(sys.argv[2])
output_raster = Path(sys.argv[3])

with fiona.open(child_aoi, "r") as src:
    geoms = [feature["geometry"] for feature in src if feature["geometry"]]
    if not geoms:
        raise RuntimeError(f"No AOI geometries found in {child_aoi}")

with rasterio.open(parent_raster) as src:
    out_image, out_transform = rasterio.mask.mask(
        src,
        geoms,
        crop=True,
        filled=True,
        nodata=src.nodata,
    )
    profile = src.profile.copy()
    profile.update(
        height=out_image.shape[1],
        width=out_image.shape[2],
        transform=out_transform,
    )

output_raster.parent.mkdir(parents=True, exist_ok=True)
tmp_output = output_raster.with_name(output_raster.stem + "_temp" + output_raster.suffix)
with rasterio.open(tmp_output, "w", **profile) as dst:
    dst.write(out_image)
tmp_output.replace(output_raster)
PY
}

download_elevation_for_aoi() {
    local aoi_vector="$1"
    local output_dir="$2"

    mkdir -p "$output_dir"
    python -m geo_deep_learning.tools.water_extraction.download_elevation \
        --aoi_path "$aoi_vector" \
        --out_folder "$output_dir" \
        --resolution "$RESOLUTION"
}

ensure_shared_parent_elevation_cache() {
    local shared_dir="$1"

    mkdir -p "$shared_dir"
    if [[ "$FORCE_RERUN" != "1" && $(is_valid_geotiff "$shared_dir/dtm.tif"; echo $?) -eq 0 && $(is_valid_geotiff "$shared_dir/dsm.tif"; echo $?) -eq 0 ]]; then
        log "Shared parent elevation cache already valid in $shared_dir"
        return 0
    fi

    log "Downloading shared parent DTM/DSM once for partitioned child workflows"
    download_elevation_for_aoi "$DOWNLOAD_AOI_VECTOR" "$shared_dir"
    require_file "$shared_dir/dtm.tif"
    require_file "$shared_dir/dsm.tif"
    is_valid_geotiff "$shared_dir/dtm.tif"
    is_valid_geotiff "$shared_dir/dsm.tif"
}

promote_shared_parent_elevation_to_child() {
    local parent_run_dir shared_dir
    parent_run_dir="$(resolve_parent_run_dir_for_child)" || return 1
    shared_dir="$parent_run_dir/partitions/_shared_elevation"

    if [[ ! -f "$shared_dir/dtm.tif" || ! -f "$shared_dir/dsm.tif" ]]; then
        return 1
    fi

    if [[ "$FORCE_RERUN" != "1" && $(is_valid_geotiff "$RAW_AOI_DIR/dtm.tif"; echo $?) -eq 0 && $(is_valid_geotiff "$RAW_AOI_DIR/dsm.tif"; echo $?) -eq 0 ]]; then
        return 0
    fi

    log "Cropping shared parent elevation cache into child AOI outputs"
    crop_parent_elevation_to_child "$shared_dir/dtm.tif" "$DOWNLOAD_AOI_VECTOR" "$RAW_AOI_DIR/dtm.tif"
    crop_parent_elevation_to_child "$shared_dir/dsm.tif" "$DOWNLOAD_AOI_VECTOR" "$RAW_AOI_DIR/dsm.tif"
    require_file "$RAW_AOI_DIR/dtm.tif"
    require_file "$RAW_AOI_DIR/dsm.tif"
    is_valid_geotiff "$RAW_AOI_DIR/dtm.tif"
    is_valid_geotiff "$RAW_AOI_DIR/dsm.tif"
}

run_outputs_complete_for_root() {
    local manifest_path="$1"
    local model_results_root="$2"
    shift 2

    python - "$manifest_path" "$model_results_root" "$MODEL_REGISTRY" "$@" <<'PY'
import json
import sys
from pathlib import Path

import yaml


def latest_candidates(model_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    full_candidates = sorted(
        model_dir.glob("water_prediction_*_full_*.tif"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    cropped_candidates = sorted(
        [
            path
            for path in model_dir.glob("water_prediction_*.tif")
            if "_full_" not in path.name
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    vector_candidates = sorted(
        model_dir.glob("water_bodies_*.gpkg"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return (
        full_candidates[0] if full_candidates else None,
        cropped_candidates[0] if cropped_candidates else None,
        vector_candidates[0] if vector_candidates else None,
    )


manifest_path = Path(sys.argv[1])
model_results_root = Path(sys.argv[2])
registry_path = Path(sys.argv[3])
requested_model_ids = [arg for arg in sys.argv[4:] if arg]

with registry_path.open(encoding="utf-8") as file:
    payload = yaml.safe_load(file) or {}

models = payload.get("models", {})
model_ids = requested_model_ids or sorted(models)

if not model_ids:
    raise SystemExit(1)

if manifest_path.exists():
    try:
        with manifest_path.open(encoding="utf-8") as file:
            manifest_payload = json.load(file)
    except Exception:
        manifest_payload = {}
    else:
        manifest_models = manifest_payload.get("models", {})
        manifest_complete = True
        for model_id in model_ids:
            model_outputs = manifest_models.get(model_id)
            prediction_raster_full = None if model_outputs is None else model_outputs.get(
                "prediction_raster_full",
            )
            if not prediction_raster_full or not Path(prediction_raster_full).is_file():
                manifest_complete = False
                break
        if manifest_complete:
            raise SystemExit(0)

for model_id in model_ids:
    model_dir = model_results_root / model_id
    if not model_dir.exists():
        raise SystemExit(1)

    prediction_raster_full, prediction_raster, prediction_vector = latest_candidates(
        model_dir,
    )
    if (
        prediction_raster_full is None
        or prediction_raster is None
        or prediction_vector is None
    ):
        raise SystemExit(1)

    for path in (prediction_raster_full, prediction_raster, prediction_vector):
        if not path.is_file() or path.stat().st_size <= 0:
            raise SystemExit(1)

raise SystemExit(0)
PY
}

write_run_outputs_manifest_for_root() {
    local manifest_path="$1"
    local model_results_root="$2"
    shift 2

    python - "$manifest_path" "$model_results_root" "$MODEL_REGISTRY" "$@" <<'PY'
import json
import sys
from pathlib import Path

import yaml

manifest_path = Path(sys.argv[1])
model_results_root = Path(sys.argv[2])
registry_path = Path(sys.argv[3])
requested_model_ids = [arg for arg in sys.argv[4:] if arg]

with registry_path.open(encoding="utf-8") as file:
    payload = yaml.safe_load(file) or {}

models = payload.get("models", {})
model_ids = requested_model_ids or sorted(models)

outputs = {"models": {}}
for model_id in model_ids:
    model_dir = model_results_root / model_id
    if not model_dir.exists():
        continue

    full_candidates = sorted(
        model_dir.glob("water_prediction_*_full_*.tif"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    cropped_candidates = sorted(
        [
            path
            for path in model_dir.glob("water_prediction_*.tif")
            if "_full_" not in path.name
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    vector_candidates = sorted(
        model_dir.glob("water_bodies_*.gpkg"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not full_candidates:
        continue

    outputs["models"][model_id] = {
        "prediction_raster_full": str(full_candidates[0].resolve()),
        "prediction_raster": (
            str(cropped_candidates[0].resolve()) if cropped_candidates else None
        ),
        "prediction_vector": (
            str(vector_candidates[0].resolve()) if vector_candidates else None
        ),
    }

manifest_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
print(manifest_path)
PY
}

write_run_outputs_manifest() {
    write_run_outputs_manifest_for_root "$RUN_OUTPUTS_MANIFEST" "$MODEL_RESULTS_ROOT" ${MODEL_IDS:-}
}

ensure_run_outputs_manifest_for_root() {
    local manifest_path="$1"
    local model_results_root="$2"
    shift 2

    if ! run_outputs_complete_for_root "$manifest_path" "$model_results_root" "$@"; then
        return 1
    fi

    write_run_outputs_manifest_for_root "$manifest_path" "$model_results_root" "$@" >/dev/null
}

merge_partitioned_outputs() {
    local output_timestamp parent_aoi_vector merge_temp_root
    output_timestamp="$(date '+%Y%m%d_%H%M%S')"
    parent_aoi_vector="$RAW_AOI_VECTOR"
    merge_temp_root="$RUN_DIR/merge_core_predictions"
    mkdir -p "$merge_temp_root"

    mapfile -t RESOLVED_MODEL_IDS < <(resolve_model_ids)
    if [[ ${#RESOLVED_MODEL_IDS[@]} -eq 0 ]]; then
        echo "No model ids resolved from MODEL_IDS or $MODEL_REGISTRY" >&2
        exit 1
    fi

    mapfile -t CHILD_AOI_DIRS < <(partition_child_dirs)
    if [[ ${#CHILD_AOI_DIRS[@]} -eq 0 ]]; then
        echo "No child AOIs found in $PARTITION_MANIFEST" >&2
        exit 1
    fi

    activate_gdl

    for model_id in "${RESOLVED_MODEL_IDS[@]}"; do
        if [[ -z "$model_id" ]]; then
            continue
        fi

        local model_output_dir merged_full merged_cropped merged_vector output_stem
        model_output_dir="$MODEL_RESULTS_ROOT/$model_id"
        mkdir -p "$model_output_dir"

        output_stem="$(python - "$MODEL_REGISTRY" "$model_id" "$AOI_NAME" <<'PY'
import sys
from geo_deep_learning.tools.water_extraction.inference import _resolve_model_config

registry, model_id, fallback = sys.argv[1:4]
config = _resolve_model_config(model_id, registry)
print(config["output_name"] or fallback)
PY
)"

        mapfile -t CHILD_CORE_RASTER_SPECS < <(
            python - "$RUN_ROOT" "$PARTITION_MANIFEST" "$model_id" "${CHILD_AOI_DIRS[@]}" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
partition_manifest_path = Path(sys.argv[2])
model_id = sys.argv[3]
child_dirs = sys.argv[4:]

with partition_manifest_path.open(encoding="utf-8") as file:
    partition_payload = json.load(file)

children_by_name = {
    Path(child["source_dir"]).name: child
    for child in partition_payload.get("children", [])
    if child.get("source_dir") and child.get("aoi_path")
}

for child_dir in child_dirs:
    child_name = Path(child_dir).name
    child_meta = children_by_name.get(child_name)
    if not child_meta:
        raise SystemExit(
            f"Missing partition metadata for child {child_name}: {partition_manifest_path}"
        )

    manifest_path = run_root / child_name / "run_outputs_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"Missing child run outputs manifest for {child_name}: {manifest_path}"
        )

    with manifest_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    model_outputs = payload.get("models", {}).get(model_id)
    if not model_outputs:
        raise SystemExit(
            f"Missing model '{model_id}' outputs in child manifest: {manifest_path}"
        )

    prediction_raster_full = model_outputs.get("prediction_raster_full")
    if not prediction_raster_full:
        raise SystemExit(
            f"Missing full prediction raster for model '{model_id}' in {manifest_path}"
        )
    aoi_path = child_meta.get("aoi_path")
    if not aoi_path:
        raise SystemExit(
            f"Missing core AOI path for child {child_name} in {partition_manifest_path}"
        )
    print(f"{prediction_raster_full}\t{aoi_path}\t{child_name}")
PY
        )

        if [[ ${#CHILD_CORE_RASTER_SPECS[@]} -ne ${#CHILD_AOI_DIRS[@]} ]]; then
            echo "Expected ${#CHILD_AOI_DIRS[@]} child prediction/core AOI specs for model $model_id, found ${#CHILD_CORE_RASTER_SPECS[@]}" >&2
            exit 1
        fi

        merged_full="$model_output_dir/water_prediction_${output_stem}_full_${output_timestamp}.tif"
        merged_cropped="$model_output_dir/water_prediction_${output_stem}_${output_timestamp}.tif"
        merged_vector="$model_output_dir/water_bodies_${output_stem}_${output_timestamp}.gpkg"
        model_merge_temp_dir="$merge_temp_root/$model_id"
        mkdir -p "$model_merge_temp_dir"
        CHILD_CORE_RASTERS=()
        for child_spec in "${CHILD_CORE_RASTER_SPECS[@]}"; do
            IFS=$'\t' read -r child_full_raster child_core_aoi child_name <<<"$child_spec"
            child_core_raster="$model_merge_temp_dir/${child_name}_core.tif"
            python - "$child_full_raster" "$child_core_raster" "$child_core_aoi" <<'PY'
import sys

from geo_deep_learning.tools.water_extraction.inference import crop_raster_to_aoi

crop_raster_to_aoi(sys.argv[1], sys.argv[2], sys.argv[3])
PY
            CHILD_CORE_RASTERS+=("$child_core_raster")
        done

        if [[ ${#CHILD_CORE_RASTERS[@]} -ne ${#CHILD_AOI_DIRS[@]} ]]; then
            echo "Expected ${#CHILD_AOI_DIRS[@]} cropped child core rasters for model $model_id, found ${#CHILD_CORE_RASTERS[@]}" >&2
            exit 1
        fi

        python - "$merged_full" "$merged_cropped" "$merged_vector" "$parent_aoi_vector" "$SIMPLIFY_TOLERANCE" "${CHILD_CORE_RASTERS[@]}" <<'PY'
import sys

from geo_deep_learning.tools.water_extraction.inference import (
    crop_raster_to_aoi,
    export_vectors,
    merge_prediction_rasters,
)

merged_full, merged_cropped, merged_vector, aoi_path, simplify = sys.argv[1:6]
child_rasters = sys.argv[6:]

merge_prediction_rasters(child_rasters, merged_full, method="max")
crop_raster_to_aoi(merged_full, merged_cropped, aoi_path)
export_vectors(
    prediction_raster_path=merged_cropped,
    output_vector_path=merged_vector,
    simplify_tolerance=float(simplify),
)
PY

        log "Merged ${#CHILD_CORE_RASTERS[@]} child core predictions for $model_id -> $model_output_dir"
    done
}

SOURCE_AOI_VECTOR="$(find_vector "$SOURCE_AOI_DIR" "$AOI_VECTOR_STEM" || true)"
if [[ -z "${SOURCE_AOI_VECTOR:-}" ]]; then
    echo "Missing AOI vector in $SOURCE_AOI_DIR (expected $AOI_VECTOR_STEM.gpkg or $AOI_VECTOR_STEM.shp)" >&2
    exit 1
fi
SOURCE_BUFFERED_AOI_VECTOR="$(find_vector "$SOURCE_AOI_DIR" "$BUFFERED_AOI_STEM" || true)"

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
if [[ -n "$SOURCE_BUFFERED_AOI_VECTOR" ]]; then
    copy_vector_family "$SOURCE_BUFFERED_AOI_VECTOR" "$RAW_AOI_DIR"
fi
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

if should_partition_parent_aoi; then
    log "Evaluating parent AOI partitioning with max pixel budget $MAX_BUFFERED_AOI_PIXELS"
    partition_parent_aoi
    require_file "$PARTITION_MANIFEST"

    if [[ "$(partition_required)" == "1" ]]; then
        log "Parent AOI exceeds pixel budget; running partitioned child workflows"
        ensure_shared_parent_elevation_cache "$PARTITION_SHARED_ELEVATION_DIR"
        mapfile -t CHILD_AOI_DIRS < <(partition_child_dirs)
        if [[ ${#CHILD_AOI_DIRS[@]} -eq 0 ]]; then
            echo "Partitioning requested but no child AOIs were created in $PARTITION_ROOT" >&2
            exit 1
        fi

        for child_dir in "${CHILD_AOI_DIRS[@]}"; do
            child_name="$(basename "$child_dir")"
            child_run_outputs_manifest="$RUN_ROOT/$child_name/run_outputs_manifest.json"
            child_model_results_root="$RESULTS_ROOT/$child_name"

            if [[ "$FORCE_RERUN" != "1" ]] && ensure_run_outputs_manifest_for_root "$child_run_outputs_manifest" "$child_model_results_root" ${MODEL_IDS:-}; then
                log "Skipping child AOI workflow for $child_name: existing inference outputs validated"
                continue
            fi

            log "Launching child AOI workflow for $child_name"
            ENABLE_AOI_PARTITIONING=0 \
            SKIP_DYNAMIC_SLURM_LOG=1 \
            bash "$0" "$child_dir"
        done

        log "Merging child inference outputs back to parent AOI"
        merge_partitioned_outputs

        echo "============================================"
        echo "Finished : $(date)"
        echo "Buffered AOI          : $RAW_BUFFERED_AOI"
        echo "Partition manifest    : $PARTITION_MANIFEST"
        echo "Inference outputs     : $MODEL_RESULTS_ROOT"
        echo "============================================"
        exit 0
    fi

    log "Buffered parent AOI is within the configured pixel budget; continuing single-AOI workflow"
fi

log "Step 2/9: Download DTM and DSM using buffered AOI"
if [[ "$FORCE_RERUN" != "1" && $(is_valid_geotiff "$RAW_AOI_DIR/dtm.tif"; echo $?) -eq 0 && $(is_valid_geotiff "$RAW_AOI_DIR/dsm.tif"; echo $?) -eq 0 ]]; then
    log "Skipping DTM/DSM download: existing rasters validated"
elif promote_shared_parent_elevation_to_child; then
    log "Skipping STAC download: reused shared parent elevation cache"
else
    download_elevation_for_aoi "$DOWNLOAD_AOI_VECTOR" "$ELEVATION_CACHE_DIR"

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

    if [[ "$FORCE_RERUN" != "1" ]] && ensure_run_outputs_manifest_for_root "$RUN_OUTPUTS_MANIFEST" "$MODEL_RESULTS_ROOT" ${MODEL_IDS:-}; then
        log "Skipping registry-driven inference: existing outputs validated"
    else
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
        write_run_outputs_manifest >/dev/null
    fi
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

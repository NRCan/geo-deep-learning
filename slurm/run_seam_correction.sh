#!/bin/bash
#SBATCH --job-name=seam_correction
#SBATCH --partition=standard
#SBATCH --account=nrcan_geobase
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%j_seam_correction.out
#SBATCH --error=slurm/logs/%j_seam_correction.out
#SBATCH --qos=low

set -euo pipefail

usage() {
    cat <<'EOF'
Apply boundary-based seam smoothing to one raster.

Usage:
  sbatch slurm/run_seam_correction.sh INPUT_RASTER BOUNDARIES OUTPUT_RASTER [options]
  bash   slurm/run_seam_correction.sh INPUT_RASTER BOUNDARIES OUTPUT_RASTER [options]

Required positional arguments:
  INPUT_RASTER   Input single- or multi-band GeoTIFF.
  BOUNDARIES     Polygon vector defining LiDAR project boundaries (for example GPKG).
  OUTPUT_RASTER  Destination GeoTIFF. It will be overwritten if it exists.

Options:
  --buffer-pixels N       Half-width of the blend taper (default: 3).
  --seam-width-pixels N   Half-width of the inpainting zone (default: 1).
  --nodata-fill-pixels N  NoData fill distance from a boundary (default: 5).
  --sigma VALUE           Gaussian inpainting sigma (default: 1.5).
  --blend-sigma VALUE     Taper-zone smoothing sigma (default: 0.0).
  --chunk-rows N          Core rows processed per chunk (default: 512).
  -h, --help              Show this help message.

Environment overrides:
  GDL_CONDA_SH  Conda initialization script.
  GDL_ENV       Conda environment name or path (default: gdl_env).

Example:
  sbatch slurm/run_seam_correction.sh \
    data/dtm.tif data/lidar_project_boundaries.gpkg results/dtm_smoothed.tif \
    --sigma 2.0 --seam-width-pixels 1
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 3 ]]; then
    usage >&2
    exit 2
fi

INPUT_RASTER="$1"
BOUNDARIES="$2"
OUTPUT_RASTER="$3"
shift 3

BUFFER_PIXELS=3
SEAM_WIDTH_PIXELS=1
NODATA_FILL_PIXELS=5
SIGMA=1.5
BLEND_SIGMA=0.0
CHUNK_ROWS=512

while [[ $# -gt 0 ]]; do
    case "$1" in
        --buffer-pixels)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            BUFFER_PIXELS="$2"
            shift 2
            ;;
        --seam-width-pixels)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            SEAM_WIDTH_PIXELS="$2"
            shift 2
            ;;
        --nodata-fill-pixels)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            NODATA_FILL_PIXELS="$2"
            shift 2
            ;;
        --sigma)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            SIGMA="$2"
            shift 2
            ;;
        --blend-sigma)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            BLEND_SIGMA="$2"
            shift 2
            ;;
        --chunk-rows)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            CHUNK_ROWS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

# SLURM jobs may start outside the submit directory. Interpret relative user paths
# from the directory where sbatch was submitted (or the current directory for bash).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
to_absolute_path() {
    if [[ "$1" == /* ]]; then
        printf '%s\n' "$1"
    else
        printf '%s/%s\n' "$SUBMIT_DIR" "$1"
    fi
}

INPUT_RASTER="$(to_absolute_path "$INPUT_RASTER")"
BOUNDARIES="$(to_absolute_path "$BOUNDARIES")"
OUTPUT_RASTER="$(to_absolute_path "$OUTPUT_RASTER")"

if [[ ! -f "$INPUT_RASTER" ]]; then
    echo "Input raster not found: $INPUT_RASTER" >&2
    exit 1
fi
if [[ ! -f "$BOUNDARIES" ]]; then
    echo "Boundary vector not found: $BOUNDARIES" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GDL_CONDA_SH="${GDL_CONDA_SH:-/space/partner/nrcan/geobase/work/opt/miniconda-gdl-ops/etc/profile.d/conda.sh}"
GDL_ENV="${GDL_ENV:-gdl_env}"

if [[ ! -f "$GDL_CONDA_SH" ]]; then
    echo "Conda initialization script not found: $GDL_CONDA_SH" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_RASTER")"

set +u
# shellcheck disable=SC1090
source "$GDL_CONDA_SH"
conda activate "$GDL_ENV"
set -u

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "Input raster : $INPUT_RASTER"
echo "Boundaries   : $BOUNDARIES"
echo "Output raster: $OUTPUT_RASTER"
echo "Parameters   : buffer=$BUFFER_PIXELS seam_width=$SEAM_WIDTH_PIXELS nodata_fill=$NODATA_FILL_PIXELS sigma=$SIGMA blend_sigma=$BLEND_SIGMA chunk_rows=$CHUNK_ROWS"

python -m geo_deep_learning.tools.water_extraction.seam_correction \
    --input "$INPUT_RASTER" \
    --project_extents "$BOUNDARIES" \
    --output "$OUTPUT_RASTER" \
    --buffer_pixels "$BUFFER_PIXELS" \
    --seam_width_pixels "$SEAM_WIDTH_PIXELS" \
    --nodata_fill_pixels "$NODATA_FILL_PIXELS" \
    --sigma "$SIGMA" \
    --blend_sigma "$BLEND_SIGMA" \
    --chunk_rows "$CHUNK_ROWS"

echo "Seam correction complete: $OUTPUT_RASTER"

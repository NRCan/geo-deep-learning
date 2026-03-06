#!/bin/bash
#SBATCH --job-name=water_prep_data
#SBATCH --partition=standard
#SBATCH --account=nrcan_geobase
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%j_prepare_data.out
#SBATCH --error=slurm/logs/%j_prepare_data.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=luca.romanini@nrcan-rncan.gc.ca
#SBATCH --qos=low

# ══════════════════════════════════════════════════════════════
# Water Extraction Data Preparation - SLURM Job
# ══════════════════════════════════════════════════════════════
# This script prepares data for water extraction training:
# - Aligns DTM, DSM, and intensity rasters
# - Applies seam correction (if project_extents_path provided)
# - Computes TWI (Topographic Wetness Index)
# - Computes nDSM (normalized Digital Surface Model)
# - Stacks input bands
# - Tiles the data for training
# - Generates train/val/test CSV splits
# - Computes normalization statistics
#
# No GPU required - CPU-intensive preprocessing only
# ══════════════════════════════════════════════════════════════

# ── Environment ───────────────────────────────────────────────
export https_proxy=http://webproxy.science.gc.ca:8888/
export http_proxy=http://webproxy.science.gc.ca:8888/
export PYTHONUNBUFFERED=1
export TMPDIR=/gpfs/fs5/nrcan/nrcan_geobase/gdl_tmp

source /space/partner/nrcan/geobase/work/opt/miniconda-gdl-ops/etc/profile.d/conda.sh
conda activate gdl_env

# ── Paths ─────────────────────────────────────────────────────
REPO=/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/gdl_projects/geo-deep-learning
cd $REPO
export PYTHONPATH=$REPO

# ── Log job info ──────────────────────────────────────────────
echo "============================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "CPUs       : $SLURM_CPUS_PER_TASK"
echo "Memory     : ${SLURM_MEM_PER_NODE}MB"
echo "Started    : $(date)"
echo "Repo       : $REPO"
echo "============================================"

# ── Data Preparation ──────────────────────────────────────────
echo ""
echo "Starting data preparation..."
echo ""

# ══════════════════════════════════════════════════════════════
# OPTION 1: Use config file as-is (no overrides)
# ══════════════════════════════════════════════════════════════
python -m geo_deep_learning.tools.water_extraction.prepare_data \
    --config configs/02NB000.yaml

# ══════════════════════════════════════════════════════════════
# OPTION 2: Use config file with CLI overrides
# ══════════════════════════════════════════════════════════════
# NOTE: CLI overrides with OmegaConf can be tricky. If they don't work,
# set values directly in the config file instead (Option 1 above).

# python -m geo_deep_learning.tools.water_extraction.prepare_data \
#     --config configs/02NB000.yaml \
#     --data.init_args.project_extents_path=/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/gdl_projects/geo-deep-learning/data/02NB000/02NB000_lidar_projects_boundaries.gpkg \
#     --data.init_args.seam_gaussian_sigma=1.5 \
#     --data.init_args.include_intensity=true \
#     --data.init_args.regenerate_csv=false \
#     --data.init_args.stride=256 \
#     --data.init_args.patch_size='[512,512]' \
#     --data.init_args.test_ratio=0.2 \
#     --data.init_args.min_water_pixels=1 \
#     --data.init_args.valid_mask_min_ratio=0.9 \
#     --data.init_args.save_rejected_tiles=false \
#     --data.init_args.test_only=false \
#     --data.init_args.batch_size=16 \
#     --data.init_args.num_workers=32

echo ""
echo "============================================"
echo "Finished : $(date)"
echo "============================================"

#!/bin/bash
#SBATCH --job-name=chn_create_mask
#SBATCH --partition=standard
#SBATCH --account=nrcan_geobase
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%j_create_mask.out
#SBATCH --error=slurm/logs/%j_create_mask.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=luca.romanini@nrcan-rncan.gc.ca

# ══════════════════════════════════════════════════════════════
# Create Valid LiDAR Mask - SLURM Job
# ══════════════════════════════════════════════════════════════
# This script creates valid_lidar_mask.gpkg by:
# 1. Finding LiDAR projects that intersect with AOI
# 2. Dissolving them into one polygon
# 3. Clipping to AOI boundary
# 4. Saving as valid_lidar_mask.gpkg
#
# Fast CPU-only task - no GPU needed
# ══════════════════════════════════════════════════════════════

# ── Environment ───────────────────────────────────────────────
export https_proxy=http://webproxy.science.gc.ca:8888/
export http_proxy=http://webproxy.science.gc.ca:8888/
export PYTHONUNBUFFERED=1

source /space/partner/nrcan/geobase/work/opt/miniconda-gdl-ops/etc/profile.d/conda.sh
conda activate gdl_env

# ── Paths ─────────────────────────────────────────────────────
REPO=/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/gdl_projects/geo-deep-learning
cd $REPO
export PYTHONPATH=$REPO

# LiDAR project index (default path)
LIDAR_INDEX=/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/lidar/utils/input_data/index_lidar/projet_lidar_infos_detaillees_2.gpkg

# ── Log job info ──────────────────────────────────────────────
echo "============================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "CPUs       : $SLURM_CPUS_PER_TASK"
echo "Memory     : ${SLURM_MEM_PER_NODE}MB"
echo "Started    : $(date)"
echo "Repo       : $REPO"
echo "============================================"

# ── Create Valid LiDAR Mask ───────────────────────────────────
echo ""
echo "Creating valid_lidar_mask.gpkg..."
echo ""

# OPTION 1: Single AOI
python -m geo_deep_learning.tools.water_extraction.create_valid_lidar_mask \
    --aoi_folder data/02NF000 \
    --lidar_index "$LIDAR_INDEX" \
    --save_intermediate

# OPTION 2: Multiple AOIs (uncomment and edit as needed)
# for AOI in data/02NB000 data/02NC000 data/gaspesie_01BG001; do
#     echo "Processing: $AOI"
#     python -m geo_deep_learning.tools.water_extraction.create_valid_lidar_mask \
#         --aoi_folder "$AOI" \
#         --lidar_index "$LIDAR_INDEX"
# done

echo ""
echo "============================================"
echo "Finished : $(date)"
echo "============================================"

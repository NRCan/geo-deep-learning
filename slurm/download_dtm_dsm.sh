#!/bin/bash
#SBATCH --job-name=chn_dtm_dsm_extract
#SBATCH --partition=standard
#SBATCH --account=nrcan_geobase
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%j_sonata_scan.out
#SBATCH --error=slurm/logs/%j_sonata_scan.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=luca.romanini@nrcan-rncan.gc.ca
#SBATCH --qos=low

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

# ── Log job info ───────────────────────────────────────────────
echo "============================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "Started    : $(date)"
echo "Repo       : $REPO"
echo "============================================"

# ── Class inventory scan ──────────────────────────────────────
python -m geo_deep_learning.tools.water_extraction.download_elevation --aoi_path /gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/gdl_projects/geo-deep-learning/data/02NB000/aoi.gpkg

echo "============================================"
echo "Finished : $(date)"
echo "============================================"

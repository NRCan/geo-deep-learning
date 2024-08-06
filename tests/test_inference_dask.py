from dask.distributed import Client as daskclient
from dask.distributed import LocalCluster
from pathlib import Path
import sys
import numpy as np
import dask.array as da
import time
import torch
import GPUtil
import tracemalloc
import psutil

if str(Path(__file__).parents[0]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[0]))
from utils.logger import get_logger
from utils.aoiutils import aois_from_csv

logging = get_logger(__name__)  # import logging

# git push origin HEAD:develop
def inference_with_dask():
    # GET LIST OF INPUT IMAGES FOR INFERENCE
    list_aois = aois_from_csv(
        csv_path="/gpfs/fs5/nrcan/nrcan_geobase/work/dev/datacube/parallel/deep_learning_repo/geo-deep-learning/tests/data/inference/test.csv",
        bands_requested=["red", "green", "blue"],
        data_dir="/gpfs/fs5/nrcan/nrcan_geobase/work/dev/datacube/parallel/geo_inference_big_enhancement/",
        write_dest_raster=True,
        equalize_clahe_clip_limit=25,
    )

def print_gpu_usage():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}:")
        print(f"  Load: {gpu.load * 100:.2f}%")
        print(f"  Memory Free: {gpu.memoryFree:.2f}MB")
        print(f"  Memory Used: {gpu.memoryUsed:.2f}MB")
        print(f"  Memory Total: {gpu.memoryTotal:.2f}MB")
        print(f"  Temperature: {gpu.temperature:.2f} C")


if __name__ == "__main__":
    inference_with_dask()

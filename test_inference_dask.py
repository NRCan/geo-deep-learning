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
import asyncio
import psutil
import gc
if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.logger import get_logger
from utils.aoiutils import aois_from_csv
logging = get_logger(__name__)  # import logging

def inference_with_dask():
    # GET LIST OF INPUT IMAGES FOR INFERENCE
    list_aois = aois_from_csv(
        csv_path="/gpfs/fs5/nrcan/nrcan_geobase/work/dev/datacube/parallel/deep_learning_repo/geo-deep-learning/tests/inference/inference_segmentation_binary.csv",
        bands_requested=['red', 'green','blue'],
        data_dir="/gpfs/fs5/nrcan/nrcan_geobase/work/dev/datacube/parallel/dask_geo_deep_learning/dask_geo_inference/performance_track/",
        write_dest_raster=True,
        write_dest_zarr = False,
        equalize_clahe_clip_limit=25,
    )


if __name__ == "__main__":
    inference_with_dask()

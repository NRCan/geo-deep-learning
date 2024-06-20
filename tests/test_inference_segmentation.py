import os
from omegaconf import DictConfig
from dask.distributed import Client as daskclient
from dask.distributed import LocalCluster
from pathlib import Path
import sys
import numpy as np
import dask.array as da
import time
import torch
from torch.nn import functional as F

if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))
from inference_segmentation import main as run_inference
from dataset.aoi import get_tiff_paths_from_csv, single_aoi
import gc


# Function to trim memory
def trim_memory():
    gc.collect()


def test_inference_segmentation():
    """Test inference segmentation"""
    model_path = "/gpfs/fs5/nrcan/nrcan_geobase/work/dev/datacube/parallel/deep_learning/geo-deep-learning/rgb-4class-segformer-b5.pt"
    raw_data_csv = "tests/data/inference/test.csv"
    data_dir = "tests/data/inference"
    # bands_requested = ["R", "G", "B"]
    # bands_requested = ["nir", "green", "red"]
    bands_requested = [1, 2, 3]
    cluster = LocalCluster(n_workers=15, memory_limit="50GB")
    data_tiff_paths = get_tiff_paths_from_csv(csv_path=raw_data_csv)
    for aoi_dict in data_tiff_paths:
        aoi = single_aoi(
            aoi_dict=aoi_dict,
            bands_requested=bands_requested,
            data_dir=data_dir,
            equalize_clahe_clip_limit=0.25,
            raster_stats=True,
            chunk_size=1024,
        )
        with daskclient(cluster, timeout="60s") as client:
            try:
                print(f"The dashboard link for dask cluster is {client.dashboard_link}")
                start_time = time.time()
                # create a dask array on the cluster
                aoi_dask_array = aoi.create_dask_array()
                print(aoi_dask_array)
                # check the contrast and if the contrast is low, apply the enhancement method on it
                if aoi.high_or_low_contrast and aoi.enhance_clip_limit > 0:
                    aoi_dask_array = da.map_overlap(
                        aoi.equalize_adapthist_enhancement,
                        aoi_dask_array,
                        depth={
                            1: int(aoi.chunk_size / 8),
                            2: int(aoi.chunk_size / 8),
                        },  # we add some overlap to the edges, the default kernal size is 1/8 of image size
                        trim=True,
                        clip_limit=aoi.enhance_clip_limit,
                        dtype=np.int32,
                    )
                # now we rechunk data so that each chunk has 3 bands
                aoi_dask_array = aoi_dask_array.rechunk(
                    (aoi.num_bands, int(aoi.chunk_size / 2), int(aoi.chunk_size / 2))
                )
                print(aoi_dask_array)
                # When the data is ready we create chunks of data which have 50% overlap in x and y with their left and bottom neighbors

                aoi_dask_array = da.map_overlap(
                    aoi.add_overlp_to_chunks,
                    aoi_dask_array,
                    depth={1: int(aoi.chunk_size / 2), 2: int(aoi.chunk_size / 2)},
                    trim=False,
                    chunks=(
                        aoi.num_bands,
                        aoi.chunk_size,
                        aoi.chunk_size,
                    ),
                    chunk_size=aoi.chunk_size,
                    dtype=aoi_dask_array.dtype,
                )
                print(aoi_dask_array)
                # now we run the model on each chunk
                raster_model = da.map_blocks(
                    aoi.runModel,
                    aoi_dask_array,
                    chunks=(
                        5,
                        aoi.chunk_size,
                        aoi.chunk_size,
                    ),
                    chunk_size=aoi.chunk_size,
                    model_path=model_path,
                    dtype=aoi_dask_array.dtype,
                )
                raster_torch = da.map_blocks(
                    aoi.apply_window_on_chunks,
                    raster_model,
                    chunks=(
                        5,
                        aoi.chunk_size,
                        aoi.chunk_size,
                    ),
                    chunk_size=aoi.chunk_size,
                    dtype=aoi_dask_array.dtype,
                )
                print(raster_torch)
                raster_model_output = da.map_overlap(
                    aoi.sum_overlapped_chunks,
                    raster_torch,
                    depth={1: aoi.chunk_size, 2: aoi.chunk_size},
                    trim=False,
                    chunks=(
                        5,
                        int(aoi.chunk_size / 2),
                        int(aoi.chunk_size / 2),
                    ),
                    chunk_size=aoi.chunk_size,
                    dtype=aoi_dask_array.dtype,
                )

                print(raster_model_output)
                aoi_dask_array = client.gather(raster_model_output)
                print(aoi_dask_array)
                final_array = aoi_dask_array.compute()

                aoi.write_inference_to_tif(final_array)
                end_time = time.time()
                print(f"\ntotal time is {end_time -start_time}")

            except Exception as e:
                print(f"Failed to read and stack data: {e}")
                client.close()
                raise e


if __name__ == "__main__":
    test_inference_segmentation()

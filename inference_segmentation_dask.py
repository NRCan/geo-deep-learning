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
from dataset.aoi import get_tiff_paths_from_csv, single_aoi


logging = get_logger(__name__)  # import logging


def inference_with_dask():
    """Test inference segmentation"""

    model_path = "/gpfs/fs5/nrcan/nrcan_geobase/work/dev/datacube/parallel/deep_learning_model/4cls_RGB_5_1_2_3_scripted.pt"
    raw_data_csv = "tests/data/inference/test.csv"
    # bands_requested = ["R", "G", "B"]
    bands_requested = [1, 2, 3]

    """ Processing starts"""
    cluster = LocalCluster(n_workers=15, memory_limit="50GB")
    tracemalloc.start()  # For memory tracking
    process = psutil.Process()  # For cpu tracking

    start_time = time.time()
    data_tiff_paths = get_tiff_paths_from_csv(csv_path=raw_data_csv)
    for aoi_dict in data_tiff_paths:
        aoi = single_aoi(
            aoi_dict=aoi_dict,
            bands_requested=bands_requested,
            equalize_clahe_clip_limit=0.0,
            raster_stats=False,
            chunk_size=512,
        )
        with daskclient(cluster, timeout="60s") as client:
            try:
                logging.info(
                    f"The dashboard link for dask cluster is at "
                    f"{client.dashboard_link}"
                )
                print(client.dashboard_link)
                aoi_dask_array = (
                    aoi.create_dask_array()
                )  # create a dask array on the dask cluster

                # check the contrast and if the contrast is low, apply the enhancement method on it
                if aoi.high_or_low_contrast and aoi.enhance_clip_limit > 0:
                    aoi_dask_array = da.map_overlap(
                        aoi.equalize_adapthist_enhancement,
                        aoi_dask_array,
                        clip_limit=aoi.enhance_clip_limit,
                        depth={
                            1: int(aoi.chunk_size / 8),
                            2: int(aoi.chunk_size / 8),
                        },  # we add some overlap to the edges, the default kernal size is 1/8 of image size
                        trim=True,
                        dtype=np.int32,
                    )

                pad_height = (
                    int(aoi.chunk_size / 2)
                    - aoi_dask_array.shape[1] % int(aoi.chunk_size / 2)
                ) % int(aoi.chunk_size / 2)
                pad_width = (
                    int(aoi.chunk_size / 2)
                    - aoi_dask_array.shape[2] % int(aoi.chunk_size / 2)
                ) % int(aoi.chunk_size / 2)
                # Pad the array to make dimensions multiples of the chunk size
                aoi_dask_array = da.pad(
                    aoi_dask_array,
                    ((0, 0), (0, pad_height), (0, pad_width)),
                    mode="constant",
                ).rechunk(
                    (aoi.num_bands, int(aoi.chunk_size / 2), int(aoi.chunk_size / 2))
                )  # now we rechunk data so that each chunk has 3 bands
                logging.info(
                    f" The dask array to be fed to Inference model is \n"
                    f"{aoi_dask_array}"
                )
                # Run the model and gather results
                raster_model = da.map_overlap(
                    aoi.runModel,
                    aoi_dask_array,
                    chunk_size=aoi.chunk_size,
                    model_path=model_path,
                    chunks=(
                        6,
                        aoi.chunk_size,
                        aoi.chunk_size,
                    ),
                    depth={1: int(aoi.chunk_size / 2), 2: int(aoi.chunk_size / 2)},
                    trim=False,
                    boundary="none",
                    dtype=np.float16,
                )
                logging.info(
                    f" The dask array to be fed to sum_overlapped_chunks is \n"
                    f"{raster_model}"
                )
                raster_model_output = da.map_overlap(
                    aoi.sum_overlapped_chunks,
                    raster_model,
                    drop_axis=0,
                    chunk_size=aoi.chunk_size,
                    chunks=(
                        int(aoi.chunk_size / 2),
                        int(aoi.chunk_size / 2),
                    ),
                    depth={1: int(aoi.chunk_size / 2), 2: int(aoi.chunk_size / 2)},
                    allow_rechunk=True,
                    trim=False,
                    boundary="none",
                    dtype=np.uint8,
                )
                logging.info(
                    f"The Inference output dask arrray is \n" f"{raster_model_output}"
                )
                aoi_dask_array = client.gather(raster_model_output)
                final_array = aoi_dask_array.compute()
                print(final_array)
                print(final_array.shape)
                aoi.write_inference_to_tif(final_array)

                # Print Performance results
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                hours, remainder = divmod(total_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                logging.info(
                    f"The total time of running Inference is {int(hours)}:{int(minutes)}:{seconds:.2f} \n"
                )

                logging.info(
                    f"Memory usage for the Inference is: {process.memory_info().rss / 1024 ** 2} MB"
                )
                logging.info(
                    f"CPU usage for the Inference is: {psutil.cpu_percent(interval=1)}%"
                )
                logging.info(
                    f"Allocated memory on GPU for the Inference is: {torch.cuda.memory_allocated() / 1024 ** 2} MB"
                )
                logging.info(
                    f"Cached memory on GPU for the Inference is: {torch.cuda.memory_reserved() / 1024 ** 2} MB"
                )
                logging.info(f"GPU usage during the process:\n" f"{print_gpu_usage()}")

                # memory track
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics("lineno")
                """
                for stat in top_stats[:10]:
                    print(stat)
                """
            except Exception as e:
                print(f"Processing on the Dask cluster failed due to: {e}")
                client.close()
                raise e


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

import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from tqdm import tqdm
import scipy.signal.windows as w
import numpy as np
import dask
import rioxarray
import xarray as xr
import functools
import json
import dask.array as da
from scipy.special import expit
import torch
import sys
import dask
from hydra.utils import to_absolute_path
from pandas.io.common import is_url
from omegaconf import listconfig, ListConfig
from tqdm import tqdm
from rasterio.windows import from_bounds
import scipy.signal.windows as w
import pims
import traceback

from dask_image.imread import imread as dask_imread  # type: ignore
from dask_image.imread import _map_read_frame  # type: ignore

if str(Path(__file__).parents[0]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[0]))
from aoi_test import get_tiff_paths_from_csv, single_aoi


class GeoInference:
    """
    A class for performing geo inference on geospatial imagery using a pre-trained model.

    Args:
        model (str): The path or url to the model file
        work_dir (str): The directory where the model and output files will be saved.
        batch_size (int): The batch size to use for inference.
        mask_to_vec (bool): Whether to convert the output mask to vector format.
        device (str): The device to use for inference (either "cpu" or "gpu").
        gpu_id (int): The ID of the GPU to use for inference (if device is "gpu").

    Attributes:
        batch_size (int): The batch size to use for inference.
        work_dir (Path): The directory where the model and output files will be saved.
        device (torch.device): The device to use for inference.
        mask_to_vec (bool): Whether to convert the output mask to vector format.
        model (torch.jit.ScriptModule): The pre-trained model to use for inference.
        classes (int): The number of classes in the output of the model.

    """

    def __init__(
        self,
        model: str = None,
        work_dir: str = None,
        batch_size: int = 1,
        mask_to_vec: bool = False,
        device: str = "gpu",
        gpu_id: int = 0,
    ):
        self.gpu_id = int(gpu_id)
        self.batch_size = int(batch_size)
        model_path = Path(
            "/gpfs/fs5/nrcan/nrcan_geobase/work/dev/datacube/parallel/deep_learning_model/4cls_RGB_5_1_2_3_scripted.pt"
        )
        self.device = torch.device("cuda")
        self.mask_to_vec = mask_to_vec
        self.model = torch.jit.load(model_path, map_location=self.device)

    @torch.no_grad()
    def __call__(
        self,
        tiff_image: str = "None",
        bbox: str = None,
        patch_size: int = 512,
        stride_size: str = None,
    ) -> None:
        raw_data_csv = "/gpfs/fs5/nrcan/nrcan_geobase/work/dev/datacube/parallel/deep_learning_repo/geo-deep-learning/tests/data/inference/test2.csv"
        work_dir = "/gpfs/fs5/nrcan/nrcan_geobase/work/dev/datacube/parallel/geo_deep_learning_results_improved_test/"
        # bands_requested = ["green", "blue", "red"]
        bands_requested = [1, 2, 3]
        data_tiff_paths = get_tiff_paths_from_csv(csv_path=raw_data_csv)
        for aoi_dict in data_tiff_paths:
            aoi = single_aoi(
                aoi_dict=aoi_dict,
                bands_requested=bands_requested,
                equalize_clahe_clip_limit=25,
                raster_stats=False,
                work_dir=work_dir,
                chunk_size=patch_size * 2,
            )
            start_time = time.time()
            if len(aoi.raster_parsed) == 1:
                aoi_dask_array = dask_imread(list(aoi.raster_parsed.values())[0])
                aoi_dask_array = da.transpose(da.squeeze(aoi_dask_array), (2, 0, 1))
            else:
                aoi_dask_array = [
                    imread(url) for band, url in aoi.raster_parsed.items()
                ]
                aoi_dask_array = da.stack(aoi_dask_array, axis=0)
                aoi_dask_array = da.squeeze(
                    da.transpose(
                        aoi_dask_array,
                        (
                            1,
                            0,
                            2,
                            3,
                        ),
                    )
                )

            if hasattr(aoi, "roi_window"):
                col_off, row_off = aoi.roi_window.col_off, aoi.roi_window.row_off
                width, height = aoi.roi_window.width, aoi.roi_window.height
                aoi_dask_array = aoi_dask_array[
                    :, row_off : row_off + height, col_off : col_off + width
                ]
            # check the contrast and if the contrast is low, apply the enhancement method on it
            pad_height = (
                patch_size - aoi_dask_array.shape[1] % patch_size
            ) % patch_size
            pad_width = (patch_size - aoi_dask_array.shape[2] % patch_size) % patch_size
            # Pad the array to make dimensions multiples of the chunk size
            data = da.pad(
                aoi_dask_array,
                ((0, 0), (0, pad_height), (0, pad_width)),
                mode="constant",
            ).rechunk(
                (3, patch_size, patch_size)
            )  # now we rechunk data so that each chunk has 3 bands
            """
            data = aoi_dask_array.rechunk(
                (3, patch_size, patch_size)
            )  # now we rechunk data so that each chunk has 3 bands
            """
            mask_array = data.map_overlap(
                runModel_partial_neighbor,
                model=self.model,
                chunks=(
                    6,
                    1024,
                    1024,
                ),
                depth={1: (0, 512), 2: (0, 512)},
                boundary="none",
                trim=False,
                dtype=np.float16,
            )
            aoi_dask_array_3 = da.map_overlap(
                sum_overlapped_chunks,
                mask_array,
                chunk_size=1024,
                drop_axis=0,
                chunks=(
                    512,
                    512,
                ),
                depth={1: (512, 0), 2: (512, 0)},
                trim=False,
                boundary="none",
                dtype=np.uint8,
            )

            final_array = aoi_dask_array_3[
                : aoi_dask_array.shape[1], : aoi_dask_array.shape[2]
            ].compute(n_workers=16)
            aoi.write_inference_to_tiff(final_array, False)
            total_time = time.time() - start_time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(
                f"The total time of running Inference is {int(hours)}:{int(minutes)}:{seconds:.2f} \n"
            )


def infer_chip(data: torch.Tensor, model) -> torch.Tensor:
    # Input is GPU, output is GPU.
    with torch.no_grad():
        result = model(data).argmax(dim=1).to(torch.uint8)
    return result.to("cpu")


def copy_and_infer_chunked(tile, model, block_info=None, token=None):
    device = torch.device("cuda")
    tensor = torch.as_tensor(tile[np.newaxis, ...]).to(device)
    out = np.empty(
        shape=(5, tile.shape[1], tile.shape[2])
    )  # Create the output but empty
    # run the model
    with torch.no_grad():
        out = model(tensor).cpu().numpy()[0]
    del tensor
    x = np.argmax(out, axis=0).astype(np.uint8)
    return x


def runModel_partial_neighbor2(
    chunk_data,
    model,
    block_info=None,
):
    """
    This function is for running the model on partial neighbor --> The right and bottom neighbors
    After running the model, depending on the location of chuck, it multiplies the chunk with a window and adds the windows to another dimension of the chunk
    This window is used to deal with edge artifact
    @param chunk_data: np.ndarray, this is a chunk of data in dask array
            chunk_size: int, the size of chunk data that we want to feed the model with
            model_path: str, the path to the scripted model
            block_info: none, this is having all the info about the chunk relative to the whole data (dask array)
    @return: predited chunks
    """
    num_chunks = block_info[0]["num-chunks"]
    chunk_location = block_info[0]["chunk-location"]
    if chunk_data.size > 0 and chunk_data is not None:
        try:
            # Defining the base window for window creation later

            step = 1024 >> 1
            window = w.hann(M=1024, sym=False)
            window = window[:, np.newaxis] * window[np.newaxis, :]
            final_window = np.empty((1, 1))

            chunk_data_modified = chunk_data
            ignore_last_chunk_x = False
            ignore_last_chunk_y = False

            if chunk_location[2] == num_chunks[2] - 2 and chunk_location[1] == 0:
                if chunk_data.shape[2] < 1024:
                    chunk_data_modified = np.pad(
                        chunk_data,
                        (
                            (0, 0),
                            (0, 0),
                            (0, 1024 - chunk_data.shape[2]),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    window_u = np.vstack(
                        [
                            np.tile(window[step : step + 1, :], (step, 1)),
                            window[step:, :],
                        ]
                    )
                    window_r = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_u[:step, :step], np.ones((step, step))],
                            [window_r[step:, :step], window_r[step:, step:]],
                        ]
                    )
                else:
                    final_window = np.vstack(
                        [
                            np.tile(window[step : step + 1, :], (step, 1)),
                            window[step:, :],
                        ]
                    )
            elif chunk_location[2] == num_chunks[2] - 1 and chunk_location[1] == 0:
                if chunk_data.shape[2] <= int(1024 / 2):
                    ignore_last_chunk_x = True
                elif chunk_data.shape[2] > int(1024 / 2):
                    chunk_data_modified = chunk_data[:, :, int(1024 / 2) :]
                    chunk_data_modified = np.pad(
                        chunk_data_modified,
                        (
                            (0, 0),
                            (0, 0),
                            (0, 1024 - chunk_data_modified.shape[2]),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    window_u = np.vstack(
                        [
                            np.tile(window[step : step + 1, :], (step, 1)),
                            window[step:, :],
                        ]
                    )
                    window_r = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_u[:step, :step], np.ones((step, step))],
                            [window_r[step:, :step], window_r[step:, step:]],
                        ]
                    )
            elif chunk_location[2] == num_chunks[2] - 2 and (
                chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 2
            ):
                # left egde window
                if chunk_data.shape[2] < 1024:
                    chunk_data_modified = np.pad(
                        chunk_data,
                        (
                            (0, 0),
                            (0, 0),
                            (0, 1024 - chunk_data.shape[2]),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    final_window = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                else:
                    final_window = window
            elif chunk_location[2] == num_chunks[2] - 1 and (
                chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 2
            ):
                # left egde window
                if chunk_data.shape[2] <= int(1024 / 2):
                    ignore_last_chunk_x = True
                else:
                    chunk_data_modified = chunk_data[:, :, int(1024 / 2) :]
                    chunk_data_modified = np.pad(
                        chunk_data_modified,
                        (
                            (0, 0),
                            (0, 0),
                            (0, 1024 - chunk_data_modified.shape[2]),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    final_window = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
            elif chunk_location[2] == num_chunks[2] - 2 and (
                chunk_location[1] == num_chunks[1] - 2
            ):
                # bottom right window
                if chunk_data.shape[2] == 1024 and chunk_data.shape[1] == 1024:
                    final_window = window
                elif chunk_data.shape[2] < 1024 and chunk_data.shape[1] == 1024:
                    chunk_data_modified = np.pad(
                        chunk_data,
                        (
                            (0, 0),
                            (0, 0),
                            (0, 1024 - chunk_data.shape[2]),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    final_window = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                elif chunk_data.shape[2] == 1024 and chunk_data.shape[1] < 1024:
                    chunk_data_modified = np.pad(
                        chunk_data,
                        (
                            (0, 0),
                            (0, 1024 - chunk_data.shape[1]),
                            (0, 0),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    final_window = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
                elif chunk_data.shape[2] < 1024 and chunk_data.shape[1] < 1024:
                    chunk_data_modified = np.pad(
                        chunk_data,
                        (
                            (0, 0),
                            (0, 1024 - chunk_data.shape[1]),
                            (0, 1024 - chunk_data.shape[2]),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    window_r = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                    window_b = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_r[:step, :step], window_r[:step, step:]],
                            [window_b[step:, :step], np.ones((step, step))],
                        ]
                    )
            elif chunk_location[2] == num_chunks[2] - 1 and (
                chunk_location[1] == num_chunks[1] - 1
            ):
                # bottom right window
                if chunk_data.shape[2] <= int(1024 / 2) and chunk_data.shape[1] <= int(
                    1024 / 2
                ):
                    ignore_last_chunk_x = True
                    ignore_last_chunk_y = True
                elif chunk_data.shape[2] > int(1024 / 2) and chunk_data.shape[1] <= int(
                    1024 / 2
                ):
                    chunk_data_modified = chunk_data[:, :, int(1024 / 2) :]
                    chunk_data_modified = np.pad(
                        chunk_data_modified,
                        (
                            (0, 0),
                            (0, 1024 - chunk_data_modified.shape[1]),
                            (0, 1024 - chunk_data_modified.shape[2]),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    window_r = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                    window_b = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_r[:step, :step], window_r[:step, step:]],
                            [window_b[step:, :step], np.ones((step, step))],
                        ]
                    )
                elif chunk_data.shape[2] <= int(1024 / 2) and chunk_data.shape[1] > int(
                    1024 / 2
                ):
                    chunk_data_modified = chunk_data[:, int(1024 / 2) :, :]
                    chunk_data_modified = np.pad(
                        chunk_data_modified,
                        (
                            (0, 0),
                            (0, 1024 - chunk_data_modified.shape[1]),
                            (0, 1024 - chunk_data_modified.shape[2]),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    window_r = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                    window_b = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_r[:step, :step], window_r[:step, step:]],
                            [window_b[step:, :step], np.ones((step, step))],
                        ]
                    )
                elif chunk_data.shape[2] > int(1024 / 2) and chunk_data.shape[1] > int(
                    1024 / 2
                ):
                    chunk_data_modified = chunk_data[
                        :, int(1024 / 2) :, int(1024 / 2) :
                    ]
                    chunk_data_modified = np.pad(
                        chunk_data_modified,
                        (
                            (0, 0),
                            (0, 1024 - chunk_data_modified.shape[1]),
                            (0, 1024 - chunk_data_modified.shape[2]),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    window_r = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                    window_b = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_r[:step, :step], window_r[:step, step:]],
                            [window_b[step:, :step], np.ones((step, step))],
                        ]
                    )
            elif chunk_location[1] == num_chunks[1] - 2 and (
                chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 2
            ):
                # bottom egde window
                if chunk_data.shape[1] < 1024:
                    chunk_data_modified = np.pad(
                        chunk_data,
                        (
                            (0, 0),
                            (0, 1024 - chunk_data.shape[1]),
                            (0, 0),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    final_window = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
                else:
                    final_window = window
            elif chunk_location[1] == num_chunks[1] - 1 and (
                chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 2
            ):
                if chunk_data.shape[1] <= int(1024 / 2):
                    ignore_last_chunk_y = True
                elif chunk_data.shape[1] > int(1024 / 2):
                    chunk_data_modified = chunk_data[:, int(1024 / 2) :, :]
                    chunk_data_modified = np.pad(
                        chunk_data_modified,
                        (
                            (0, 0),
                            (0, 1024 - chunk_data_modified.shape[1]),
                            (0, 0),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    final_window = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
            elif chunk_location[1] == num_chunks[1] - 1 and chunk_location[2] == 0:
                # bottom left window
                if chunk_data.shape[1] <= int(1024 / 2):
                    ignore_last_chunk_y = True
                else:
                    chunk_data_modified = chunk_data[:, int(1024 / 2) :, :]
                    chunk_data_modified = np.pad(
                        chunk_data_modified,
                        (
                            (0, 0),
                            (0, 1024 - chunk_data_modified.shape[1]),
                            (0, 0),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    final_window = np.hstack(
                        [
                            np.tile(window[:, step : step + 1], (1, step)),
                            window[:, step:],
                        ]
                    )
            elif chunk_location[1] == num_chunks[1] - 2 and chunk_location[2] == 0:
                # bottom egde window
                if chunk_data.shape[1] < 1024:
                    chunk_data_modified = np.pad(
                        chunk_data,
                        (
                            (0, 0),
                            (0, 0),
                            (0, 1024 - chunk_data.shape[2]),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    final_window = np.hstack(
                        [
                            np.tile(window[:, step : step + 1], (1, step)),
                            window[:, step:],
                        ]
                    )
                else:
                    window_b = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
                    window_l = np.hstack(
                        [
                            np.tile(window[:, step : step + 1], (1, step)),
                            window[:, step:],
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_l[:step, :step], window_l[:step, step:]],
                            [np.ones((step, step)), window_b[step:, step:]],
                        ]
                    )
            elif chunk_location[1] == 0 and chunk_location[2] == 0:
                # Top left window
                window_u = np.vstack(
                    [
                        np.tile(window[step : step + 1, :], (step, 1)),
                        window[step:, :],
                    ]
                )
                window_l = np.hstack(
                    [
                        np.tile(window[:, step : step + 1], (1, step)),
                        window[:, step:],
                    ]
                )
                final_window = np.block(
                    [
                        [np.ones((step, step)), window_u[:step, step:]],
                        [window_l[step:, :step], window_l[step:, step:]],
                    ]
                )
            elif chunk_location[2] == 0 and (
                chunk_location[1] > 0 and chunk_location[1] < num_chunks[2] - 2
            ):
                # top edge window
                final_window = np.hstack(
                    [np.tile(window[:, step : step + 1], (1, step)), window[:, step:]]
                )
            elif (chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 2) and (
                chunk_location[1] == 0
            ):
                # top edge window
                final_window = np.vstack(
                    [
                        np.tile(window[step : step + 1, :], (step, 1)),
                        window[step:, :],
                    ]
                )
            elif (chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 2) and (
                chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 2
            ):
                final_window = window

            device = torch.device("cuda")
            tensor = torch.as_tensor(chunk_data_modified[np.newaxis, ...]).to(device)
            out = np.empty(
                shape=(5, chunk_data_modified.shape[1], chunk_data_modified.shape[2])
            )  # Create the output but empty
            # run the model
            with torch.no_grad():
                out = model(tensor).cpu().numpy()[0]
            del tensor
            if (
                out.shape[1:] == final_window.shape
                and out.shape[1:] == (1024, 1024)
                and not ignore_last_chunk_x == False
                and not ignore_last_chunk_y == False
            ):
                return np.concatenate(
                    (out * final_window, final_window[np.newaxis, :, :]), axis=0
                )
            else:
                return np.zeros((6, 1024, 1024))
                """ 
                out_modified = np.pad(
                    out,
                    ((0, 0), (0, 1024 - out.shape[1]), (0, 1024 - out.shape[2])),
                    mode="constant",
                    constant_values=0,
                )
                if out_modified.shape[1:] == final_window.shape and out_modified.shape[
                    1:
                ] == (1024, 1024):
                    return np.concatenate(
                        (out_modified * final_window, final_window[np.newaxis, :, :]),
                        axis=0,
                    )
                """
        except Exception as e:
            logging.error(f"Error occured in IrunModel: {e}")
            tb = traceback.format_exc()
            logging.error(f"Error occurred in IrunModel: {e}. Traceback:\n{tb}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Release unused memory


def runModel_partial_neighbor(
    chunk_data,
    model,
    block_info=None,
):
    """
    This function is for running the model on partial neighbor --> The right and bottom neighbors
    After running the model, depending on the location of chuck, it multiplies the chunk with a window and adds the windows to another dimension of the chunk
    This window is used to deal with edge artifact
    @param chunk_data: np.ndarray, this is a chunk of data in dask array
            chunk_size: int, the size of chunk data that we want to feed the model with
            model_path: str, the path to the scripted model
            block_info: none, this is having all the info about the chunk relative to the whole data (dask array)
    @return: predited chunks
    """
    num_chunks = block_info[0]["num-chunks"]
    chunk_location = block_info[0]["chunk-location"]
    if chunk_data.size > 0 and chunk_data is not None:
        try:
            # Defining the base window for window creation later

            step = 1024 >> 1
            window = w.hann(M=1024, sym=False)
            window = window[:, np.newaxis] * window[np.newaxis, :]
            final_window = np.empty((1, 1))
            if chunk_location[2] >= num_chunks[2] - 2 and chunk_location[1] == 0:
                window_u = np.vstack(
                    [
                        np.tile(window[step : step + 1, :], (step, 1)),
                        window[step:, :],
                    ]
                )
                window_r = np.hstack(
                    [
                        window[:, :step],
                        np.tile(window[:, step : step + 1], (1, step)),
                    ]
                )
                final_window = np.block(
                    [
                        [window_u[:step, :step], np.ones((step, step))],
                        [window_r[step:, :step], window_r[step:, step:]],
                    ]
                )
            elif chunk_location[2] >= num_chunks[2] - 2 and (
                chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 2
            ):
                # left egde window
                final_window = np.hstack(
                    [
                        window[:, :step],
                        np.tile(window[:, step : step + 1], (1, step)),
                    ]
                )
            elif chunk_location[2] >= num_chunks[2] - 2 and (
                chunk_location[1] >= num_chunks[1] - 2
            ):
                # bottom right window
                window_r = np.hstack(
                    [
                        window[:, :step],
                        np.tile(window[:, step : step + 1], (1, step)),
                    ]
                )
                window_b = np.vstack(
                    [
                        window[:step, :],
                        np.tile(window[step : step + 1, :], (step, 1)),
                    ]
                )
                final_window = np.block(
                    [
                        [window_r[:step, :step], window_r[:step, step:]],
                        [window_b[step:, :step], np.ones((step, step))],
                    ]
                )
            elif chunk_location[1] >= num_chunks[1] - 2 and (
                chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 2
            ):
                # bottom egde window
                final_window = np.vstack(
                    [
                        window[:step, :],
                        np.tile(window[step : step + 1, :], (step, 1)),
                    ]
                )
            elif chunk_location[1] >= num_chunks[1] - 2 and chunk_location[2] == 0:
                # bottom left window
                window_l = np.hstack(
                    [
                        np.tile(window[:, step : step + 1], (1, step)),
                        window[:, step:],
                    ]
                )
                window_b = np.vstack(
                    [
                        window[:step, :],
                        np.tile(window[step : step + 1, :], (step, 1)),
                    ]
                )
                final_window = np.block(
                    [
                        [window_l[:step, :step], window_l[:step, step:]],
                        [np.ones((step, step)), window_b[step:, step:]],
                    ]
                )
            elif chunk_location[1] == 0 and chunk_location[2] == 0:
                # Top left window
                window_u = np.vstack(
                    [
                        np.tile(window[step : step + 1, :], (step, 1)),
                        window[step:, :],
                    ]
                )
                window_l = np.hstack(
                    [
                        np.tile(window[:, step : step + 1], (1, step)),
                        window[:, step:],
                    ]
                )
                final_window = np.block(
                    [
                        [np.ones((step, step)), window_u[:step, step:]],
                        [window_l[step:, :step], window_l[step:, step:]],
                    ]
                )
            elif chunk_location[2] == 0 and (
                chunk_location[1] > 0 and chunk_location[1] < num_chunks[1]
            ):
                # top edge window
                final_window = np.hstack(
                    [
                        np.tile(window[:, step : step + 1], (1, step)),
                        window[:, step:],
                    ]
                )
            elif (chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 2) and (
                chunk_location[1] == 0
            ):
                # top edge window
                final_window = np.vstack(
                    [
                        np.tile(window[step : step + 1, :], (step, 1)),
                        window[step:, :],
                    ]
                )
            elif (chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 2) and (
                chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 2
            ):
                final_window = window

            device = torch.device("cuda")
            tensor = torch.as_tensor(chunk_data[np.newaxis, ...]).to(device)
            out = np.empty(
                shape=(5, chunk_data.shape[1], chunk_data.shape[2])
            )  # Create the output but empty
            # run the model
            with torch.no_grad():
                out = model(tensor).cpu().numpy()[0]
            del tensor
            if out.shape[1:] == final_window.shape and out.shape[1:] == (1024, 1024):
                return np.concatenate(
                    (out * final_window, final_window[np.newaxis, :, :]), axis=0
                )
            else:
                return np.zeros((6, 1024, 1024))
                """ 
                out_modified = np.pad(
                    out,
                    ((0, 0), (0, 1024 - out.shape[1]), (0, 1024 - out.shape[2])),
                    mode="constant",
                    constant_values=0,
                )
                if out_modified.shape[1:] == final_window.shape and out_modified.shape[
                    1:
                ] == (1024, 1024):
                    return np.concatenate(
                        (out_modified * final_window, final_window[np.newaxis, :, :]),
                        axis=0,
                    )
                """
        except Exception as e:
            logging.error(f"Error occured in IrunModel: {e}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Release unused memory


def sum_overlapped_chunks(
    aoi_chunk: np.ndarray,
    chunk_size: int,
    block_info=None,
):
    if aoi_chunk.size > 0 and aoi_chunk is not None:
        num_chunks = block_info[0]["num-chunks"]
        chunk_location = block_info[0]["chunk-location"]
        full_array = np.empty((1, 1))
        if (chunk_location[1] == 0 or chunk_location[1] == num_chunks[1] - 1) and (
            chunk_location[2] == 0 or chunk_location[2] == num_chunks[2] - 1
        ):
            """ All 4 corners"""
            full_array = aoi_chunk[
                :,
                : int(chunk_size / 2),
                : int(chunk_size / 2),
            ]
        elif (chunk_location[1] == 0 or chunk_location[1] == num_chunks[1] - 1) and (
            chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 1
        ):
            """ Top and bottom edges but not corners"""
            full_array = (
                aoi_chunk[
                    :,
                    : int(chunk_size / 2),
                    int(chunk_size / 2) : int(chunk_size / 2) * 2,
                ]
                + aoi_chunk[
                    :,
                    : int(chunk_size / 2),
                    : int(chunk_size / 2),
                ]
            )
        elif (chunk_location[2] == 0 or chunk_location[2] == num_chunks[2] - 1) and (
            chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 1
        ):
            """ Left and right edges but not corners"""
            full_array = (
                aoi_chunk[
                    :,
                    int(chunk_size / 2) : int(chunk_size / 2) * 2,
                    : int(chunk_size / 2),
                ]
                + aoi_chunk[
                    :,
                    : int(chunk_size / 2),
                    : int(chunk_size / 2),
                ]
            )
        elif (chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 1) and (
            chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 1
        ):
            """ Middle chunks """
            full_array = (
                aoi_chunk[
                    :,
                    : int(chunk_size / 2),
                    : int(chunk_size / 2),
                ]
                + aoi_chunk[
                    :,
                    : int(chunk_size / 2),
                    int(chunk_size / 2) : int(chunk_size / 2) * 2,
                ]
                + aoi_chunk[
                    :,
                    int(chunk_size / 2) : int(chunk_size / 2) * 2,
                    : int(chunk_size / 2),
                ]
                + aoi_chunk[
                    :,
                    int(chunk_size / 2) : int(chunk_size / 2) * 2,
                    int(chunk_size / 2) : int(chunk_size / 2) * 2,
                ]
            )

        if full_array.shape != (
            aoi_chunk.shape[0],
            int(chunk_size / 2),
            int(chunk_size / 2),
        ):
            logging.error(
                f" In sum_overlapped_chunks the shape of full_array is not {(6, int(chunk_size / 2), int(chunk_size / 2))}"
                f" The size of it {full_array.shape}"
            )
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                final_result = np.divide(
                    full_array[:-1, :, :],
                    full_array[-1, :, :][np.newaxis, :, :],
                    out=np.zeros_like(full_array[:-1, :, :], dtype=float),
                    where=full_array[-1, :, :] != 0,
                )
                if final_result.shape[0] == 1:
                    final_result = expit(final_result)
                    final_result = (
                        np.where(final_result > 0.5, 1, 0).squeeze(0).astype(np.uint8)
                    )
                else:
                    final_result = np.argmax(final_result, axis=0).astype(np.uint8)
                return final_result


def imread(fname, nframes=1, *, arraytype="numpy"):
    sfname = str(fname)
    with pims.open(sfname) as imgs:
        shape = (1,) + imgs.frame_shape
        dtype = np.dtype(imgs.pixel_type)
    ar = da.from_array([sfname] * shape[0], chunks=(nframes,))
    a = ar.map_blocks(
        _map_read_frame,
        chunks=da.core.normalize_chunks((nframes,) + shape[1:], shape),
        multiple_files=False,
        new_axis=list(range(1, len(shape))),
        arrayfunc=np.asanyarray,
        meta=np.asanyarray([]).astype(dtype),  # meta overwrites `dtype` argument
    )
    return a


def runModel_partial_neighbor_test(
    chunk_data: np.ndarray,
    chunk_size: int,
    model,
    multi_gpu: bool = False,
    gpu_id: int = 0,
    num_classes: int = 5,
    block_info=None,
):
    """
    # UPDATE it
    This function is for running the model on partial neighbor --> The right and bottom neighbors
    After running the model, depending on the location of chuck, it multiplies the chunk with a window and adds the windows to another dimension of the chunk
    This window is used to deal with edge artifact
    @param chunk_data: np.ndarray, this is a chunk of data in dask array
            chunk_size: int, the size of chunk data that we want to feed the model with
            model_path: str, the path to the scripted model
            block_info: none, this is having all the info about the chunk relative to the whole data (dask array)
    @return: predited chunks
    """
    num_chunks = block_info[0]["num-chunks"]
    chunk_location = block_info[0]["chunk-location"]
    if chunk_data.size > 0 and chunk_data is not None:
        try:
            # Defining the base window for window creation later
            step = chunk_size >> 1
            window = w.hann(M=chunk_size, sym=False)
            window = window[:, np.newaxis] * window[np.newaxis, :]
            final_window = np.empty((1, 1))
            chunk_data_ = chunk_data
            if chunk_location[2] == num_chunks[2] - 2 and chunk_location[1] == 0:
                if chunk_data_.shape[2] < chunk_size:
                    chunk_data_ = np.pad(
                        chunk_data,
                        ((0, 0), (0, 0), (0, chunk_size - chunk_data.shape[2])),
                        mode="constant",
                        constant_values=0,
                    )

                    window_u = np.vstack(
                        [
                            np.tile(window[step : step + 1, :], (step, 1)),
                            window[step:, :],
                        ]
                    )
                    window_r = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_u[:step, :step], np.ones((step, step))],
                            [window_r[step:, :step], window_r[step:, step:]],
                        ]
                    )
                else:
                    final_window = np.vstack(
                        [
                            np.tile(window[step : step + 1, :], (step, 1)),
                            window[step:, :],
                        ]
                    )
            elif chunk_location[2] == num_chunks[2] - 1 and chunk_location[1] == 0:
                if chunk_data_.shape[2] > chunk_size:
                    chunk_data_ = chunk_data[:, :, int(chunk_size / 2) :]
                    chunk_data_ = np.pad(
                        chunk_data_,
                        ((0, 0), (0, 0), (0, chunk_size - chunk_data.shape[2])),
                        mode="constant",
                        constant_values=0,
                    )

                    window_u = np.vstack(
                        [
                            np.tile(window[step : step + 1, :], (step, 1)),
                            window[step:, :],
                        ]
                    )
                    window_r = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_u[:step, :step], np.ones((step, step))],
                            [window_r[step:, :step], window_r[step:, step:]],
                        ]
                    )
            elif chunk_location[2] == num_chunks[2] - 2 and (
                chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 2
            ):
                if chunk_data_.shape[2] < chunk_size:
                    chunk_data_ = np.pad(
                        chunk_data,
                        ((0, 0), (0, 0), (0, chunk_size - chunk_data.shape[2])),
                        mode="constant",
                        constant_values=0,
                    )
                    final_window = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                else:
                    final_window = window
            elif chunk_location[2] == num_chunks[2] - 1 and (
                chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 2
            ):
                if chunk_data_.shape[2] > chunk_size:
                    chunk_data_ = chunk_data[:, :, int(chunk_size / 2) :]
                    chunk_data_ = np.pad(
                        chunk_data_,
                        ((0, 0), (0, 0), (0, chunk_size - chunk_data.shape[2])),
                        mode="constant",
                        constant_values=0,
                    )
                    final_window = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
            elif chunk_location[2] == num_chunks[2] - 2 and (
                chunk_location[1] == num_chunks[1] - 2
            ):
                if (
                    chunk_data_.shape[2] < chunk_size
                    and chunk_data_.shape[1] == chunk_size
                ):
                    chunk_data_ = np.pad(
                        chunk_data,
                        ((0, 0), (0, 0), (0, chunk_size - chunk_data.shape[2])),
                        mode="constant",
                        constant_values=0,
                    )
                    final_window = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                elif (
                    chunk_data_.shape[2] == chunk_size
                    and chunk_data_.shape[1] < chunk_size
                ):
                    chunk_data_ = np.pad(
                        chunk_data,
                        ((0, 0), (0, chunk_size - chunk_data.shape[2]), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )
                    final_window = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ],
                    )
                elif (
                    chunk_data_.shape[2] < chunk_size
                    and chunk_data_.shape[1] < chunk_size
                ):
                    chunk_data_ = np.pad(
                        chunk_data,
                        (
                            (0, 0),
                            (0, chunk_size - chunk_data.shape[2]),
                            (0, chunk_size - chunk_data.shape[2]),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    window_r = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                    window_b = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_r[:step, :step], window_r[:step, step:]],
                            [window_b[step:, :step], np.ones((step, step))],
                        ]
                    )
            elif chunk_location[2] == num_chunks[2] - 2 and (
                chunk_location[1] == num_chunks[1] - 1
            ):
                if chunk_data_.shape[1] > chunk_size:
                    chunk_data_ = chunk_data[:, int(chunk_size / 2) :, :]
                    chunk_data_ = np.pad(
                        chunk_data_,
                        (
                            (0, 0),
                            (0, chunk_size - chunk_data.shape[1]),
                            (0, 0),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    final_window = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
            elif chunk_location[2] == num_chunks[2] - 1 and (
                chunk_location[1] == num_chunks[1] - 2
            ):
                if chunk_data_.shape[2] > chunk_size:
                    chunk_data_ = chunk_data[:, :, int(chunk_size / 2) :]
                    chunk_data_ = np.pad(
                        chunk_data_,
                        (
                            (0, 0),
                            (0, 0),
                            (0, chunk_size - chunk_data.shape[2]),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    final_window = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
            elif chunk_location[2] == num_chunks[2] - 1 and (
                chunk_location[1] == num_chunks[1] - 1
            ):
                if (
                    chunk_data_.shape[2] > chunk_size
                    and chunk_data_.shape[1] == chunk_size
                ):
                    chunk_data_ = chunk_data[:, :, int(chunk_size / 2) :]
                    chunk_data_ = np.pad(
                        chunk_data_,
                        (
                            (0, 0),
                            (0, 0),
                            (0, chunk_size - chunk_data.shape[2]),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    window_r = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                    window_b = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_r[:step, :step], window_r[:step, step:]],
                            [window_b[step:, :step], np.ones((step, step))],
                        ]
                    )
                elif (
                    chunk_data_.shape[2] == chunk_size
                    and chunk_data_.shape[1] > chunk_size
                ):
                    chunk_data_ = chunk_data[:, int(chunk_size / 2) :, :]
                    chunk_data_ = np.pad(
                        chunk_data_,
                        (
                            (0, 0),
                            (0, chunk_size - chunk_data.shape[1]),
                            (0, 0),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    window_r = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                    window_b = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_r[:step, :step], window_r[:step, step:]],
                            [window_b[step:, :step], np.ones((step, step))],
                        ]
                    )
                elif (
                    chunk_data_.shape[2] > chunk_size
                    and chunk_data_.shape[1] > chunk_size
                ):
                    chunk_data_ = chunk_data[
                        :, int(chunk_size / 2) :, int(chunk_size / 2) :
                    ]
                    chunk_data_ = np.pad(
                        chunk_data_,
                        (
                            (0, 0),
                            (0, chunk_size - chunk_data.shape[1]),
                            (0, chunk_size - chunk_data.shape[2]),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    window_r = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                    window_b = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_r[:step, :step], window_r[:step, step:]],
                            [window_b[step:, :step], np.ones((step, step))],
                        ]
                    )
            elif chunk_location[1] == num_chunks[1] - 2 and (
                chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 2
            ):
                if chunk_data_.shape[1] < chunk_size:
                    chunk_data_ = np.pad(
                        chunk_data,
                        (
                            (0, 0),
                            (0, chunk_size - chunk_data.shape[1]),
                            (0, 0),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    final_window = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
                else:
                    final_window = window
            elif chunk_location[1] == num_chunks[1] - 1 and (
                chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 2
            ):
                if chunk_data_.shape[1] > chunk_size:
                    chunk_data_ = chunk_data[
                        :,
                        int(chunk_size / 2) :,
                        :,
                    ]
                    chunk_data_ = np.pad(
                        chunk_data_,
                        (
                            (0, 0),
                            (0, chunk_size - chunk_data.shape[2]),
                            (0, 0),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    final_window = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
            elif chunk_location[1] == num_chunks[1] - 1 and (chunk_location[2] == 0):
                if chunk_data_.shape[1] > chunk_size:
                    chunk_data_ = chunk_data[
                        :,
                        int(chunk_size / 2) :,
                        :,
                    ]
                    chunk_data_ = np.pad(
                        chunk_data_,
                        (
                            (0, 0),
                            (0, chunk_size - chunk_data.shape[2]),
                            (0, 0),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    window_b = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
                    window_l = np.hstack(
                        [
                            np.tile(window[:, step : step + 1], (1, step)),
                            window[:, step:],
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_l[:step, :step], window_l[:step, step:]],
                            [np.ones((step, step)), window_b[step:, step:]],
                        ]
                    )
            elif chunk_location[1] == num_chunks[1] - 2 and (chunk_location[2] == 0):
                if chunk_data_.shape[1] < chunk_size:
                    chunk_data_ = np.pad(
                        chunk_data,
                        (
                            (0, 0),
                            (0, chunk_size - chunk_data.shape[1]),
                            (0, 0),
                        ),
                        mode="constant",
                        constant_values=0,
                    )
                    window_b = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
                    window_l = np.hstack(
                        [
                            np.tile(window[:, step : step + 1], (1, step)),
                            window[:, step:],
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_l[:step, :step], window_l[:step, step:]],
                            [np.ones((step, step)), window_b[step:, step:]],
                        ]
                    )
                else:
                    final_window = np.hstack(
                        [
                            np.tile(window[:, step : step + 1], (1, step)),
                            window[:, step:],
                        ]
                    )
            elif chunk_location[1] >= num_chunks[1] - 2 and chunk_location[2] == 0:
                # bottom left window
                window_l = np.hstack(
                    [
                        np.tile(window[:, step : step + 1], (1, step)),
                        window[:, step:],
                    ]
                )
                window_b = np.vstack(
                    [
                        window[:step, :],
                        np.tile(window[step : step + 1, :], (step, 1)),
                    ]
                )
                final_window = np.block(
                    [
                        [window_l[:step, :step], window_l[:step, step:]],
                        [np.ones((step, step)), window_b[step:, step:]],
                    ]
                )
            elif chunk_location[1] == 0 and chunk_location[2] == 0:
                # Top left window
                window_u = np.vstack(
                    [
                        np.tile(window[step : step + 1, :], (step, 1)),
                        window[step:, :],
                    ]
                )
                window_l = np.hstack(
                    [
                        np.tile(window[:, step : step + 1], (1, step)),
                        window[:, step:],
                    ]
                )
                final_window = np.block(
                    [
                        [np.ones((step, step)), window_u[:step, step:]],
                        [window_l[step:, :step], window_l[step:, step:]],
                    ]
                )
            elif chunk_location[2] == 0 and (
                chunk_location[1] > 0 and chunk_location[1] < num_chunks[1]
            ):
                # top edge window
                final_window = np.hstack(
                    [
                        np.tile(window[:, step : step + 1], (1, step)),
                        window[:, step:],
                    ]
                )
            elif (chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 2) and (
                chunk_location[1] == 0
            ):
                # top edge window
                final_window = np.vstack(
                    [
                        np.tile(window[step : step + 1, :], (step, 1)),
                        window[step:, :],
                    ]
                )
            elif (chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 2) and (
                chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 2
            ):
                final_window = window

            device = torch.device("cuda")
            input_tensor = torch.as_tensor(chunk_data[np.newaxis, ...]).to(device)
            model_out = np.empty(
                shape=(num_classes, chunk_data.shape[1], chunk_data.shape[2])
            )  # Create the output but empty
            # run the model
            with torch.no_grad():
                model_out = model(input_tensor).cpu().numpy()[0]
            del input_tensor
            if model_out.shape[1:] == final_window.shape and model_out.shape[1:] == (
                chunk_size,
                chunk_size,
            ):
                return np.concatenate(
                    (model_out * final_window, final_window[np.newaxis, :, :]), axis=0
                )
            else:
                return np.zeros((6, chunk_size, chunk_size))
        except Exception as e:
            logging.error(f"Error occured in IrunModel: {e}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Release unused memory


def main() -> None:
    geo_inference = GeoInference(
        work_dir="/space/partner/nrcan/geobase/work/dev/datacube/parallel/Change_detection_Results",
        batch_size=512,
        mask_to_vec=False,
    )
    geo_inference()


if __name__ == "__main__":
    main()

import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from tqdm import tqdm

import numpy as np
import dask
import rioxarray
import xarray as xr


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
        # model_path: Path = get_model(model_path_or_url=model,
        #                             work_dir=self.work_dir)
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
        """
        Perform geo inference on geospatial imagery.

        Args:
            tiff_image (str): The path to the geospatial image to perform inference on.
            bbox (str): The bbox or extent of the image in this format "minx, miny, maxx, maxy"
            patch_size (int): The size of the patches to use for inference.
            stride_size (int): The stride to use between patches.

        Returns:
            None

        """
        self.mask_to_vec = False

        tiff_image = "BC18-013904302070_01_P001-GE01_clahe25"
        ratio = 0.5

        # image_prefix = Path(tiff_image).parent
        image_prefix = f"/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/operationalization/data/{tiff_image}"
        mask_path = "/gpfs/fs5/nrcan/nrcan_geobase/work/dev/datacube/parallel/Change_detection_Results/mas_mask.tif"

        r = rioxarray.open_rasterio(f"{image_prefix}-R.tif", chunks=patch_size)
        g = rioxarray.open_rasterio(f"{image_prefix}-G.tif", chunks=patch_size)
        b = rioxarray.open_rasterio(f"{image_prefix}-B.tif", chunks=patch_size)
        dataset = xr.concat([r, g, b], dim="band")
        # print(dataset)

        start_time = time.time()

        meta = np.array([[]], dtype="uint8")[:0]

        mask_array = dataset.data.map_overlap(
            copy_and_infer_chunked,
            meta=meta,
            drop_axis=0,
            model=self.model,
            name="predict",
            depth={1: patch_size * ratio, 2: patch_size * ratio},
            trim=True,
        )
        # print(mask_array)

        mask = xr.DataArray(
            mask_array,
            coords=dataset.drop_vars("band").coords,
            dims=("y", "x"),
        )
        mask.rio.to_raster(mask_path)

        end_time = time.time() - start_time


def infer_chip(data: torch.Tensor, model) -> torch.Tensor:
    # Input is GPU, output is GPU.
    with torch.no_grad():
        result = model(data).argmax(dim=1).to(torch.uint8)
    return result.to("cpu")


def copy_and_infer_chunked(tile, model, token=None):
    slices = dask.array.core.slices_from_chunks(dask.array.empty(tile.shape).chunks)
    out = np.empty(shape=tile.shape[1:], dtype="uint8")
    device = torch.device("cuda")

    for slice_ in slices:
        gpu_chip = torch.as_tensor(tile[slice_][np.newaxis, ...]).to(device)
        out[slice_[1:]] = infer_chip(gpu_chip, model).cpu().numpy()[0]
    return out


def main() -> None:
    geo_inference = GeoInference(
        work_dir="/space/partner/nrcan/geobase/work/dev/datacube/parallel/Change_detection_Results",
        batch_size=512,
        mask_to_vec=False,
    )
    geo_inference()


if __name__ == "__main__":
    main()

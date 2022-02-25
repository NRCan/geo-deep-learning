# Licensed under the MIT License.
# Adapted from: https://github.com/microsoft/torchgeo/blob/3f7e525fbd01dddd25804e7a1b7634269ead1760/evaluate.py

# Hardware requirements: 64 Gb RAM (cpu), 8 Gb GPU RAM.

"""CCMEO model inference script."""
import argparse
import logging
from pathlib import Path
from time import sleep
from typing import Dict, Any, Optional, Callable, Sequence, List, cast, Union, Tuple, Iterator

import docker
import numpy as np
import pystac
import rasterio
from docker.errors import DockerException
from omegaconf import OmegaConf
from pystac.extensions.eo import ItemEOExtension, Band
from pytorch_lightning import LightningDataModule, LightningModule, seed_everything
import rasterio.windows
from rasterio.crs import CRS
from rtree import Index
from rtree.index import Property
import segmentation_models_pytorch as smp
import ttach as tta
import torch
from torch import Tensor, nn, autocast
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchgeo.samplers import GridGeoSampler
from torchgeo.datasets import BoundingBox, RasterDataset, GeoDataset
from torchgeo.datasets.utils import download_url, stack_samples
from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import _to_tuple
from torchgeo.trainers import SemanticSegmentationTask
from torchvision.transforms import Compose
from tqdm import tqdm
from ttach import SegmentationTTAWrapper

from utils.logger import get_logger
from utils.utils import _window_2D, get_device_ids, get_key_def, set_device

# Set the logging file
logging = get_logger(__name__)


class SingleBandItemEO(ItemEOExtension):
    def __init__(self, item: pystac.Item):
        super().__init__(item)
        self._assets_by_common_name = None

    @property
    def asset_by_common_name(self) -> Dict:
        """
        Adapted from: https://github.com/sat-utils/sat-stac/blob/40e60f225ac3ed9d89b45fe564c8c5f33fdee7e8/satstac/item.py#L75
        Get assets by common band name (only works for assets containing 1 band
        @param common_name:
        @return:
        """
        if self._assets_by_common_name is None:
            self._assets_by_common_name = {}
            for name, a_meta in self.item.assets.items():
                bands = []
                if 'eo:bands' in a_meta.extra_fields.keys():
                    bands = a_meta.extra_fields['eo:bands']
                if len(bands) == 1:
                    eo_band = bands[0]
                    if 'common_name' in eo_band.keys():
                        common_name = eo_band['common_name']
                        if not Band.band_range(common_name):  # Hacky but easiest way to validate common names
                            raise ValueError(f'Must be one of the accepted common names. Got "{common_name}".')
                        else:
                            self._assets_by_common_name[common_name] = {'href': a_meta.href, 'name': name}
        if not self._assets_by_common_name:
            raise ValueError(f"Common names for assets cannot be retrieved")
        return self._assets_by_common_name


class InferenceDataset(RasterDataset):
    def __init__(
            self,
            item_path: str,
            root: str = "data",
            outpath: Union[Path, str] = "pred.tif",
            crs: Optional[CRS] = None,
            res: Optional[float] = None,
            bands: Sequence[str] = [],
            transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
            cache: bool = False,
            download: bool = False,
            singleband_files: bool = True,
            pad: int = 256,  # TODO softcode pad mode (currently: reflect)
    ) -> None:
        """Initialize a new CCCOT Dataset instance.

        Arguments
        ---------
            item_path: TODO
            root: root directory where dataset can be found
            bands: band selection which must be a list of STAC Item common names from eo extension.
                        See: https://github.com/stac-extensions/eo/#common-band-names
            download: if True, download dataset and store it in the root directory.
        """
        self.item_url = item_path
        self.bands = bands
        self.root = Path(root)
        self.transforms = transforms
        self.separate_files = singleband_files
        self.cache = cache
        self.download = download
        self.pad = pad
        self.outpath = outpath

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))

        # Read Stac item from url
        if self.separate_files:
            self.item = SingleBandItemEO(pystac.Item.from_file(self.item_url))
        else:
            pass  # TODO: implement

        # Create band inventory (all available bands)
        self.all_bands = [band for band in self.item.asset_by_common_name.keys()]

        # Filter only desired bands
        self.bands_dict = {k: v for k, v in self.item.asset_by_common_name.items() if k in self.bands}

        # Make sure desired bands are subset of inventory
        if not set(self.bands).issubset(set(self.all_bands)):
            raise ValueError(f"Selected bands ({self.bands}) should be a subset of available bands ({self.all_bands})")

        # Open first asset with rasterio (for metadata: colormap, crs, resolution, etc.)
        self.first_asset = "/media/data/GDL_all_images/temp/VancouverP003_054230029070_01_P003_WV2-R.tif" # self.bands_dict[self.bands[0]]['href'] TODO: just for testing

        self.src = rasterio.open(self.first_asset)

        # See if file has a color map
        try:
            self.cmap = self.src.colormap(1)
        except ValueError:
            pass

        if crs is None:
            crs = self.src.crs
        if res is None:
            res = self.src.res[0]

        # to implement reprojection, see:
        # https://github.com/microsoft/torchgeo/blob/3f7e525fbd01dddd25804e7a1b7634269ead1760/torchgeo/datasets/geo.py#L361
        minx, miny, maxx, maxy = self.src.bounds

        # Get temporal information from STAC item
        self.date = self.item.item.datetime
        mint = maxt = self.date.timestamp()

        # Add paths to Rtree index
        coords = (minx, maxx, miny, maxy, mint, maxt)
        for cname in self.bands:
            asset_path = Path(self.bands_dict[cname]['href'])

            if self.download:
                out_name = self.root / asset_path.name
                download_url(str(asset_path), root=str(self.root), filename=str(out_name))
                self.bands_dict[cname]['href'] = out_name

        self.index.insert(0, coords, self.first_asset)
        self._crs = cast(CRS, crs)
        self.res = cast(float, res)

    def create_outraster(self):
        pred = np.zeros(self.src.shape, dtype=np.uint8)
        pred = pred[np.newaxis, :, :].astype(np.uint8)
        out_meta = self.src.profile
        out_meta.update({"driver": "GTiff",
                         "height": pred.shape[1],
                         "width": pred.shape[2],
                         "count": pred.shape[0],
                         "dtype": 'uint8',
                         'tiled': True,
                         'blockxsize': 256,
                         'blockysize': 256,
                         "compress": 'lzw'})
        with rasterio.open(self.outpath, 'w+', **out_meta) as dest:
            dest.write(pred)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = [hit.object for hit in hits]

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        with rasterio.Env(CPL_CURL_VERBOSE=False):  # TODO: turn off rasterio._env warnings
            if self.separate_files:
                data_list: List[Tensor] = []
                for band in getattr(self, "bands", self.all_bands):
                    band_filepaths = []
                    filepath = self.bands_dict[band]['href']  # hardcoded to this use case
                    band_filepaths.append(filepath)
                    data_list.append(self._merge_files(band_filepaths, query))
                data = torch.cat(data_list)  # type: ignore[attr-defined]
            else:
                # FIXME: implement multi-band Stac item: https://github.com/stac-extensions/eo/blob/main/examples/item.json
                data = self._merge_files(filepaths, query)
        data = data.float()

        key = "image" if self.is_image else "mask"
        sample = {key: data, "crs": self.crs, "bbox": query}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


# TODO remove this useless class from inference
class InferenceTask(SemanticSegmentationTask):
    def config_task(self) -> None:  # TODO: use Hydra
        """Configures the task based on kwargs parameters passed to the constructor."""
        if self.hparams["segmentation_model"] == "unet":
            self.model = smp.Unet(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
            )
        elif self.hparams["segmentation_model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
            )
        elif self.hparams["segmentation_model"] == "manet":
            self.model = smp.MAnet(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
            )
        else:
            raise ValueError(
                f"Model type '{self.hparams['segmentation_model']}' is not valid."
            )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function.
        Keyword Args:
            segmentation_model: Name of the segmentation model type to use
            encoder_name: Name of the encoder model backbone to use
            encoder_weights: None or "imagenet" to use imagenet pretrained weights in
                the encoder model
            in_channels: Number of channels in input image
            num_classes: Number of semantic classes to predict
        Raises:
            ValueError: if kwargs arguments are invalid
        """
        super().__init__(ignore_zeros=False)
        self.save_hyperparameters()  # creates `self.hparams` from kwargs

        self.config_task()

    def inference_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Inference step - outputs prediction for non-labeled imagery.
        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y_hat = self.forward(x)
        # y_hat_hard = y_hat.argmax(dim=1)

        return y_hat


# adapted from https://github.com/microsoft/torchgeo/blob/3f7e525fbd01dddd25804e7a1b7634269ead1760/torchgeo/datamodules/chesapeake.py
class InferenceDataModule(LightningDataModule):
    """LightningDataModule implementation for the InferenceDataset.
    Uses the random splits defined per state to partition tiles into train, val,
    and test sets.
    """
    def __init__(
        self,
        item_path: Union[str, Path],
        root_dir: Union[str, Path],
        outpath: Union[str, Path],
        bands: Sequence = ('red', 'green', 'blue'),
        patch_size: int = 256,
        stride: int = 256,
        pad: int = 256,
        batch_size: int = 1,
        num_workers: int = 0,
        download: bool = False,
        use_projection_units: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for InferenceDataset based Dataloader.
        Args:
            root_dir: The ``root`` arugment to pass to the ChesapeakeCVPR Dataset
                classes
            test_splits: The splits used to test the model, e.g. ["ny-test"]
            patch_size: The size of each patch in pixels
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            download: TODO
            use_projection_units : bool, optional
            Is `patch_size` in pixel units (default) or distance units?
        Raises:
            ValueError: if ``use_prior_labels`` is used with ``class_set==7``
        """
        super().__init__()  # type: ignore[no-untyped-call]

        self.item_path = item_path
        self.root_dir = root_dir
        self.outpath = outpath
        self.patch_size = patch_size
        self.stride = stride
        self.pad_size = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.use_projection_units = use_projection_units
        self.bands = bands

    # adapted from: https://github.com/microsoft/torchgeo/blob/3f7e525fbd01dddd25804e7a1b7634269ead1760/torchgeo/datamodules/chesapeake.py#L100
    def pad(
        self, size: int = 512, mode='constant'
    ) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Returns a function to perform a padding transform on a single sample.
        Args:
            size: size of padding to apply TODO
            image_value: value to pad image with
            mask_value: value to pad mask with
        Returns:
            function to perform padding
        """
        # use _pad from utils
        def _pad(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
            """ Pads img_arr """
            sample["image"] = F.pad(sample["image"], (size, size, size, size), mode=mode)
            return sample

        return _pad

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocesses a single sample.
        Args:
            sample: sample dictionary containing image
        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"] / 255.0
        sample["image"] = sample["image"].float()

        return sample

    def prepare_data(self) -> None:
        """Confirms that the dataset is downloaded on the local node.
        This method is called once per node, while :func:`setup` is called once per GPU.
        """
        InferenceDataset(
            root=self.root_dir,
            outpath=self.outpath,
            transforms=None,
            download=False,
        )

    def setup(self):
        """Instantiate the InferenceDataset.
        The splits should be done here vs. in :func:`__init__` per the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup.
        Args:
            stage: stage to set up
        self.test_dataset = ChesapeakeCVPR(
            self.root_dir,
            splits=self.test_splits,
            layers=self.layers,
            transforms=test_transforms,
            download=False,
            checksum=False,
        )
        """
        test_transforms = Compose([self.pad(self.pad_size, mode='reflect'),
                                   self.preprocess,
                                   ])

        self.inference_dataset = InferenceDataset(
            item_path=self.item_path,
            root=self.root_dir,
            outpath=self.outpath,
            bands=self.bands,
            transforms=test_transforms,
            download=self.download,
            pad=self.pad_size,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for inference.
        Returns:
            inference data loader
        """
        units = Units.PIXELS if not self.use_projection_units else Units.CRS
        self.sampler = GridGeoSamplerPlus(
            self.inference_dataset,
            size=self.patch_size,
            stride=self.stride,
            units=units,
        )
        return DataLoader(
            self.inference_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )

    def write_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for inference.
        Returns:
            inference data loader
        """
        self.write_dataset = self.inference_dataset.copy()

        sampler = GridGeoSampler(
            self.inference_dataset,
            size=self.patch_size,
            stride=self.patch_size,
        )
        return DataLoader(
            self.inference_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )


class GridGeoSamplerPlus(GridGeoSampler):
    def __init__(  # TODO: remove when issue #431 is solved
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        stride: Union[Tuple[float, float], float],
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` and ``stride`` arguments can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            stride: distance to skip between each patch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` and ``stride`` are in pixel or CRS units

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units
        """
        super().__init__(dataset=dataset, roi=roi, stride=stride, size=size)
        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)
            self.stride = (self.stride[0] * self.res, self.stride[1] * self.res)

        self.hits = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx > self.size[1]
                and bounds.maxy - bounds.miny > self.size[0]
            ):
                self.hits.append(hit)

        self.length: int = 0
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)

            rows = int((bounds.maxy - bounds.miny - self.size[0] + self.stride[0]) // self.stride[0]) + 1
            cols = int((bounds.maxx - bounds.minx - self.size[1] + self.stride[1]) // self.stride[1]) + 1
            self.length += rows * cols

    def __iter__(self) -> Iterator[BoundingBox]:  # TODO: remove when issue #431 is solved
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        # For each tile...
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)

            rows = int((bounds.maxy - bounds.miny - self.size[0] + self.stride[0]) // self.stride[0]) + 1
            cols = int((bounds.maxx - bounds.minx - self.size[1] + self.stride[1]) // self.stride[1]) + 1

            mint = bounds.mint
            maxt = bounds.maxt

            # For each row...
            for i in range(rows):
                miny = bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]
                if maxy > bounds.maxy:
                    maxy = bounds.maxy
                    miny = bounds.maxy - self.size[0]

                # For each column...
                for j in range(cols):
                    minx = bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]
                    if maxx > bounds.maxx:
                        maxx = bounds.maxx
                        minx = bounds.maxx - self.size[1]

                    yield BoundingBox(minx, maxx, miny, maxy, mint, maxt)

    def chip_indices_from_bbox(self, bounds, source):

        chip_minx, chip_maxx, chip_miny, chip_maxy, *_ = bounds
        try:
            samp_window = rasterio.windows.from_bounds(chip_minx, chip_maxy, chip_maxx, chip_miny, transform=source.transform)
        except rasterio.windows.WindowError:  # TODO how to deal with CRS units that don't go left->right, top->bottom
            samp_window = rasterio.windows.from_bounds(chip_minx, chip_miny, chip_maxx, chip_maxy,
                                                       transform=source.transform)
        left, bottom, right, top = samp_window.col_off, samp_window.row_off+np.ceil(samp_window.height), samp_window.col_off+np.ceil(samp_window.width), samp_window.row_off
        return [int(side) for side in (left, bottom, right, top)]


def auto_chip_size_finder(datamodule, device, model, tta_transforms, single_class_mode=False, max_used_ram: int = 95):
    """
    TODO
    @param datamodule:
    @param device:
    @param model:
    @param tta_transforms: test-time transforms as implemented run_eval_loop()
    @param single_class_mode:
    @param max_used_ram:
    @return:
    """
    chip_size_init = chip_size_trial = 256
    _tqdm = tqdm(range(32), desc=f"Finding chip size filling GPU to {max_used_ram} % or less")
    for trial in _tqdm:
        chip_size_trial += chip_size_init*(trial+1)
        _tqdm.set_postfix_str(f"Trying: {chip_size_trial}")
        datamodule.patch_size = chip_size_trial
        if chip_size_trial > min(datamodule.inference_dataset.src.shape):
            logging.info(f"Reached maximum chip size for image size. "
                         f"Chip size tuned to {chip_size_trial-chip_size_init}.")
            return chip_size_trial-chip_size_init
        window_spline_2d = create_spline_window(chip_size_trial).to(device)
        eval_gen2tune = run_eval_loop(
            model=model,
            dataloader=datamodule.predict_dataloader(),
            device=device,
            single_class_mode=single_class_mode,
            window_spline_2d=window_spline_2d,
            pad=datamodule.pad_size,
            tta_transforms=tta_transforms,
        )
        _ = next(eval_gen2tune)
        free, total = torch.cuda.mem_get_info(device)
        if (total-free)/total > max_used_ram/100:
            logging.info(f"Reached GPU RAM threshold of {int(max_used_ram)}%. Chip size tuned to {chip_size_trial}.")
            return chip_size_trial


def create_spline_window(chip_size, power=1):
    window_spline_2d = _window_2D(window_size=chip_size, power=power)
    window_spline_2d = torch.as_tensor(np.moveaxis(window_spline_2d, 2, 0), ).float()
    return window_spline_2d


def run_eval_loop(
    model: LightningModule,
    dataloader: Any,
    device: torch.device,
    single_class_mode: bool,
    window_spline_2d,
    pad,
    tta_transforms: Union[List, str] = "horizontal_flip",
    tta_merge_mode: str = 'max',
) -> Any:
    """Runs an adapted version of test loop without label data over a dataloader and returns prediction.
    Args:
        model: the model used for inference
        dataloader: the dataloader to get samples from
        device: the device to put data on
        single_class_mode: TODO
        window_spline_2d: TODO
        pad: TODO
        tta_transforms:
        tta_merge_mode: TODO
    Returns:
        the prediction for a dataloader batch
    """
    # initialize test time augmentation
    transforms_dict = {"horizontal_flip": tta.Compose([tta.HorizontalFlip()]), "d4": tta.aliases.d4_transform()}
    if tta_transforms not in transforms_dict.keys():
        raise ValueError(f"TTA transform {tta_transforms} is not implemented. Choose between {transforms_dict.keys()}")
    transforms = transforms_dict[tta_transforms]  # TODO: tune with optuna
    model = SegmentationTTAWrapper(model, transforms, merge_mode=tta_merge_mode)

    batch_output = {}
    for batch in tqdm(dataloader):
        batch_output['bbox'] = batch['bbox']
        batch_output['crs'] = batch['crs']
        inputs = batch["image"].to(device)
        with autocast(device_type=device.type):
            outputs = model(inputs)
        if single_class_mode:
            outputs = outputs.squeeze(dim=0)
        else:
            outputs = F.softmax(outputs, dim=1).squeeze(dim=0)
        outputs = outputs[:, pad:-pad, pad:-pad]
        outputs = torch.mul(outputs, window_spline_2d)
        if single_class_mode:
            outputs = torch.sigmoid(outputs)
        outputs = outputs.permute(1, 2, 0).cpu().numpy().astype('float16')
        batch_output['data'] = outputs
        yield batch_output


def main(params):
    """High-level pipeline.
    Runs a model checkpoint on non-labeled imagery and saves results to file.
    Args:
        params: configuration parameters
    """
    # Main params
    item_url = get_key_def('input_stac_item', params['inference'], expected_type=str)  #, to_path=True, validate_path_exists=True) TODO implement for url
    checkpoint = get_key_def('state_dict_path', params['inference'], expected_type=str, to_path=True, validate_path_exists=True)
    root = get_key_def('root_dir', params['inference'], default="data", to_path=True, validate_path_exists=True)
    # model_name = get_key_def('model_name', params['model'], expected_type=str).lower()  # TODO couple with model_choice.py
    download_data = get_key_def('download_data', params['inference'], default=False, expected_type=bool)
    raster_to_vec = get_key_def('ras2vec', params['inference'], default=True)
    post_proc = get_key_def('post_process', params['inference'], default=True)

    # Dataset params
    modalities = get_key_def('modalities', params['dataset'], default=("red", "blue", "green"), expected_type=Sequence)
    num_bands = len(modalities)
    num_classes = len(get_key_def('classes_dict', params['dataset']).keys())
    num_classes = num_classes + 1 #if num_classes > 1 else num_classes  # multiclass account for background TODO bug fix for old models

    # Hardware
    num_devices = get_key_def('gpu', params['inference'], default=1, expected_type=(int, bool))  # TODO implement >1
    max_used_ram = get_key_def('max_used_ram', params['inference'], default=50, expected_type=int)
    if not (0 <= max_used_ram <= 100):
        raise ValueError(f'\nMax used ram parameter should be a percentage. Got {max_used_ram}.')
    max_used_perc = get_key_def('max_used_perc', params['inference'], default=25, expected_type=int)
    # list of GPU devices that are available and unused. If no GPUs, returns empty dict
    gpu_devices_dict = get_device_ids(num_devices, max_used_ram_perc=max_used_ram, max_used_perc=max_used_perc)
    device = set_device(gpu_devices_dict=gpu_devices_dict)
    num_workers_default = len(gpu_devices_dict.keys()) * 4 if len(gpu_devices_dict.keys()) > 1 else 4
    num_workers = get_key_def('num_workers', params['inference'], default=num_workers_default, expected_type=int)

    # Sampling, batching and augmentations configuration
    chip_size = get_key_def('chunk_size', params['inference'], default=None, expected_type=int)
    auto_chip_size = True if not chip_size else False
    auto_cs_threshold = get_key_def('auto_chunk_size_threshold', params['inference'], default=95, expected_type=int)
    stride_default = int(chip_size / 2)
    stride = get_key_def('stride', params['inference'], default=stride_default, expected_type=int)
    pad = get_key_def('pad', params['inference'], default=16, expected_type=int)
    batch_size = num_devices  # for inference, batch size should be equal to number of GPU being used TODO implement bs>1
    # TODO implement with hydra for all possible ttach transforms
    tta_transforms = get_key_def('tta_transforms', params['inference'], default="horizontal_flip", expected_type=str)
    tta_merge_mode = get_key_def('tta_merge_mode', params['inference'], default="max", expected_type=str)

    # Create yaml to use pytorch lightning model management
    hparams = OmegaConf.create()
    hparams["segmentation_model"] = "manet"  # TODO temporary
    hparams["encoder_name"] = "resnext50_32x4d"
    hparams["encoder_weights"] = 'imagenet'
    hparams["in_channels"] = num_bands
    hparams["num_classes"] = num_classes
    single_class_mode = False #if hparams["num_classes"] > 2 else True TODO bug fix in GDL
    hparams["ignore_zeros"] = None

    with open('hparams.yaml', 'w') as fp:
        OmegaConf.save(config=hparams, f=fp)

    seed = 123
    seed_everything(seed)

    model = InferenceTask.load_from_checkpoint(checkpoint, hparams_file='hparams.yaml')
    model.freeze()
    model.eval()

    dm = InferenceDataModule(root_dir=root,
                             item_path=item_url,
                             outpath=root/f"{Path(item_url).stem}_pred.tif",
                             bands=modalities,
                             patch_size=chip_size,
                             stride=stride,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             download=download_data,
                             seed=seed,
                             pad=pad,
                             )  # TODO: test batch_size>1 for multi-gpu implementation
    dm.setup()

    model = model.to(device) #, dtype=torch.half) TODO test

    h, w = [side for side in dm.inference_dataset.src.shape]

    if auto_chip_size and device != "cpu":
        chip_size = auto_chip_size_finder(datamodule=dm,
                                          device=device,
                                          model=model,
                                          tta_transforms=tta_transforms,
                                          single_class_mode=single_class_mode,
                                          max_used_ram=auto_cs_threshold,
                                          )
        dm.patch_size = chip_size

    window_spline_2d = create_spline_window(chip_size).to(device)
    print(dm.patch_size)
    eval_gen = run_eval_loop(
        model=model,
        dataloader=dm.predict_dataloader(),
        device=device,
        single_class_mode=single_class_mode,
        window_spline_2d=window_spline_2d,
        pad=dm.pad_size,
        tta_transforms=tta_transforms,
        tta_merge_mode=tta_merge_mode,
    )

    tempfile = root / f"{Path(dm.inference_dataset.outpath).stem}.dat"
    fp = np.memmap(tempfile, dtype='float16', mode='w+', shape=(h, w, hparams["num_classes"]))
    for i, inference_prediction in enumerate(eval_gen):
        for bbox in inference_prediction['bbox']:
            col_min, *_, row_min = dm.sampler.chip_indices_from_bbox(bbox, dm.inference_dataset.src)
            right = col_min + chip_size if col_min + chip_size <= w else w
            bottom = row_min + chip_size if row_min + chip_size <= h else h
            fp[row_min:bottom, col_min:right, :] = \
                fp[row_min:bottom, col_min:right, :] + inference_prediction['data'][:bottom-row_min, :right-col_min]
    fp.flush()
    del fp

    fp = np.memmap(tempfile, dtype='float16', mode='r', shape=(h, w, hparams["num_classes"]))
    pred_img = fp.argmax(axis=-1).astype('uint8')
    pred_img = pred_img[np.newaxis, :, :].astype(np.uint8)
    dm.inference_dataset.create_outraster()  # Unknown bug when moved further down
    outpath = Path(dm.inference_dataset.outpath)
    meta = rasterio.open(outpath).meta
    with rasterio.open(outpath, 'w+', **meta) as dest:
        dest.write(pred_img)

    if raster_to_vec:
        out_vect = root / f"{outpath.stem}.gpkg"
        out_vect_temp = root / f"{outpath.stem}_temp.gpkg"
        logging.info(f"Vectorizing prediction to {out_vect}")
        commands = [f"qgis_process run grass7:r.to.vect -- input=/home/{str(outpath.name)} type=2 "
                    f"output=/home/{str(out_vect_temp.name)}",
                    f'ogr2ogr /home/{str(out_vect.name)} /home/{str(out_vect_temp.name)} -where "value" > 0']  # FIXME
        for command in commands:
            print(command)
            try:
                client = docker.from_env()
                qgis_pp_docker_img = [image for image in client.images.list() if "qgis_pp" in image.tags[0]][0]
                if not qgis_pp_docker_img:
                    logging.critical(f"Please docker pull latest image from https://hub.docker.com/r/remtav/qgis_pp")
                container = client.containers.run(
                    image=qgis_pp_docker_img,
                    command=command,
                    volumes={f'{str(outpath.parent)}': {'bind': '/home', 'mode': 'rw'}},
                    detach=True)
                exit_code = container.wait(timeout=1200)
                logging.info(exit_code)
                container.logs(stream=True)
            except DockerException as e:
                logging.info(f"Problem with docker client: {e}")

    logging.info('Single prediction done')


if __name__ == "__main__":  # serves as back up
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-stac-item",
        required=True,
        help="url or path to stac item pointing to imagery stored one band / file and implementing stac's eo extension",
        metavar="STAC")
    parser.add_argument(
        "--state-dict-path",
        required=True,
        help="path to the checkpoint file to test",
        metavar="CKPT")
    parser.add_argument(
        "--root-dir",
        help="root directory where dataset downloaded if desired and outputs will be written",
        metavar="DIR")
    args = parser.parse_args()
    params = {"inference": {"input_stac_item": args.input_stac_item, "state_dict_path": args.state_dict_path,
              "root_dir": args.root_dir},
              "dataset": {'classes_dict': {'Building': 1}}}  # TODO softcode?
    main(params)

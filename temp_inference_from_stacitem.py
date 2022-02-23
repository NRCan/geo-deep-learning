# Licensed under the MIT License.
# Adapted from: https://github.com/microsoft/torchgeo/blob/3f7e525fbd01dddd25804e7a1b7634269ead1760/evaluate.py

"""CCMEO model inference script."""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Sequence, List, cast, OrderedDict, Iterable

import numpy as np
import pystac
import rasterio
from omegaconf import OmegaConf
from pystac.extensions.eo import ItemEOExtension, Band
from pytorch_lightning import LightningDataModule, LightningModule, seed_everything
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from rtree import Index
from rtree.index import Property
import segmentation_models_pytorch as smp
import ttach as tta
import torch
from torch import Tensor, autocast
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchgeo.samplers import GridGeoSampler
from torchgeo.datasets import BoundingBox, RasterDataset
from torchgeo.datasets.utils import download_url, stack_samples, _list_dict_to_dict_list
from torchgeo.trainers import SemanticSegmentationTask
from torchvision.transforms import Compose
from tqdm import tqdm

from utils.utils import _window_2D


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
            root: str = 'data',
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
        self.item_url = Path(item_path)
        self.bands = bands
        self.root = Path(root)
        self.transforms = transforms
        self.separate_files = singleband_files
        self.cache = cache
        self.download = download
        self.pad = pad
        self.outpath = self.root / f"{self.item_url.stem}_pred.tif"

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))

        # Read Stac item from url
        if self.separate_files:
            self.item = SingleBandItemEO(pystac.Item.from_file(item_path))
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
        self.first_asset = "/media/data/GDL_all_images/temp/VancouverP003_054230029070_01_P003_WV2-R.tif" #self.bands_dict[self.bands[0]]['href']

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
                self.bands_dict[cname]['href'] = asset_path = out_name

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
        item_path: str,
        root_dir: str,
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
            self.root_dir,
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
            self.item_path,
            self.root_dir,
            bands=self.bands,
            transforms=test_transforms,
            download=self.download,
            pad=self.pad_size,
        )

        if not self.use_projection_units:  # TODO: refactor --> proj2pix_units() in Dataset?
            # convert pixel units to projection units
            self.patch_size = int(self.patch_size * self.inference_dataset.res)
            self.stride = int(self.stride * self.inference_dataset.res)
            self.pad_size = int(self.pad_size * self.inference_dataset.res)

    def predict_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for inference.
        Returns:
            inference data loader
        """
        self.sampler = GridGeoSamplerPlus(
            self.inference_dataset,
            size=self.patch_size,
            stride=self.stride,
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
    def chip_indices_from_bbox(self, bounds):

        chip_minx, chip_maxx, chip_miny, chip_maxy, *_ = bounds

        rows = int((self.roi.maxy - self.roi.miny - self.size[0]) // self.stride[0]) + 1
        cols = int((self.roi.maxx - self.roi.minx - self.size[1]) // self.stride[1]) + 1

        # For each row...
        for i in range(rows):
            miny = self.roi.miny + i * self.stride[0]
            maxy = miny + self.size[0]

            # For each column...
            for j in range(cols):
                minx = self.roi.minx + j * self.stride[1]
                maxx = minx + self.size[1]

                if (minx, maxx, miny, maxy) == (chip_minx, chip_maxx, chip_miny, chip_maxy):
                    return rows-i, j


def create_outraster():
    path = "/media/data/GDL_all_images/VancouverP003_054230029070_01_P003_WV2-R.tif"
    src = rasterio.open(path)
    outpath = "/media/data/GDL_all_images/test.tif"

    pred = np.zeros(src.shape, dtype=np.uint8)
    pred = pred[np.newaxis, :, :].astype(np.uint8)
    out_meta = src.profile
    out_meta.update({"driver": "GTiff",
                     "height": pred.shape[1],
                     "width": pred.shape[2],
                     "count": pred.shape[0],
                     "dtype": 'uint8',
                     'tiled': True,
                     'blockxsize': 256,
                     'blockysize': 256,
                     "compress": 'lzw'})
    with rasterio.open(outpath, 'w+', **out_meta) as dest:
        dest.write(pred)


def run_eval_loop(
    model: LightningModule,
    dataloader: Any,
    device: torch.device,
    single_class_mode: bool,
    window_spline_2d,  # type: ignore[name-defined]
) -> Any:
    """Runs an adapted version of test loop without label data over a dataloader and returns prediction.
    Args:
        model: the model used for inference
        dataloader: the dataloader to get samples from
        device: the device to put data on
        single_class_mode: TODO
        window_sploine_2d: TODO
    Returns:
        the prediction for a dataloader batch
    """
    # Reverse calculate pad and distance between samples
    pad = window_spline_2d.shape[-1]
    dist_samples = int(round(pad/2 * (1 - 1.0 / 2.0)))
    # initialize test time augmentation
    transforms = tta.Compose([tta.HorizontalFlip(), ])

    batch_output = {}
    for batch in tqdm(dataloader):
        batch_output['bbox'] = batch['bbox']
        batch_output['crs'] = batch['crs']
        inputs = batch["image"].to(device)
        with torch.inference_mode():
            output_lst = []
            for transformer in transforms:
                # augment inputs
                augmented_input = transformer.augment_image(inputs)
                with autocast(device_type=device.type):
                    augmented_output = model(augmented_input)
                if isinstance(augmented_output, OrderedDict) and 'out' in augmented_output.keys():
                    augmented_output = augmented_output['out']
                logging.debug(f'Shape of augmented output: {augmented_output.shape}')
                # reverse augmentation for outputs
                deaugmented_output = transformer.deaugment_mask(augmented_output)
                if single_class_mode:
                    deaugmented_output = deaugmented_output.squeeze(dim=0)
                else:
                    deaugmented_output = F.softmax(deaugmented_output, dim=1).squeeze(dim=0)
                output_lst.append(deaugmented_output)
            outputs = torch.stack(output_lst)
            outputs = torch.mul(outputs, window_spline_2d)
            outputs, _ = torch.max(outputs, dim=0)
            if single_class_mode:
                outputs = torch.sigmoid(outputs)
            outputs = outputs.permute(1, 2, 0)
            outputs = outputs.reshape(pad, pad, outputs.shape[-1]).cpu().numpy().astype('float16')
            outputs = outputs[dist_samples:-dist_samples, dist_samples:-dist_samples, :]
            batch_output['data'] = outputs
        yield batch_output


def main():
    """High-level pipeline.
    Runs a model checkpoint on non-labeled imagery and saves results to file.
    Args:
        args: command-line arguments
    """
    root = '/media/data/GDL_all_images/temp'
    #item_url = "https://datacube-stage.services.geo.ca/api/collections/worldview-2-ortho-pansharp/items/AB10-056102820020_01_P001-WV02"
    item_url = "https://datacube-stage.services.geo.ca/api/collections/worldview-2-ortho-pansharp/items/VancouverP003_054230029070_01_P003_WV2"
    checkpoint = "/media/data/operationalization/pl_manet_pretrained_bds3_cls1.pth.tar"
    chip_size = 512
    pad = 256
    stride = int(chip_size / 2)
    batch_size = 1  # FIXME not working with batch size of more than 1
    download_data = True
    gpu_id = 0
    num_workers = 0
    seed = 123

    # Dataset params
    modalities = ("red", "green", "blue")  # Select bands from STAC EO's common_names
    classes = {1: "buildings"}

    # Model params
    hparams = OmegaConf.create()
    hparams["segmentation_model"] = "manet"
    hparams["encoder_name"] = "resnext50_32x4d"
    hparams["encoder_weights"] = 'imagenet'
    hparams["in_channels"] = len(modalities)
    hparams["num_classes"] = len(classes.keys())+1
    single_class_mode = False #if hparams["num_classes"] > 2 else True TODO bug fix in GDL
    hparams["ignore_zeros"] = None

    with open('hparams.yaml', 'w') as fp:
        OmegaConf.save(config=hparams, f=fp)

    seed_everything(seed)

    # Loads the saved model from checkpoint based on the `args.task` name that was
    # passed as input
    model = InferenceTask.load_from_checkpoint(checkpoint, hparams_file='hparams.yaml')
    model.freeze()
    model.eval()

    dm = InferenceDataModule(root_dir=root,
                             item_path=item_url,
                             bands=modalities,
                             patch_size=chip_size,
                             stride=stride,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             download=download_data,
                             seed=seed,
                             )  # TODO: test batch_size>1 for multi-gpu implementation
    dm.setup()
    dm.inference_dataset.create_outraster()

    device = torch.device("cuda:%d" % (gpu_id))  # type: ignore[attr-defined]
    model = model.to(device)

    # Main array configuration
    tempfile = "data/test.dat"

    h_padded, w_padded = [side + chip_size*2 for side in dm.inference_dataset.src.shape]
    dist_samples = int(round(chip_size * (1 - 1.0 / 2.0)))

    # construct window for smoothing
    WINDOW_SPLINE_2D = _window_2D(window_size=chip_size+2*pad, power=2.0)
    WINDOW_SPLINE_2D = torch.as_tensor(np.moveaxis(WINDOW_SPLINE_2D, 2, 0), ).type(torch.float)
    WINDOW_SPLINE_2D = WINDOW_SPLINE_2D.to(device)

    fp = np.memmap(tempfile, dtype='float16', mode='w+', shape=(h_padded, w_padded, hparams["num_classes"]))

    eval_gen = run_eval_loop(
        model, dm.predict_dataloader(),
        device,
        single_class_mode=single_class_mode,
        window_spline_2d=WINDOW_SPLINE_2D
    )

    for inference_prediction in eval_gen:
        for bbox in inference_prediction['bbox']:
            #print(bbox)
            row, col = dm.sampler.chip_indices_from_bbox(bbox)
            row, col = row*stride, col*stride
            #print(row, col)
            fp[row:row + chip_size, col:col + chip_size, :] = \
                fp[row:row + chip_size, col:col + chip_size, :] + inference_prediction['data']
    fp.flush()
    del fp

    fp = np.memmap(tempfile, dtype='float16', mode='r', shape=(h_padded, w_padded, hparams["num_classes"]))
    pred_img = np.zeros((h_padded, w_padded), dtype=np.uint8)
    arr1 = fp / (2 ** 2)
    arr1 = arr1.argmax(axis=-1).astype('uint8')
    pred_img = arr1
    pred_img = pred_img[pad:h_padded - (chip_size+pad), :w_padded - (chip_size+2*pad)]  # TODO CRS units messes with rows and cols
    pred_img = pred_img[np.newaxis, :, :].astype(np.uint8)
    outpath = dm.inference_dataset.outpath
    meta = rasterio.open(outpath).meta
    with rasterio.open(outpath, 'w+', **meta) as dest:
        dest.write(pred_img)


    print('Single prediction done')


if __name__ == "__main__":
    main()

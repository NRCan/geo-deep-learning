from pathlib import Path
from typing import Union, Sequence, Any, Callable, Dict, Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from skimage import exposure
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from torchgeo.samplers import Units, GridGeoSampler
from torchvision.transforms import Compose

from inference.GridGeoSamplerPlus import GridGeoSamplerPlus
from inference.InferenceDataset import InferenceDataset


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
        save_heatmap: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for InferenceDataset based Dataloader.
        Args:
            root_dir: The ``root`` argugment to pass to the InferenceDataset class
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
        self.save_heatmap = save_heatmap

    # adapted from: https://github.com/microsoft/torchgeo/blob/3f7e525fbd01dddd25804e7a1b7634269ead1760/torchgeo/datamodules/chesapeake.py#L100
    @staticmethod
    def pad(
        size: int = 512, mode='constant'
    ) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Returns a function to perform a padding transform on a single sample.
        Args:
            size: size of padding to apply
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

    @staticmethod
    def enhance(sample: Dict[str, Any], clip_limit=0.1) -> Dict[str, Any]:
        """
        Clip histogram by clip_limit # FIXME apply to entire image before inference? apply according to class?
        @param sample: sample dictionary containing image
        @param clip_limit:
        @return:
        """
        sample['image'] = np.moveaxis(sample["image"].numpy().astype(np.uint8), 0, -1)  # send channels last
        img_adapteq = []
        for band in range(sample['image'].shape[-1]):
            out_band = exposure.equalize_adapthist(sample["image"][..., band], clip_limit=clip_limit)
            out_band = (out_band*255).astype(int)
            img_adapteq.append(out_band)
        out_stacked = np.stack(img_adapteq, axis=-1)
        sample["image"] = torch.from_numpy(np.moveaxis(out_stacked, -1, 0))
        return sample

    @staticmethod
    def preprocess(sample: Dict[str, Any]) -> Dict[str, Any]:
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
            item_path=self.item_path,
            root=self.root_dir,
            outpath=self.outpath,
            bands=self.bands,
            transforms=None,
            download=self.download,
            pad=self.pad_size,
        )

    def setup(self, stage: Optional[str] = None):
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
                                   self.enhance,
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
            shuffle=False,
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
            shuffle=False,
        )

    def postprocess(self):
        pass  # TODO: move some/all post-processing operations to this method
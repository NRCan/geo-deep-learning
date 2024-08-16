import abc
import os
from pathlib import Path
from typing import Union, Optional, Sequence, Callable, Dict, Any, cast, List, Tuple

import pandas
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt
import numpy as np
from pandas.io.common import is_url
import pystac
from pystac.extensions.eo import ItemEOExtension, Band
import rasterio as rio
from rtree import Index
from rtree.index import Property
import torch
from torchgeo.datasets import RasterDataset, BoundingBox
from torchvision.datasets.utils import download_url

from matplotlib.figure import Figure
from rasterio.crs import CRS
from rasterio.transform import Affine
from torch import Tensor

from torchgeo.datasets import NonGeoDataset


class BlueSkyNonGeo(NonGeoDataset, abc.ABC):
    """
    The bluesky datasets are a set of datasets that contain mostly 4-class labels
    (forest, waterbody, road, building) mapped over high-resolution satellite imagery
    obtained from a variety of sensors such as Worldview-2, Worldview-3, Worldview-4,
    GeoEye, Quickbird and aerial imagery.
    """

    @property
    @abc.abstractmethod
    def dataset_id(self) -> str:
        """Dataset ID."""

    @property
    @abc.abstractmethod
    def imagery(self) -> Dict[str, str]:
        """Mapping of image identifier and filename."""

    def __init__(
        self,
        root: str,
        image: str,
        collections: List[str] = [],
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        checksum: bool = False,
        splits=["trn"]
    ) -> None:
        """Initialize a new CCMEO Dataset instance.

        Args:
            root: root directory where dataset can be found
            image: image selection
            collections: collection selection
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: if True, download dataset and store it in the root directory.
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing
        """
        self.root = Path(root)
        self.image = image  # For testing
        self.splits = splits

        self.files = self._load_files(root)

    def _load_files(self, root: str) -> List[Dict[str, Path]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset

        Returns:
            list of dicts containing paths for each pair of image and label
        """
        # TODO: glob or list?
        files = []
        # for collection in self.collections:
        for split in self.splits:
            rows = pandas.read_csv(split, sep=';', header=None)
            for row in rows.values:
                imgpath, lbl_path = row[:2]
                imgpath, lbl_path = self.root / imgpath, self.root / lbl_path
                if not imgpath.is_file():
                    raise FileNotFoundError(imgpath)
                if not lbl_path.is_file():
                    raise FileNotFoundError(lbl_path)
                files.append({"image_path": imgpath, "label_path": lbl_path})
                
        return files

    def _load_image(self, path: Union[str, Path]) -> Tuple[Tensor, Affine, CRS]:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with rio.open(path) as img:
            array = img.read().astype(np.int32)
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            return tensor, img.transform, img.crs

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        img, tfm, raster_crs = self._load_image(files["image_path"])
        # h, w = img.shape[1:]
        mask, *_ = self._load_image(files["label_path"])
        mask[mask == 255] = 0  # TODO: make ignore_index work
        mask = mask.squeeze()

        if not img.shape[-2:] == mask.shape[-2:]:
            raise ValueError(f"Mismatch between image chip shape ({img.shape}) and mask chip shape ({mask.shape})")
        sample = {"image": img,
                  "mask": mask,
                  "image_path": str(files["image_path"]),
                  "label_path": str(files["label_path"]),
                  "aoi_id": files["image_path"].parent.parent.stem}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
from pathlib import Path
from typing import Union, Optional, Callable, Dict, Any, cast

import torch
from rasterio.crs import CRS
from rasterio.windows import from_bounds
from rtree import Index
from rtree.index import Property
from torchgeo.datasets import RasterDataset, BoundingBox

from dataset.aoi import AOI

# Set the logging file
from utils.logger import get_logger

logging = get_logger(__name__)


class InferenceDataset(RasterDataset):
    def __init__(
            self,
            aoi: AOI,
            outpath: Union[Path, str] = "pred.tif",
            transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
            pad: int = 256,
    ) -> None:
        """Initialize a new CCCOT Dataset instance.

        @param aoi:
            AOI instance
        @param outpath:
            path to desired output
        @param transforms:
            Tranforms to apply to raw chip before feeding it to model
        @param pad:
            padding to apply to each chip
        """
        self.aoi = aoi
        self.transforms = transforms
        self.pad = pad
        self.outpath = outpath

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))

        # to implement reprojection, see:
        # https://github.com/microsoft/torchgeo/blob/3f7e525fbd01dddd25804e7a1b7634269ead1760/torchgeo/datasets/geo.py#L361
        minx, miny, maxx, maxy = self.aoi.raster.bounds

        # Get temporal information from STAC item
        if self.aoi.raster_stac_item is not None:
            self.date = self.aoi.raster_stac_item.item.datetime
            mint = maxt = self.date.timestamp()
        else:
            self.date = None
            mint = maxt = 0

        # Add paths to Rtree index
        coords = (minx, maxx, miny, maxy, mint, maxt)

        self._crs = cast(CRS, self.aoi.raster.crs)
        self.res = cast(float, self.aoi.raster.res[0])

        self.aoi.close_raster()  # make pickleable for rtree
        self.aoi.raster = None

        self.index.insert(0, coords, self.aoi)

        self.aoi.raster_open()

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
        aoi = [hit.object for hit in hits]

        if not aoi:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )
        elif len(aoi) > 1:
            raise ValueError(f"Dataset should reference only one AOI at the time.")

        aoi = aoi[0]

        aoi.raster_open()

        # TODO: turn off external logs (ex.: rasterio._env)
        # https://stackoverflow.com/questions/35325042/python-logging-disable-logging-from-imported-modules
        bounds = (query.minx, query.miny, query.maxx, query.maxy)
        out_width = round((query.maxx - query.minx) / self.res)
        out_height = round((query.maxy - query.miny) / self.res)
        out_shape = (aoi.raster.count, out_height, out_width)
        dest = aoi.raster.read(
            out_shape=out_shape, window=from_bounds(*bounds, aoi.raster.transform)
        )
        data = torch.tensor(dest)
        data = data.float()

        key = "image" if self.is_image else "mask"
        sample = {key: data, "crs": self.crs, "bbox": query}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

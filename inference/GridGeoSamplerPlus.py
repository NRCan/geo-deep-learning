from typing import Union, Tuple, Optional, Iterator

import numpy as np
import rasterio
from torchgeo.datasets import GeoDataset, BoundingBox
from torchgeo.samplers import GridGeoSampler, Units
from torchgeo.samplers.utils import _to_tuple


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
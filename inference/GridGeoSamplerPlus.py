import warnings
from typing import Union, Tuple, Optional, Iterator

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

            for i in range(rows):
                miny = bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]
                if maxy > bounds.maxy:
                    last_stride_y = self.stride[0] - (miny - (bounds.maxy - self.size[0]))
                    maxy = bounds.maxy
                    miny = bounds.maxy - self.size[0]
                    warnings.warn(
                        f"Max y coordinate of bounding box reaches passed y bounds of source tile. "
                        f"Bounding box will be moved to set max y at source tile's max y. Stride will be adjusted "
                        f"from {self.stride[0]:.2f} to {last_stride_y:.2f}"
                    )

                # For each column...
                for j in range(cols):
                    minx = bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]
                    if maxx > bounds.maxx:
                        last_stride_x = self.stride[1] - (minx - (bounds.maxx - self.size[1]))
                        maxx = bounds.maxx
                        minx = bounds.maxx - self.size[1]
                        warnings.warn(
                            f"Max x coordinate of bounding box reaches passed x bounds of source tile. "
                            f"Bounding box will be moved to set max x at source tile's max x. Stride will be adjusted "
                            f"from {self.stride[1]:.2f} to {last_stride_x:.2f}"
                        )

                    yield BoundingBox(minx, maxx, miny, maxy, mint, maxt)
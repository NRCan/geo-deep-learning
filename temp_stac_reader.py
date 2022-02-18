import sys
import warnings
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Sequence, List, cast

import pystac
import rasterio
from pystac import Collection
from pystac.extensions.eo import EOExtension, ItemEOExtension, Band
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from rio_tiler.io import COGReader, STACReader
from rio_tiler.models import ImageData
from rio_tiler.utils import render
from rtree import Index
from rtree.index import Property
from solaris.tile.raster_tile import RasterTiler
from solaris.utils import tile
from solaris.utils.geo import split_geom
import torch
from torch import Tensor
from torchgeo.samplers import GridGeoSampler
from torchgeo.datasets import GeoDataset, BoundingBox, SpaceNet1, CDL, ChesapeakeCVPR, VisionDataset, RasterDataset
from torchgeo.datasets.utils import download_url, disambiguate_timestamp
from tqdm import tqdm

test_asset_dict = {
    'blue': {'href': "http://datacube-stage-data-public.s3.ca-central-1.amazonaws.com/store/imagery/optical/worldview-3-ortho-pansharp/QC22_055128844110_01_WV3-B.tif", 'name': 'B'},
    'red': {'href': "http://datacube-stage-data-public.s3.ca-central-1.amazonaws.com/store/imagery/optical/worldview-3-ortho-pansharp/QC22_055128844110_01_WV3-R.tif", 'name': 'R'},
    'green': {'href': "http://datacube-stage-data-public.s3.ca-central-1.amazonaws.com/store/imagery/optical/worldview-3-ortho-pansharp/QC22_055128844110_01_WV3-G.tif", 'name': 'G'},
    'nir': {'href': "http://datacube-stage-data-public.s3.ca-central-1.amazonaws.com/store/imagery/optical/worldview-3-ortho-pansharp/QC22_055128844110_01_WV3-N.tif", 'name': 'N'}}


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


class InferenceDataset(VisionDataset):
    def __init__(
            self,
            item_url: str,
            chip_size: int = 512,
            resize_factor: int = 1,
            use_projection_units=False,
            root: str = 'data',
            modalities: Sequence = ("red", "blue", "green"),
            download: bool = False,
            debug: bool = False,
    ) -> None:
        """Initialize a new CCCOT Dataset instance.

        Arguments
        ---------
            item_url: TODO
            chip_size : `tuple` of `int`s
                The size of the input chips in ``(y, x)`` coordinates. By default,
                this is in pixel units; this can be changed to metric units using the
                `use_metric_size` argument.
                "Chip" is preferred to "tile" with respect to torchgeo's glossary
                https://torchgeo.readthedocs.io/en/latest/user/glossary.html#term-chip
            resize_factor: TODO
            use_projection_units : bool, optional
                Is `chip_size` in pixel units (default) or distance units? To set to distance units
                use ``use_projection_units=True``. If False, resolution must be supplied.
            root: root directory where dataset can be found
            modalities: band selection which must be a list of STAC Item common names from eo extension.
                        See: https://github.com/stac-extensions/eo/#common-band-names
            download: if True, download dataset and store it in the root directory.
        """
        self.item_url = item_url
        self.modalities = modalities
        self.root = root
        self.download = download
        self.debug = debug

        self.item = SingleBandItemEO(pystac.Item.from_file(item_url))
        self.assets_dict = self.item.asset_by_common_name  # test_asset_dict
        self.assets_name = tuple(self.assets_dict[band]['name'] for band in self.modalities)

        self.src = rasterio.open(list(self.assets_dict.values())[0]['href'])
        self.resizing_factor = resize_factor
        self.dest_chip_size = chip_size
        self.src_chip_size = self.get_src_tile_size()
        self.use_projection_units = use_projection_units
        self.chip_bounds = split_geom(
            geometry=list(self.src.bounds),
            tile_size=(self.src_chip_size, self.src_chip_size),
            resolution=(self.src.transform[0], -self.src.transform[4]),
            use_projection_units=self.use_projection_units,
            src_img=self.src
        )

        if self.download:
            for val in self.assets_dict.items():
                download_url(val['href'], root=self.root)

    def get_src_tile_size(self):
        """
        Sets outputs dimension of source tile if resizing, given destination size and resizing factor
        @param dest_tile_size: (int) Size of tile that is expected as output
        @param resize_factor: (float) Resize factor to apply to source imagery before outputting tiles
        @return: (int) Source tile size
        """
        if self.resizing_factor:
            return int(self.dest_chip_size / self.resizing_factor)
        else:
            return self.dest_chip_size

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            imagery at that index

        Raises:
            IndexError: if index is out of range of the dataset
        """
        bbox = self.chip_bounds[index]
        with STACReader(self.item_url) as stac:
            img = stac.part(bbox, bounds_crs=self.src.crs, assets=self.assets_name)
            if self.debug:
                with open(f'test{index}.tif', 'wb') as f:
                    f.write(img.render(img_format="GTIFF"))
        return img

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            length of the dataset
        """
        return len(self.chip_bounds)


class TGInferenceDataset(RasterDataset):
    def __init__(
            self,
            item_url: str,
            root: str = 'data',
            crs: Optional[CRS] = None,
            res: Optional[float] = None,
            bands: Sequence[str] = [],
            transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
            cache: bool = False,
            download: bool = False,
            chip_size: int = 512,
            resize_factor: int = 1,
            use_projection_units=False,
            debug: bool = False,
    ) -> None:
        """Initialize a new CCCOT Dataset instance.

        Arguments
        ---------
            item_url: TODO
            chip_size : `tuple` of `int`s
                The size of the input chips in ``(y, x)`` coordinates. By default,
                this is in pixel units; this can be changed to metric units using the
                `use_metric_size` argument.
                "Chip" is preferred to "tile" with respect to torchgeo's glossary
                https://torchgeo.readthedocs.io/en/latest/user/glossary.html#term-chip
            resize_factor: TODO
            use_projection_units : bool, optional
                Is `chip_size` in pixel units (default) or distance units? To set to distance units
                use ``use_projection_units=True``. If False, resolution must be supplied.
            root: root directory where dataset can be found
            bands: band selection which must be a list of STAC Item common names from eo extension.
                        See: https://github.com/stac-extensions/eo/#common-band-names
            download: if True, download dataset and store it in the root directory.
        """
        self.item_url = item_url
        self.bands = bands
        self.root = root
        self.transforms = transforms
        self.separate_files = True
        self.cache = cache
        self.download = download
        self.debug = debug

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))

        self.item = SingleBandItemEO(pystac.Item.from_file(item_url))
        self.all_bands = [band for band in self.item.asset_by_common_name.keys()]
        if not set(self.bands).issubset(set(self.all_bands)):
            raise ValueError(f"Selected bands ({self.bands}) should be a subset of available bands ({self.all_bands})")

        self.first_asset = self.item.asset_by_common_name[self.bands[0]]['href']

        self.src = rasterio.open(self.first_asset)
        self.resizing_factor = resize_factor
        self.dest_chip_size = chip_size
        self.src_chip_size = self.get_src_tile_size()
        self.use_projection_units = use_projection_units
        self.chip_bounds = split_geom(
            geometry=list(self.src.bounds),
            tile_size=(self.src_chip_size, self.src_chip_size),
            resolution=(self.src.transform[0], -self.src.transform[4]),
            use_projection_units=self.use_projection_units,
            src_img=self.src
        )

        # See if file has a color map
        try:
            self.cmap = self.src.colormap(1)
        except ValueError:
            pass

        if crs is None:
            crs = self.src.crs
        if res is None:
            res = self.src.res[0]

        with WarpedVRT(self.src, crs=crs) as vrt:
            minx, miny, maxx, maxy = vrt.bounds

        self.date = self.item.item.datetime
        mint = maxt = self.date.timestamp()

        coords = (minx, maxx, miny, maxy, mint, maxt)
        for i, cname in enumerate(self.bands):
            self.index.insert(i, coords, self.item.asset_by_common_name[cname]['href'])

        if self.download:
            for _, val in self.assets_dict.items():
                download_url(val['href'], root=self.root)

        self._crs = cast(CRS, crs)
        self.res = cast(float, res)

    def get_src_tile_size(self):
        """
        Sets outputs dimension of source tile if resizing, given destination size and resizing factor
        @param dest_tile_size: (int) Size of tile that is expected as output
        @param resize_factor: (float) Resize factor to apply to source imagery before outputting tiles
        @return: (int) Source tile size
        """
        if self.resizing_factor:
            return int(self.dest_chip_size / self.resizing_factor)
        else:
            return self.dest_chip_size

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
                filepath = self.item.asset_by_common_name[band]['href']  # hardcoded to this use case
                band_filepaths.append(filepath)
                data_list.append(self._merge_files(band_filepaths, query))
            data = torch.cat(data_list)  # type: ignore[attr-defined]
        else:
            data = self._merge_files(filepaths, query)

        key = "image" if self.is_image else "mask"
        sample = {key: data, "crs": self.crs, "bbox": query}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


collection = "https://datacube-stage.services.geo.ca/api/collections/geoeye-1-ortho-pansharp"

item_url = "https://datacube-stage.services.geo.ca/api/collections/worldview-2-ortho-pansharp/items/VancouverP003_054230029070_01_P003_WV2"
#item_url = "/media/data/GDL_all_images/QC22.json"
item = pystac.Item.from_file(item_url)
#eoitem = EOExtension.ext(item)
# ...

#item_com_names = [band.common_name for band in item]

#eoitem = EOExtension.ext(item)
#eoitem_com_names = [band.common_name for band in eoitem.bands]

# eoitem = SingleBandItemEO(item)

inf_dataset = TGInferenceDataset(item_url, bands=("red", "green", "blue", "nir"))
start1 = time.time()
for i, bounds in tqdm(enumerate(inf_dataset.chip_bounds)):
    minx, miny, maxx, maxy = bounds
    bounds = BoundingBox(minx, maxx, miny, maxy, mint=inf_dataset.bounds.mint, maxt=inf_dataset.bounds.maxt)
    result = inf_dataset[bounds]
    print(result['image'].shape)
    if i==20:
        break
end1 = time.time()
print(f"Time for execution of program: {round(end1-start1, 10)}")


inf_dataset = InferenceDataset(item_url, modalities=("red", "green", "blue", "nir"))

start2 = time.time()
for i, inference_chip in tqdm(enumerate(inf_dataset)):
    print(inference_chip.data.shape)
    if i==20:
        break
end2 = time.time()
print(f"Time for execution of program: {round(end2-start2, 10)}")

#my_asset = eoitem.asset_by_common_name("blue")
#tg_dataset = SpaceNet1(root="/media/data/spacenet", image="8band", download=True, api_key="2f56530ae7b25f7a5af1d0347bf0a2c73d8862f0ae3e061d3fcaf3ed778aaded")
#cdl_dataset = CDL(root="/media/data/CDL", download=True, cache=True)
cp_dataset = ChesapeakeCVPR(root="/media/data/Chesapeake", download=True, cache=True)
#print(tg_dataset)

#item3 = Item.open(item_url)
#test = item3.assets_by_common_name
#print(test)




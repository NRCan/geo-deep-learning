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
from torch.utils.data import DataLoader
from torchgeo.samplers import GridGeoSampler
from torchgeo.datasets import GeoDataset, BoundingBox, SpaceNet1, CDL, ChesapeakeCVPR, VisionDataset, RasterDataset, \
    ChesapeakeNY
from torchgeo.datasets.utils import download_url, disambiguate_timestamp, stack_samples
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


class InferenceDataset(RasterDataset):
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
    ) -> None:
        """Initialize a new CCCOT Dataset instance.

        Arguments
        ---------
            item_url: TODO
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

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))

        self.item = SingleBandItemEO(pystac.Item.from_file(item_url))
        self.all_bands = [band for band in self.item.asset_by_common_name.keys()]
        self.bands_dict = {k: v for k, v in self.item.asset_by_common_name.items() if k in self.bands}
        if not set(self.bands).issubset(set(self.all_bands)):
            raise ValueError(f"Selected bands ({self.bands}) should be a subset of available bands ({self.all_bands})")

        self.first_asset = self.bands_dict[self.bands[0]]['href']

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

        with WarpedVRT(self.src, crs=crs) as vrt:
            minx, miny, maxx, maxy = vrt.bounds

        self.date = self.item.item.datetime
        mint = maxt = self.date.timestamp()

        coords = (minx, maxx, miny, maxy, mint, maxt)
        for i, cname in enumerate(self.bands):
            asset_url = self.bands_dict[cname]['href']
            if self.download:
                out_name = str(Path(self.root) / Path(asset_url).name)
                download_url(asset_url, root=self.root, filename=out_name)
                self.bands_dict[cname]['href'] = out_name

            self.index.insert(i, coords, )

        self._crs = cast(CRS, crs)
        self.res = cast(float, res)

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
            data = self._merge_files(filepaths, query)

        key = "image" if self.is_image else "mask"
        sample = {key: data, "crs": self.crs, "bbox": query}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


def get_src_tile_size(dest_chip_size, resizing_factor=1):
    """
    Sets outputs dimension of source tile if resizing, given destination size and resizing factor
    @param dest_chip_size: (int) Size of tile that is expected as output
    @param resizing_factor: (float) Resize factor to apply to source imagery before outputting tiles
    @return: (int) Source tile size
    """
    return int(dest_chip_size / resizing_factor)


item_url = "https://datacube-stage.services.geo.ca/api/collections/worldview-2-ortho-pansharp/items/VancouverP003_054230029070_01_P003_WV2"
chip_size = 512
stride = 128
resize_factor = 1

resizing_factor = resize_factor
src_chip_size = get_src_tile_size(chip_size, resizing_factor)

dataset = InferenceDataset(item_url, root='/media/data/GDL_all_images', bands=("red", "green", "blue", "nir")) #, download=True)
sampler = GridGeoSampler(dataset, size=chip_size, stride=chip_size)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=stack_samples)

for data in dataloader:
    break

# for testing purposes only
chip_bounds = split_geom(
    geometry=list(dataset.src.bounds),
    tile_size=(chip_size, chip_size),
    resolution=(dataset.src.transform[0], -dataset.src.transform[4]),
    use_projection_units=False,
    src_img=dataset.src
)

start1 = time.time()
for i, bounds in tqdm(enumerate(chip_bounds)):
    minx, miny, maxx, maxy = bounds
    bounds = BoundingBox(minx, maxx, miny, maxy, mint=dataset.bounds.mint, maxt=dataset.bounds.maxt)
    result = dataset[bounds]
    print(result['image'].shape)
    if i==20:
        break
end1 = time.time()
print(f"Time for execution of program: {round(end1-start1, 10)}")

cp_datasetNY = ChesapeakeNY(root="/media/data/Chesapeake/NY", download=True, cache=True)
cp_dataset = ChesapeakeCVPR(root="/media/data/Chesapeake", download=True, cache=True)
#cdl_dataset = CDL(root="/media/data/CDL", download=True, cache=True)
#tg_dataset = SpaceNet1(root="/media/data/spacenet", image="8band", download=True, api_key="2f56530ae7b25f7a5af1d0347bf0a2c73d8862f0ae3e061d3fcaf3ed778aaded")
#print(tg_dataset)





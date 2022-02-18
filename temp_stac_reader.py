import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Sequence, List

import pystac
import rasterio
from pystac import Collection
from pystac.extensions.eo import EOExtension, ItemEOExtension, Band
from rio_tiler.io import COGReader, STACReader
from rio_tiler.models import ImageData
from rio_tiler.utils import render
from solaris.tile.raster_tile import RasterTiler
from solaris.utils import tile
from solaris.utils.geo import split_geom
from torchgeo.samplers import GridGeoSampler
from torchgeo.datasets import GeoDataset, BoundingBox, SpaceNet1, CDL, ChesapeakeCVPR, VisionDataset
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
            for _, a_meta in self.item.assets.items():
                bands = []
                if 'eo:bands' in a_meta.extra_fields.keys():
                    bands = a_meta.extra_fields['eo:bands']
                if len(bands) == 1:
                    eo_band = bands[0].get('common_name')
                    if eo_band:
                        if not Band.band_range(eo_band):  # Hacky but easiest way to validate common names
                            raise ValueError(f'Must be one of the accepted common names. Got "{eo_band}".')
                        else:
                            self._assets_by_common_name[eo_band] = a_meta.href
        if not self._assets_by_common_name:
            raise ValueError(f"Common names for assets cannot be retrieved")
        return self._assets_by_common_name


class InferenceDataset(VisionDataset):
    def __init__(
        self,
        item_url: str,
        tile_size: int = 512,
        resize_factor: int = 1,
        use_projection_units=False,
        root: str = 'data',
        modalities: Sequence = ("red", "blue", "green"),
        download: bool = False
    ) -> None:
        """Initialize a new CCCOT Dataset instance.

        Arguments
        ---------
            item_url: TODO
            tile_size : `tuple` of `int`s
                The size of the input tiles in ``(y, x)`` coordinates. By default,
                this is in pixel units; this can be changed to metric units using the
                `use_metric_size` argument.
            resize_factor: TODO
            use_projection_units : bool, optional
                Is `tile_size` in pixel units (default) or distance units? To set to distance units
                use ``use_projection_units=True``. If False, resolution must be supplied.
            root: root directory where dataset can be found
            modalities: band selection which must be a list of STAC Item common names from eo extension.
                        See: https://github.com/stac-extensions/eo/#common-band-names
            download: if True, download dataset and store it in the root directory.

            Raises:
                RuntimeError: if ``download=False`` but dataset is missing
        """
        self.item_url = item_url
        self.assets_dict = test_asset_dict  # SingleBandItemEO(pystac.Item.from_file(item_url)).asset_by_common_name FIXME
        self.main_asset = list(self.assets_dict.values())[0]['href']  # Default to first asset

        self.modalities = modalities
        self.root = root
        self.download = download

        self.src = rasterio.open(self.main_asset)
        self.resizing_factor = resize_factor
        self.dest_tile_size = tile_size
        self.src_tile_size = self.get_src_tile_size()
        self.use_projection_units = use_projection_units
        self.tile_bounds = split_geom(
            geometry=list(self.src.bounds),
            tile_size=(self.src_tile_size, self.src_tile_size),
            resolution=(self.src.transform[0], -self.src.transform[4]),
            use_projection_units=self.use_projection_units,
            src_img=self.src
        )

    def get_src_tile_size(self):
        """
        Sets outputs dimension of source tile if resizing, given destination size and resizing factor
        @param dest_tile_size: (int) Size of tile that is expected as output
        @param resize_factor: (float) Resize factor to apply to source imagery before outputting tiles
        @return: (int) Source tile size
        """
        if self.resizing_factor:
            return int(self.dest_tile_size / self.resizing_factor)
        else:
            return self.dest_tile_size

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            imagery at that index

        Raises:
            IndexError: if index is out of range of the dataset
        """
        bbox = self.tile_bounds[index]
        assets = [self.assets_dict[band]['name'] for band in self.modalities]
        with STACReader(self.item_url) as stac:
            img = stac.part(bbox, bounds_crs=self.src.crs, assets=assets)
            # with open(f'test{index}.tif', 'wb') as f:
            #     f.write(img.render(img_format="GTIFF"))
        return img

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            length of the dataset
        """
        return len(self.tile_bounds)


collection = "https://datacube-stage.services.geo.ca/api/collections/geoeye-1-ortho-pansharp"

# item_url = "https://datacube-stage.services.geo.ca/api/collections/worldview-2-ortho-pansharp/items/VancouverP003_054230029070_01_P003_WV2"
item_url = "/media/data/GDL_all_images/QC22.json"
item = pystac.Item.from_file(item_url)
#eoitem = EOExtension.ext(item)
# ...

#item_com_names = [band.common_name for band in item]

#eoitem = EOExtension.ext(item)
#eoitem_com_names = [band.common_name for band in eoitem.bands]

# eoitem = SingleBandItemEO(item)
inf_dataset = InferenceDataset(item_url, modalities=("red", "green", "blue", "nir"))
for img in tqdm(inf_dataset):
    print(img)

#my_asset = eoitem.asset_by_common_name("blue")
#tg_dataset = SpaceNet1(root="/media/data/spacenet", image="8band", download=True, api_key="2f56530ae7b25f7a5af1d0347bf0a2c73d8862f0ae3e061d3fcaf3ed778aaded")
#cdl_dataset = CDL(root="/media/data/CDL", download=True, cache=True)
#cp_dataset = ChesapeakeCVPR(root="/media/data/Chesapeake", download=True, cache=True)
#print(tg_dataset)

#item3 = Item.open(item_url)
#test = item3.assets_by_common_name
#print(test)




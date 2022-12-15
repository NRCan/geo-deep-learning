from collections import OrderedDict
import logging
from typing import Optional, Sequence, Dict, List

import pystac
from pystac.extensions.eo import ItemEOExtension, Band

from utils.geoutils import is_stac_item

logging.getLogger(__name__)


class SingleBandItemEO(ItemEOExtension):
    """
    Single-Band Stac Item with assets by common name.
    For info on common names, see https://github.com/stac-extensions/eo#common-band-names
    """
    def __init__(
            self,
            item: pystac.Item,
            bands_requested: Optional[Sequence] = None,
    ):
        """

        @param item:
            Stac item containing metadata linking imagery assets
        @param bands_requested:
            band selection which must be a list of STAC Item common names from eo extension.
            See: https://github.com/stac-extensions/eo/#common-band-names
        """
        super().__init__(item)
        if not is_stac_item(item):
            raise TypeError(f"Expected a valid pystac.Item object. Got {type(item)}")
        self.item = item
        self._assets_by_common_name = None

        if not bands_requested:
            raise ValueError(f"At least one band should be chosen if assets need to be reached")

        # Create band inventory (all available bands)
        self.bands_all = [band for band in self.asset_by_common_name.keys()]

        # Make sure desired bands are subset of inventory
        if not set(bands_requested).issubset(set(self.bands_all)):
            raise ValueError(f"Requested bands ({bands_requested}) should be a subset of available bands ({self.bands_all})")

        # Filter only requested bands
        self.bands_requested = {band: self.asset_by_common_name[band] for band in bands_requested}
        logging.debug(self.bands_all)
        logging.debug(self.bands_requested)

        bands = []
        for band in self.bands_requested.keys():
            band = Band.create(
                name=self.bands_requested[band]['name'],
                common_name=band,
                description=self.bands_requested[band]['meta'].description,
                center_wavelength=self.bands_requested[band]['meta'].extra_fields['eo:bands'][0]['center_wavelength'],
                full_width_half_max=self.bands_requested[band]['meta'].extra_fields['eo:bands'][0]['full_width_half_max'])
            bands.append(band)
        self.bands = bands

    @property
    def asset_by_common_name(self) -> Dict:
        """
        Get assets by common band name (only works for assets containing 1 band)
        Adapted from:
        https://github.com/sat-utils/sat-stac/blob/40e60f225ac3ed9d89b45fe564c8c5f33fdee7e8/satstac/item.py#L75
        @return:
        """
        if self._assets_by_common_name is None:
            self._assets_by_common_name = OrderedDict()
            for name, a_meta in self.item.assets.items():
                bands = []
                if 'eo:bands' in a_meta.extra_fields.keys():
                    bands = a_meta.extra_fields['eo:bands']
                if len(bands) == 1:
                    eo_band = bands[0]
                    if 'common_name' in eo_band.keys():
                        common_name = eo_band['common_name']
                        if not self.is_valid_cname(common_name):
                            raise ValueError(f'Must be one of the accepted common names. Got "{common_name}".')
                        else:
                            self._assets_by_common_name[common_name] = {'meta': a_meta, 'name': name}
        if not self._assets_by_common_name:
            raise ValueError(f"Common names for assets cannot be retrieved")
        return self._assets_by_common_name

    @staticmethod
    def is_valid_cname(common_name: str) -> bool:
        """Checks if a band name is a valid common name according to STAC spec"""
        return True if Band.band_range(common_name) else False

    @staticmethod
    def band_to_cname(input_band: str):
        """
        Naive conversion of a band to a valid common name
        See: https://github.com/stac-extensions/eo/issues/13
        """
        bands_ref = (("red", "R"), ("green", "G"), ("blue", "B"), ('nir', "N"))
        if isinstance(input_band, int) and 1 <= input_band <= 4:
            return bands_ref[input_band-1][0]
        elif isinstance(input_band, str) and len(input_band) == 1:
            for cname, short_name in bands_ref:
                if input_band == short_name:
                    return cname
        elif isinstance(input_band, str) and len(input_band) > 1:
            for cname, short_name in bands_ref:
                if input_band == cname:
                    return input_band
        else:
            raise ValueError(f"Cannot convert given band to valid stac common name. Got: {input_band}")

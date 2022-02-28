from typing import Dict

import pystac
from pystac.extensions.eo import ItemEOExtension, Band


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
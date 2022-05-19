from dataset.aoi import AOI


class Test_AOI(object):
    def test_parse_input_raster(self) -> None:
        raster_raw = {
            "https://datacube-stage.services.geo.ca/api/collections/spacenet-samples/items/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03": [
                "red", "green", "blue"],
            "tests/data/massachusetts_buildings_kaggle/22978945_15_uint8_clipped_${dataset.bands}.tif": ["R", "G", "B"],
            "tests/data/massachusetts_buildings_kaggle/22978945_15_uint8_clipped.tif": None,
        }
        for raster_raw, bands_requested in raster_raw.items():
            raster_parsed = AOI.parse_input_raster(csv_raster_str=raster_raw, raster_bands_requested=bands_requested)
            print(raster_parsed)

# TODO: SingleBandItem
# test raise ValueError if request more than available bands
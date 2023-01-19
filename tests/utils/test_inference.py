import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
from torchgeo.datasets.utils import extract_archive

from utils.inference import stretch_heatmap, class_from_heatmap


class TestInferenceUtils(object):
    def test_stretch_heatmap(self) -> None:
        """Tests the "stretch heatmap" utility"""
        extract_archive(src="tests/data/new_brunswick_aerial.zip")
        htmap = rasterio.open("tests/data/new_brunswick_aerial/23322E759967N_clipped_1m_inference_heatmap.tif").read()
        htmap_stretched = stretch_heatmap(heatmap_arr=htmap, out_max=100)
        assert int(htmap_stretched.min()) == 0
        assert int(htmap_stretched.max()) == 100

    def test_class_from_heatmap(self) -> None:
        """Tests the "class_from_heatmap" utility"""
        extract_archive(src="tests/data/new_brunswick_aerial.zip")
        htmap = rasterio.open("tests/data/new_brunswick_aerial/23322E759967N_clipped_1m_inference_heatmap.tif").read()
        htmap = reshape_as_image(htmap)  # comes channels last during inference
        htmap_flat = class_from_heatmap(heatmap_arr=htmap)
        assert len(htmap_flat.shape) == 2
        assert list(np.unique(htmap_flat)) == [0, 4]

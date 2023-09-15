import numpy as np
import rasterio

from typing import List
from rasterio.plot import reshape_as_image
from torchgeo.datasets.utils import extract_archive

from utils.inference import stretch_heatmap, class_from_heatmap
from utils.inference import window2d, generate_corner_windows, generate_patch_list

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

    def test_window2d(self) -> None:
        """Tests 2D Hann Periodic Window"""
        window_size = 2
        output = window2d(window_size)
        expected_shape = (window_size, window_size)
        expected_output = np.matrix([[0., 0.], [0., 1.]])
        assert output.shape == expected_shape
        assert np.allclose(output, expected_output)
        assert np.all(output >= 0) and np.all(output <= 1)

    def test_generate_corner_windows(self) -> None:
        """ Tests all 9 2D Hann Windows (center, corner and edges)"""
        window_size = 8
        step = window_size >> 1
        output = generate_corner_windows(window_size)
        center_window = window2d(window_size)
        window_top = np.vstack([np.tile(center_window[step:step + 1, :], (step, 1)), center_window[step:, :]])
        window_bottom = np.vstack([center_window[:step, :], np.tile(center_window[step:step + 1, :], (step, 1))])
        window_left = np.hstack([np.tile(center_window[:, step:step + 1], (1, step)), center_window[:, step:]])
        window_right = np.hstack([center_window[:, :step], np.tile(center_window[:, step:step + 1], (1, step))])

        window_top_left = np.block([[np.ones((step, step)), window_top[:step, step:]],
                                    [window_left[step:, :step], window_left[step:, step:]]])
        window_top_right = np.block([[window_top[:step, :step], np.ones((step, step))],
                                     [window_right[step:, :step], window_right[step:, step:]]])
        window_bottom_left = np.block([[window_left[:step, :step], window_left[:step, step:]],
                                       [np.ones((step, step)), window_bottom[step:, step:]]])
        window_bottom_right = np.block([[window_right[:step, :step], window_right[:step, step:]],
                                        [window_bottom[step:, :step], np.ones((step, step))]])

        assert output.shape == (3, 3, window_size, window_size)
        assert np.all(output >= 0) and np.all(output <= 1)
        assert np.allclose(output[1, 1], center_window)
        assert np.allclose(output[0, 1], window_top)
        assert np.allclose(output[2, 1], window_bottom)
        assert np.allclose(output[1, 0], window_left)
        assert np.allclose(output[1, 2], window_right)
        assert np.allclose(output[0, 0], window_top_left)
        assert np.allclose(output[0, 2], window_top_right)
        assert np.allclose(output[2, 0], window_bottom_left)
        assert np.allclose(output[2, 2], window_bottom_right)

    def test_generate_patch_list(self) -> None:
        """ Test non-overlapping patches """
        image_height = 256
        image_width = 256
        window = 32
        step = window
        num_columns = int(image_height / step)
        num_rows = int(image_width / step)
        len_of_patches = num_rows * num_columns
        patches = generate_patch_list(image_height, image_width, window, overlapping=False)
        assert isinstance(patches, List)
        assert len(patches) == len_of_patches

        for patch in patches:
            x, y, width, height, data = patch
            assert x + width <= image_width
            assert y + height <= image_height
            assert isinstance(data, np.ndarray)
            assert data.shape == (window, window)
            assert np.allclose(data, np.ones((window, window)))

    def test_generate_patch_list_overlap(self) -> None:
        """ Test overlapping patches """
        image_height = 256
        image_width = 256
        window = 32
        step = window >> 1
        num_columns = int(image_height / step - 1)
        num_rows = int(image_width / step - 1)
        len_of_patches = num_rows * num_columns
        patches_overlap = generate_patch_list(image_height, image_width, window, overlapping=True)

        assert isinstance(patches_overlap, List)
        assert len(patches_overlap) == len_of_patches

        for patch in patches_overlap:
            x, y, width, height, data = patch
            assert x + width <= image_width
            assert y + height <= image_height
            assert isinstance(data, np.ndarray)
            assert data.shape == (window, window)

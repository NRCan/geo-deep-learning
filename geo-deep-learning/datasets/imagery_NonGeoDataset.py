import abc
import os
from pathlib import Path
from typing import Union, Optional, Sequence, Callable, Dict, Any, cast, List, Tuple

import pandas
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt
import numpy as np
from pandas.io.common import is_url
import pystac
from pystac.extensions.eo import ItemEOExtension, Band
import rasterio as rio
from rtree import Index
from rtree.index import Property
import torch
from torchgeo.datasets import RasterDataset, BoundingBox
from torchvision.datasets.utils import download_url

from matplotlib.figure import Figure
from rasterio.crs import CRS
from rasterio.transform import Affine
from torch import Tensor

from torchgeo.datasets import NonGeoDataset


class BlueSkyNonGeo(NonGeoDataset):
    """ 
    This dataset class is intended to handle data for semantic segmentation of geospatial imagery (Binary | Multiclass).
    It loads geospatial image and label patches from csv files.
    
    Dataset format:
    
    * images are composed of arbitrary number of bands
    * labels are single band images with pixel values representing classes
    * images and labels are stored in trn, val, and tst folders
    * csv files contain the path part starting to the image and label pairs e.g. 'trn/image/0.tif;trn/label/0_lbl.tif'
    * csv files may contain additional columns for other metadata e.g. 'trn/image/0.tif;trn/label/0_lbl.tif;aoi_id'
    * csv files are named after the data split e.g. 'trn.csv', 'val.csv', 'tst.csv'
    
    root_directory
    ├───trn
    │   ├───image
    │   │       0.tif
    │   ├───label
    │           0_lbl.tif
    ├───val
    │   ├───image
    │   │       0.tif
    │   ├───label
    │           0_lbl.tif
    ├───tst
    │   ├───image
    │   │       0.tif
    │   ├───label
    │           0_lbl.tif
    ├───trn.csv
    ├───val.csv
    ├───tst.csv
    
    Args:
        NonGeoDataset (_type_): _description_

    Raises:
        FileNotFoundError: _description_
        FileNotFoundError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    def __init__(self, 
                 csv_root_folder: str,
                 patches_root_folder: str,
                 split: str = "trn",
                 ):
        """_summary_

        Args:
            csv_root_folder (str): The root folder where the csv files are stored
            patches_root_folder (str): The root folder where the image and label patches are stored
            splits (List[str], optional): The data split. Defaults to ["trn", "val", "tst"].
        """
        self.csv_root_folder = csv_root_folder
        self.patches_root_folder = patches_root_folder
        self.split = split
        self.files = self._load_files()
    
    def _load_files(self):
        """Load image and label paths from csv files.

        Returns:
            List[Dict[str, str]]: list of dictionaries containing image and label paths 
        """
        csv_path = os.path.join(self.csv_root_folder, f"{self.split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file {csv_path} not found.")
        df = pandas.read_csv(csv_path, header=None, sep=";")
        if len(df.columns) == 1:
            raise ValueError("CSV file must contain at least two columns: image_path;label_path")
        
        files = [
            {"image": os.path.join(self.patches_root_folder, img),
             "label": os.path.join(self.patches_root_folder, lbl)} 
            for img, lbl in df[[0, 1]].itertuples(index=False)
            ]
        
        return files
         
    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)
    
    def _load_image(self, index: int):
        """ Load image

        Args:
            index (int): index of the image to load

        Returns:
            torch.Tensor: image tensor
        """
        with rio.open(self.files[index]["image"]) as image:
            image_array = image.read().astype(np.int32)
            image_tensor = torch.from_numpy(image_array).float()
        return image_tensor
    
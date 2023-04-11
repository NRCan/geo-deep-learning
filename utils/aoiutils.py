from pathlib import Path
from typing import Union, List
from tqdm import tqdm
from dataset.aoi import AOI
from utils.logger import get_logger
from utils.utils import read_csv, read_csv_change_detection

logging = get_logger(__name__)  # import logging


def aois_from_csv(
        csv_path: Union[str, Path],
        bands_requested: List = [],
        attr_field_filter: str = None,
        attr_values_filter: str = None,
        download_data: bool = False,
        data_dir: str = "data",
        for_multiprocessing = False,
        write_dest_raster = False,
        equalize_clahe_clip_limit: int = 0,
) -> list:
    """
    Creates list of AOIs by parsing a csv file referencing input data.
    
    .. note::
        See AOI docstring for information on other parameters and 
        see the dataset docs for details on expected structure of csv.

    Args:
        csv_path (Union[str, Path]): path to csv file containing list of input data. 
        bands_requested (List, optional): _description_. Defaults to [].
        attr_field_filter (str, optional): _description_. Defaults to None.
        attr_values_filter (str, optional): _description_. Defaults to None.
        download_data (bool, optional): _description_. Defaults to False.
        data_dir (str, optional): _description_. Defaults to "data".
        for_multiprocessing (bool, optional): _description_. Defaults to False.
        write_dest_raster (bool, optional): _description_. Defaults to False.
        equalize_clahe_clip_limit (int, optional): _description_. Defaults to 0.

    Returns:
        list: list of AOIs objects.
    """    
    aois = []
    data_list = read_csv(csv_path)
    logging.info(f'\n\tSuccessfully read csv file: {Path(csv_path).name}\n'
                 f'\tNumber of rows: {len(data_list)}\n'
                 f'\tCopying first row:\n{data_list[0]}\n')
    with tqdm(enumerate(data_list), desc="Creating AOI's", total=len(data_list)) as _tqdm:
        for i, aoi_dict in _tqdm:
            _tqdm.set_postfix_str(f"Image: {Path(aoi_dict['tif']).stem}")
            try:
                new_aoi = AOI.from_dict(
                    aoi_dict=aoi_dict,
                    bands_requested=bands_requested,
                    attr_field_filter=attr_field_filter,
                    attr_values_filter=attr_values_filter,
                    download_data=download_data,
                    root_dir=data_dir,
                    for_multiprocessing=for_multiprocessing,
                    write_dest_raster=write_dest_raster,
                    equalize_clahe_clip_limit=equalize_clahe_clip_limit,
                )
                logging.debug(new_aoi)
                aois.append(new_aoi)
            except FileNotFoundError as e:
                logging.error(f"{e}\nGround truth file may not exist or is empty.\n"
                              f"Failed to create AOI:\n{aoi_dict}\n"
                              f"Index: {i}")
    return aois


def aois_from_csv_change_detection(
        csv_path: Union[str, Path],
        bands_requested: List = [],
        attr_field_filter: str = None,
        attr_values_filter: str = None,
        download_data: bool = False,
        data_dir: str = "data",
        for_multiprocessing = False,
        write_dest_raster = False,
        equalize_clahe_clip_limit: int = 0,
) -> dict:
    """
    Creates list of AOIs by parsing a csv file referencing input data.
    
    .. note::
        See AOI docstring for information on other parameters and 
        see the dataset docs for details on expected structure of csv.

    Args:
        csv_path (Union[str, Path]): path to csv file containing list of input data. 
        bands_requested (List, optional): _description_. Defaults to [].
        attr_field_filter (str, optional): _description_. Defaults to None.
        attr_values_filter (str, optional): _description_. Defaults to None.
        download_data (bool, optional): _description_. Defaults to False.
        data_dir (str, optional): _description_. Defaults to "data".
        for_multiprocessing (bool, optional): _description_. Defaults to False.
        write_dest_raster (bool, optional): _description_. Defaults to False.
        equalize_clahe_clip_limit (int, optional): _description_. Defaults to 0.

    Returns:
        dict: dictionary of list of AOIs objects.
    """    
    aois = {}
    data_dict = read_csv_change_detection(csv_path)
    logging.info(
        f'\nSuccessfully read csv file: {Path(csv_path).name}\n'
        f'Number of rows: {len(data_dict["t1"])}\n'
        f'Copying first row at t1:\n{data_dict["t1"][0]}\n'
        f'Copying first row at t2:\n{data_dict["t2"][0]}\n'
    )
    for k in data_dict.keys():
        aois[k] = []
        desc = f"Creating AOI's for {k}"
        nb_e = len(data_dict[k])
        with tqdm(enumerate(data_dict[k]), desc=desc, total=nb_e) as _tqdm:
            for i, aoi_dict in _tqdm:
                _tqdm.set_postfix_str(f"Image: {Path(aoi_dict['tif']).stem}")
                try:
                    new_aoi = AOI.from_dict(
                        aoi_dict=aoi_dict,
                        bands_requested=bands_requested,
                        attr_field_filter=attr_field_filter,
                        attr_values_filter=attr_values_filter,
                        download_data=download_data,
                        root_dir=data_dir,
                        for_multiprocessing=for_multiprocessing,
                        write_dest_raster=write_dest_raster,
                        equalize_clahe_clip_limit=equalize_clahe_clip_limit,
                    )
                    logging.debug(new_aoi)
                    aois[k].append(new_aoi)
                except FileNotFoundError as e:
                    logging.error(
                        f"{e}\nGround truth file may not exist or is empty.\n"
                        f"Failed to create AOI:\n{aoi_dict}\n"
                        f"Index: {i}"
                    )
    return aois

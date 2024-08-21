import gc
import asyncio
from tqdm import tqdm
from pathlib import Path
from dataset.aoi import AOI
from typing import Union, List
from utils.logger import get_logger
from utils.utils import read_csv, read_csv_change_detection

logging = get_logger(__name__)  # import logging


def aois_from_csv(
    csv_path: (Union[str, Path]),
    bands_requested: List = [],
    attr_field_filter: str = None,
    attr_values_filter: str = None,
    data_dir: str = "data",
    for_multiprocessing = True,
    write_dest_raster = False,
    write_dest_zarr = False,
    raster_stats= False,
    equalize_clahe_clip_limit: int = 0,
):

    async def run_async(csv_path,bands_requested, attr_field_filter, attr_values_filter, data_dir, for_multiprocessing, write_dest_raster, write_dest_zarr, raster_stats, equalize_clahe_clip_limit):
        
        # Start the periodic garbage collection task
        gc_task = asyncio.create_task(constant_gc(3 if not write_dest_zarr else 50000))  # Calls gc.collect() every 3 seconds
        
        # Run the main computation asynchronously
        AOIs =  await run_aoi_async(
            csv_path= csv_path,
            bands_requested=bands_requested,
            attr_field_filter= attr_field_filter,
            attr_values_filter= attr_values_filter,
            data_dir= data_dir,
            for_multiprocessing = for_multiprocessing,
            write_dest_raster=write_dest_raster,
            write_dest_zarr=write_dest_zarr,
            raster_stats= raster_stats,
            equalize_clahe_clip_limit= equalize_clahe_clip_limit
        )
        gc_task.cancel()
        try:
            await gc_task
        except asyncio.CancelledError:
            pass
        
        return AOIs
    return asyncio.run(run_async(
        csv_path,
        bands_requested,
        attr_field_filter,
        attr_values_filter,
        data_dir,
        for_multiprocessing,
        write_dest_raster,
        write_dest_zarr,
        raster_stats,
        equalize_clahe_clip_limit
    ))
    
            
async def constant_gc(interval_seconds):
    while True:
        gc.collect() 
        await asyncio.sleep(interval_seconds)  # Wait for the specified interval

async def run_aoi_async(csv_path: (Union[str, Path]),
    bands_requested: List = [],
    attr_field_filter: str = None,
    attr_values_filter: str = None,
    data_dir: str = "data",
    for_multiprocessing = True,
    write_dest_raster = False,
    write_dest_zarr = False,
    raster_stats= False,
    equalize_clahe_clip_limit: int = 0,
):
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
        data_dir (str, optional): _description_. Defaults to "data".
        for_multiprocessing (bool, optional): _description_. Defaults to False.
        write_dest_raster (bool, optional): _description_. Defaults to False.
        write_dest_zarr (bool, optional): _description_. Defaults to False.
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
                    root_dir=data_dir,
                    for_multiprocessing=for_multiprocessing,
                    write_dest_raster=write_dest_raster,
                    write_dest_zarr=write_dest_zarr,
                    raster_stats= raster_stats,
                    equalize_clahe_clip_limit=equalize_clahe_clip_limit,
                )
                aois.append(new_aoi)
                logging.debug(new_aoi)
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
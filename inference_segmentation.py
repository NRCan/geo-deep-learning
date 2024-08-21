import csv
from math import sqrt
from tqdm import tqdm
from pathlib import Path
from numbers import Number
from tempfile import mkstemp
from omegaconf import DictConfig
from typing import Dict, Sequence, Union
from dataset.stacitem import SingleBandItemEO


from utils.aoiutils import aois_from_csv
from utils.logger import get_logger, set_tracker
from geo_inference.geo_inference import GeoInference
from utils.utils import get_device_ids, get_key_def, set_device

# Set the logging file
logging = get_logger(__name__)

def stac_input_to_temp_csv(input_stac_item: Union[str, Path]) -> Path:
    """Saves a stac item path or url to a temporary csv"""
    _, stac_temp_csv = mkstemp(suffix=".csv")
    with open(stac_temp_csv, "w", newline="") as fh:
        csv.writer(fh).writerow([str(input_stac_item), None, "inference", Path(input_stac_item).stem])
    return Path(stac_temp_csv)

def calc_inference_chunk_size(gpu_devices_dict: dict, max_pix_per_mb_gpu: int = 200, default: int = 512) -> int:
    """
    Calculate maximum chunk_size that could fit on GPU during inference based on thumb rule with hardcoded
    "pixels per MB of GPU RAM" as threshold. Threshold based on inference with a large model (Deeplabv3_resnet101)
    :param gpu_devices_dict: dictionary containing info on GPU devices as returned by lst_device_ids (utils.py)
    :param max_pix_per_mb_gpu: Maximum number of pixels that can fit on each MB of GPU (better to underestimate)
    :return: returns a downgraded evaluation batch size if the original batch size is considered too high
    """
    if not gpu_devices_dict:
        return default
    # get max ram for smallest gpu
    smallest_gpu_ram = min(gpu_info['max_ram'] for _, gpu_info in gpu_devices_dict.items())
    # rule of thumb to determine max chunk size based on approximate max pixels a gpu can handle during inference
    max_chunk_size = sqrt(max_pix_per_mb_gpu * smallest_gpu_ram)
    max_chunk_size_rd = int(max_chunk_size - (max_chunk_size % 256))  # round to the closest multiple of 256
    logging.info(f'Data will be split into chunks of {max_chunk_size_rd} if chunk_size is not specified.')
    return max_chunk_size_rd


def main(params:Union[DictConfig, Dict]):
    
    working_folder = get_key_def('root_dir', params['inference'], default="inference", to_path=True)
    working_folder.mkdir(exist_ok=True)
    model_path = get_key_def('model_path', 
                             params['inference'], 
                             to_path=True,
                             validate_path_exists=True,
                             wildcard='*.pt')
    mask_to_vector = get_key_def('mask_to_vector', params['inference'], default=False, expected_type=bool)
    
    # Set the device
    num_devices = get_key_def('gpu', params['inference'], default=0, expected_type=(int, bool))
    if num_devices > 1:
        logging.warning(f"Inference is not yet implemented for multi-gpu use. Will request only 1 GPU.")
        num_devices = 1
    max_used_ram = get_key_def('max_used_ram', params['inference'], default=25, expected_type=int)
    if not (0 <= max_used_ram <= 100):
        raise ValueError(f'\nMax used ram parameter should be a percentage. Got {max_used_ram}.')
    max_used_perc = get_key_def('max_used_perc', params['inference'], default=25, expected_type=int)
    gpu_devices_dict = get_device_ids(num_devices, max_used_ram_perc=max_used_ram, max_used_perc=max_used_perc)
    max_pix_per_mb_gpu = get_key_def('max_pix_per_mb_gpu', params['inference'], default=25, expected_type=int)
    auto_chunk_size = calc_inference_chunk_size(gpu_devices_dict=gpu_devices_dict,
                                                max_pix_per_mb_gpu=max_pix_per_mb_gpu, default=512)
    
    
    chunk_size = get_key_def('chunk_size', params['inference'], default=auto_chunk_size, expected_type=int)
    batch_size = get_key_def('batch_size', params['inference'], default=8, expected_type=int)
    device = set_device(gpu_devices_dict=gpu_devices_dict)
    
    
    # Dataset params
    bands_requested = get_key_def('bands', params['dataset'], default=[1, 2, 3], expected_type=Sequence)
    data_dir = get_key_def('raw_data_dir', params['dataset'], default="data", to_path=True, validate_path_exists=True)
    write_dest_raster = get_key_def('write_dest_raster', params['dataset'], default=True)
    clahe_clip_limit = get_key_def('clahe_clip_limit', params['tiling'], expected_type=Number, default=0)
    raw_data_csv = get_key_def('raw_data_csv', params['inference'], expected_type=str, to_path=True,
                               validate_path_exists=True)
    input_stac_item = get_key_def('input_stac_item', params['inference'], expected_type=str, to_path=True,
                                  validate_path_exists=True)
    
    if raw_data_csv and input_stac_item:
        raise ValueError(f"Input imagery should be either a csv of stac item. Got inputs from both \"raw_data_csv\" "
                         f"and \"input stac item\"")
    if input_stac_item:
        raw_data_csv = stac_input_to_temp_csv(input_stac_item)
        if not all([SingleBandItemEO.is_valid_cname(band) for band in bands_requested]):
            logging.warning(f"Requested bands are not valid stac item common names. Got: {bands_requested}")
            bands_requested = [SingleBandItemEO.band_to_cname(band) for band in bands_requested]
            logging.warning(f"Will request: {bands_requested}")
            
    # LOGGING PARAMETERS
    exper_name = get_key_def('project_name', params['general'], default='gdl-training')
    run_name = get_key_def(['tracker', 'run_name'], params, default='gdl')
    tracker_uri = get_key_def(['tracker', 'uri'], params, default=None, expected_type=str, to_path=False)
    set_tracker(mode='inference', type='mlflow', task='segmentation', experiment_name=exper_name, run_name=run_name,
                tracker_uri=tracker_uri, params=params, keys2log=['general', 'dataset', 'model', 'inference'])
    
    # GET LIST OF INPUT IMAGES FOR INFERENCE
    list_aois = aois_from_csv(
        csv_path=raw_data_csv,
        bands_requested=bands_requested,
        data_dir=data_dir,
        write_dest_raster =write_dest_raster,
        equalize_clahe_clip_limit=clahe_clip_limit,
    )
    
    # Create the inference object
    device_str = "gpu" if device.type == 'cuda' else "cpu"
    gpu_index = device.index if device.type == 'cuda' else 0
    
    geo_inference = GeoInference(model=str(model_path),
                                 work_dir=str(working_folder),
                                 batch_size=batch_size,
                                 mask_to_vec=mask_to_vector,
                                 device=device_str,
                                 gpu_id=gpu_index,
                                 )
    
    # LOOP THROUGH LIST OF INPUT IMAGES
    for aoi in tqdm(list_aois, desc='Inferring from images', position=0, leave=True):
        logging.info(f'\nReading image: {aoi.aoi_id}')
        raster = aoi.raster
        geo_inference(raster, tiff_name=aoi.raster_name, patch_size=chunk_size)
        
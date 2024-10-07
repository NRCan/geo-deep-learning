import csv
import rasterio

from tqdm import tqdm
from shutil import move
from pathlib import Path
from numbers import Number
from tempfile import mkstemp
from omegaconf import DictConfig
from typing import Dict, Sequence, Union

from utils.aoiutils import aois_from_csv
from dataset.stacitem import SingleBandItemEO
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


def main(params:Union[DictConfig, Dict]):
    
    working_folder = get_key_def('root_dir', params['inference'], default="inference", to_path=True)
    working_folder.mkdir(exist_ok=True)
    model_path = get_key_def('model_path', 
                             params['inference'], 
                             to_path=True,
                             validate_path_exists=True,
                             wildcard='*pt')
    
    prep_data_only = get_key_def('prep_data_only', params['inference'], default=False, expected_type=bool)

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
    patch_size = get_key_def('patch_size', params['inference'], default=1024, expected_type=int)
    workers = get_key_def('workers', params['inference'], default=0, expected_type=int)
    prediction_threshold = get_key_def('prediction_threshold', params['inference'], default=0.3, expected_type=float)
    device = set_device(gpu_devices_dict=gpu_devices_dict)
    
    
    # Dataset params
    bands_requested = get_key_def('bands', params['dataset'], default=[1, 2, 3], expected_type=Sequence)
    classes_dict = get_key_def('classes_dict', params['dataset'], expected_type=DictConfig)
    download_data = get_key_def('download_data', params['inference'], default=False, expected_type=bool)
    data_dir = get_key_def('raw_data_dir', params['dataset'], default="data", to_path=True, validate_path_exists=True)
    clahe_clip_limit = get_key_def('clahe_clip_limit', params['tiling'], expected_type=Number, default=0)
    raw_data_csv = get_key_def('raw_data_csv', params['inference'], expected_type=str, to_path=True,
                               validate_path_exists=True)
    input_stac_item = get_key_def('input_stac_item', params['inference'], expected_type=str, to_path=True,
                                  validate_path_exists=True)
    num_classes = get_key_def('num_classes', params['inference'], expected_type=int, default=5)
    vectorize = get_key_def('ras2vec', params['inference'], expected_type=bool, default=False)
    transform_flip = get_key_def('flip', params['inference'], expected_type=bool, default=False)
    transform_rotate = get_key_def('rotate', params['inference'], expected_type=bool, default=False)
    transforms = True if transform_flip or transform_rotate else False
    
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
        download_data=download_data,
        data_dir=data_dir,
        equalize_clahe_clip_limit=clahe_clip_limit,
    )

    if prep_data_only:
        logging.info(f"[prep_data_only mode] Data preparation for inference is complete. Exiting...")
        exit()
    
    # Create the inference object
    device_str = "gpu" if device.type == 'cuda' else "cpu"
    gpu_index = device.index if device.type == 'cuda' else 0
    
    geo_inference = GeoInference(model=str(model_path),
                                 work_dir=str(working_folder),
                                 mask_to_vec=vectorize,
                                 device=device_str,
                                 gpu_id=gpu_index,
                                 num_classes=num_classes,
                                 prediction_threshold=prediction_threshold,
                                 transformers=transforms,
                                 transformer_flip=transform_flip,
                                 transformer_rotate=transform_rotate,
                                 )
    
    # LOOP THROUGH LIST OF INPUT IMAGES
    for aoi in tqdm(list_aois, desc='Inferring from images', position=0, leave=True):
        logging.info(f'\nReading image: {aoi.aoi_id}')
        input_path = str(aoi.raster.name)
        mask_name = geo_inference(input_path, patch_size=patch_size, workers=workers)
        mask_path = working_folder / mask_name
        
        # update metadata info and rename mask tif.
        if classes_dict is not None:
            meta_data_dict = {"checkpoint": str(model_path), 
                            "classes_dict": classes_dict}
            with rasterio.open(mask_path, 'r+') as raster:
                raster.update_tags(**meta_data_dict)
        output_path = get_key_def('output_path', params['inference'], expected_type=str, to_path=True, 
                                  default=mask_path)
        move(mask_path, output_path)
        logging.info(f"finished inferring image: {aoi.aoi_id} ")
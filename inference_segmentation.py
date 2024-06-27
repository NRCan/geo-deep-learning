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
from utils.utils import get_device_ids, set_device, fetch_param

# Set the logging file
logging = get_logger(__name__)


def stac_input_to_temp_csv(input_stac_item: Union[str, Path]) -> Path:
    """Saves a stac item path or url to a temporary csv"""
    _, stac_temp_csv = mkstemp(suffix=".csv")
    with open(stac_temp_csv, "w", newline="") as fh:
        csv.writer(fh).writerow(
            [str(input_stac_item), None, "inference", Path(input_stac_item).stem]
        )
        csv.writer(fh).writerow(
            [str(input_stac_item), None, "inference", Path(input_stac_item).stem]
        )
    return Path(stac_temp_csv)


def calc_inference_chunk_size(
    gpu_devices_dict: dict, max_pix_per_mb_gpu: int = 200, default: int = 512
) -> int:
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
    smallest_gpu_ram = min(
        gpu_info["max_ram"] for _, gpu_info in gpu_devices_dict.items()
    )
    # rule of thumb to determine max chunk size based on approximate max pixels a gpu can handle during inference
    max_chunk_size = sqrt(max_pix_per_mb_gpu * smallest_gpu_ram)
    max_chunk_size_rd = int(
        max_chunk_size - (max_chunk_size % 256)
    )  # round to the closest multiple of 256
    logging.info(
        f"Data will be split into chunks of {max_chunk_size_rd} if chunk_size is not specified."
    )
    return max_chunk_size_rd


def main(params: Union[DictConfig, Dict]):
    # all inference params in one spot --> adding/deleting is easier
    inference_params = {
        "root_dir": {"default": "inference", "to_path": True},
        "model_path": {
            "to_path": True,
            "validate_path_exists": True,
            "wildcard": "*pt",
        },
        "gpu": {"default": 0, "expected_type": (int, bool)},
        "max_used_ram": {"default": 25, "expected_type": int},
        "max_used_perc": {"default": 25, "expected_type": int},
        "max_pix_per_mb_gpu": {"default": 25, "expected_type": int},
        "chunk_size": {"default": None, "expected_type": int},  # Will be set later
        "batch_size": {"default": 8, "expected_type": int},
        "download_data": {
            "default": False,
            "expected_type": bool,
        },  # can be changed to streaming data
        "raw_data_csv": {
            "expected_type": str,
            "to_path": True,
            "validate_path_exists": True,
        },
        "input_stac_item": {
            "expected_type": str,
            "to_path": True,
            "validate_path_exists": True,
        },
    }
    tiling_params = {
        "clahe_clip_limit": {"default": 0, "expected_type": Number},
    }
    dataset_params = {
        "bands": {"default": [1, 2, 3], "expected_type": Sequence},
        "raw_data_dir": {
            "default": "data",
            "to_path": True,
            "validate_path_exists": True,
        },
    }

    # Merge all parameter with their corresponding categories
    global_params = {
        **{key: (attributes, "dataset") for key, attributes in dataset_params.items()},
        **{
            key: (attributes, "inference")
            for key, attributes in inference_params.items()
        },
        **{key: (attributes, "tiling") for key, attributes in tiling_params.items()},
    }

    # Fetch and assign all parameters to global variables
    for key, (attributes, cat) in global_params.items():
        globals()[key] = fetch_param(params, key, cat, **attributes)

    # Additional processing for specific parameters

    global_params["root_dir"].mkdir(exist_ok=True)
    if global_params["gpu"] > 1:
        logging.warning(
            "Inference is not yet implemented for multi-gpu use. Will request only 1 GPU."
        )
        gpu = 1
    print(global_params["max_used_ram"])
    if not (0 <= global_params["max_used_ram"] <= 100):
        raise ValueError(
            f"\nMax used ram parameter should be a percentage. Got {global_params['max_used_ram']}."
        )

    gpu_devices_dict = get_device_ids(
        num_devices=gpu,
        max_used_ram_perc=global_params["max_used_ram"],
        max_used_perc=global_params["max_used_perc"],
    )

    auto_chunk_size = calc_inference_chunk_size(
        gpu_devices_dict=gpu_devices_dict,
        max_pix_per_mb_gpu=global_params["max_pix_per_mb_gpu"],
        default=512,
    )
    chunk_size = global_params["chunk_size"] or auto_chunk_size

    device = set_device(gpu_devices_dict=gpu_devices_dict)

    # Validate raw data input
    # maybe we can have both?
    if global_params["raw_data_csv"] and global_params["input_stac_item"]:
        raise ValueError(
            'Input imagery should be either a csv or a stac item. Got inputs from both "raw_data_csv" '
            'and "input stac item".'
        )

    if global_params["input_stac_item"]:
        raw_data_csv = stac_input_to_temp_csv(global_params["input_stac_item"])
        if not all(
            [SingleBandItemEO.is_valid_cname(band) for band in global_params["bands"]]
        ):
            logging.warning(
                f"Requested bands are not valid stac item common names. Got: {global_params['bands']}"
            )
            # returns red, blue, green
            bands = [
                SingleBandItemEO.band_to_cname(band) for band in global_params["bands"]
            ]
            logging.warning(f"Will request: {bands}")

    # Logging parameters
    exper_name = fetch_param(params, "project_name", "general", default="gdl-training")
    run_name = fetch_param(params, "run_name", ["tracker"], default="gdl")
    tracker_uri = fetch_param(
        params, "uri", ["tracker"], default=None, expected_type=str, to_path=False
    )

    set_tracker(
        mode="inference",
        type="mlflow",
        task="segmentation",
        experiment_name=exper_name,
        run_name=run_name,
        tracker_uri=tracker_uri,
        params=params,
        keys2log=["general", "dataset", "model", "inference"],
    )

    # GET LIST OF INPUT IMAGES FOR INFERENCE
    list_aois = aois_from_csv(
        csv_path=raw_data_csv,
        bands_requested=bands,
        download_data=global_params["download_data"],
        data_dir=global_params["raw_data_dir"],
        equalize_clahe_clip_limit=global_params["clahe_clip_limit"],
    )

    # Create the inference object
    device_str = "gpu" if device.type == "cuda" else "cpu"
    gpu_index = device.index if device.type == "cuda" else 0

    geo_inference = GeoInference(
        model=str(global_params["model_path"]),
        work_dir=str(global_params["root_dir"]),
        batch_size=global_params["batch_size"],
        mask_to_vec=False,
        device=device_str,
        gpu_id=gpu_index,
    )

    # every time we are training with a single band
    # LOOP THROUGH LIST OF INPUT IMAGES
    for aoi in tqdm(list_aois, desc="Inferring from images", position=0, leave=True):
        logging.info(f"\nReading image: {aoi.aoi_id}")
        input_path = aoi.raster.name
        geo_inference(input_path, patch_size=chunk_size)


if __name__ == "__main__":
    geo_inference = GeoInference(
        model=str(
            "/gpfs/fs5/nrcan/nrcan_geobase/work/dev/datacube/parallel/deep_learning_model/4cls_RGB_5_1_2_3_scripted.pt"
        ),
        work_dir=str(
            "/gpfs/fs5/nrcan/nrcan_geobase/work/dev/datacube/parallel/Change_detection_Results/"
        ),
        batch_size=512,
        mask_to_vec=False,
        device="gpu",
        gpu_id=0,
    )
    input_path = "/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/operationalization/data/BC18-013904302070_01_P001-GE01_red-green-blue_clahe25.tif"
    geo_inference(input_path, patch_size=256)

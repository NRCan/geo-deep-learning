import csv
import itertools
from math import sqrt
from numbers import Number
from tempfile import mkstemp
from typing import List, Union, Sequence

import torch
import torch.nn.functional as F
# import torch should be first. Unclear issue, mentionned here: https://github.com/pytorch/pytorch/issues/2083
import numpy as np
import time
import fiona  # keep this import. it sets GDAL_DATA to right value
import rasterio
import ttach as tta
from collections import OrderedDict
from fiona.crs import to_string
from ruamel_yaml.comments import CommentedSeq
from tqdm import tqdm
from rasterio import features
from rasterio.windows import Window
from rasterio.plot import reshape_as_image, reshape_as_raster
from pathlib import Path
from omegaconf import OmegaConf, DictConfig, open_dict
from omegaconf.listconfig import ListConfig

from dataset.aoi import aois_from_csv
from dataset.stacitem import SingleBandItemEO
from utils.geoutils import create_new_raster_from_base
from utils.inference import stretch_heatmap, class_from_heatmap
from utils.logger import get_logger, set_tracker
from models.model_choice import define_model, read_checkpoint
from utils import augmentation
from utils.utils import get_device_ids, get_key_def, \
    add_metadata_from_raster_to_sample, _window_2D, set_device

# Set the logging file
logging = get_logger(__name__)
def _pad(arr, chunk_size):
    """ Pads img_arr """
    w_diff = chunk_size - arr.shape[0]
    h_diff = chunk_size - arr.shape[1]
    aug = int(round(chunk_size * (1 - 1.0 / 2.0)))
    aug_w = int(round(chunk_size * (1 - 1.0 / 2.0))) + w_diff
    aug_h = int(round(chunk_size * (1 - 1.0 / 2.0))) + h_diff
    if len(arr.shape) > 2:
        padded_arr = np.pad(arr, ((aug, aug_w), (aug, aug_h), (0, 0)), mode="reflect")
    else:
        padded_arr = np.pad(arr, ((aug, aug_w), (aug, aug_h)), mode="reflect")

    return padded_arr


def ras2vec(raster_file, output_path):
    # Create a generic polygon schema for the output vector file
    i = 0
    feat_schema = {'geometry': 'Polygon',
                   'properties': OrderedDict([('value', 'int')])
                   }
    class_value_domain = set()
    out_features = []

    print("   - Processing raster file: {}".format(raster_file))
    with rasterio.open(raster_file, 'r') as src:
        raster = src.read(1)
    mask = raster != 0
    # Vectorize the polygons
    polygons = features.shapes(raster, mask, transform=src.transform)

    # Create shapely polygon features
    for polygon in polygons:
        feature = {'geometry': {
            'type': 'Polygon',
            'coordinates': None},
            'properties': OrderedDict([('value', 0)])}

        feature['geometry']['coordinates'] = polygon[0]['coordinates']
        value = int(polygon[1])  # Pixel value of the class (layer)
        class_value_domain.add(value)
        feature['properties']['value'] = value
        i += 1
        out_features.append(feature)

    print("   - Writing output vector file: {}".format(output_path))
    num_layers = list(class_value_domain)  # Number of unique pixel value
    for num_layer in num_layers:
        polygons = [feature for feature in out_features if feature['properties']['value'] == num_layer]
        layer_name = 'vector_' + str(num_layer).rjust(3, '0')
        print("   - Writing layer: {}".format(layer_name))

        with fiona.open(output_path, 'w',
                        crs=to_string(src.crs),
                        layer=layer_name,
                        schema=feat_schema,
                        driver='GPKG') as dest:
            for polygon in polygons:
                dest.write(polygon)
    print("")
    print("Number of features written: {}".format(i))


def gen_img_samples(src, chunk_size, step, *band_order):
    """
    TODO
    Args:
        src: input image (rasterio object)
        chunk_size: image tile size
        step: stride used during inference (in pixels)
        *band_order: ignore

    Returns: generator object

    """
    for row in range(0, src.height, step):
        for column in range(0, src.width, step):
            window = Window.from_slices(slice(row, row + chunk_size),
                                        slice(column, column + chunk_size))
            if band_order:
                window_array = reshape_as_image(src.read(band_order[0], window=window))
            else:
                window_array = reshape_as_image(src.read(window=window))
            window_array = _pad(window_array, chunk_size)

            yield window_array, row, column

def sigmoid(x):
   return 1/(1+np.exp(-x))

@torch.no_grad()
def segmentation(param,
                 input_image,
                 num_classes: int,
                 model,
                 chunk_size: int,
                 device,
                 scale: List,
                 tp_mem,
                 heatmap_dtype=np.uint16,
                 debug=False,
                 ):
    """

    Args:
        param: parameter dict
        input_image: opened image (rasterio object)
        num_classes: number of classes
        model: model weights
        chunk_size: image tile size
        device: cuda/cpu device
        scale: scale range
        tp_mem: memory temp file for saving numpy array to disk
        debug: True/False
        heatmap_dtype:
            Output data type for heatmap. Ex.: Uint16 captures more information, but takes more space in memory or disk.

    Returns:

    """
    subdiv = 2
    sample = {"image": None, "mask": None, 'metadata': None}
    start_seg = time.time()
    print_log = True if logging.level == 20 else False  # 20 is INFO
    pad = chunk_size * 2
    h_padded, w_padded = [side + pad for side in input_image.shape]
    dist_samples = int(round(chunk_size * (1 - 1.0 / 2.0)))

    model.eval()  # switch to evaluate mode

    # initialize test time augmentation
    transforms = tta.Compose([tta.HorizontalFlip(), ])
    tf_len = len(transforms)
    # construct window for smoothing
    WINDOW_SPLINE_2D = _window_2D(window_size=pad, power=2.0)
    WINDOW_SPLINE_2D = torch.as_tensor(np.moveaxis(WINDOW_SPLINE_2D, 2, 0), ).type(torch.float)
    WINDOW_SPLINE_2D = WINDOW_SPLINE_2D.to(device)

    fp = np.memmap(tp_mem, dtype='float16', mode='w+', shape=(tf_len, h_padded, w_padded, num_classes))
    step = int(chunk_size / subdiv)
    total_inf_windows = int(np.ceil(input_image.height / step) * np.ceil(input_image.width / step))
    img_gen = gen_img_samples(src=input_image, chunk_size=chunk_size, step=step)
    single_class_mode = False if num_classes > 1 else True
    for sub_image, row, col in tqdm(img_gen, position=1, leave=False,
                    desc=f'Inferring on window slices of size {chunk_size}',
                    total=total_inf_windows):
        image_metadata = add_metadata_from_raster_to_sample(sat_img_arr=sub_image,
                                                            raster_handle=input_image,
                                                            raster_info={})

        sample['metadata'] = image_metadata
        totensor_transform = augmentation.compose_transforms(param,
                                                             dataset="tst",
                                                             scale=scale,
                                                             aug_type='totensor',
                                                             print_log=print_log)
        sample["image"] = sub_image
        sample = totensor_transform(sample)
        inputs = sample["image"].unsqueeze_(0)
        inputs = inputs.to(device)
        if inputs.shape[1] == 4 and any("module.modelNIR" in s for s in model.state_dict().keys()):
            # Init NIR   TODO: make a proper way to read the NIR channel
            #                  and put an option to be able to give the index of the NIR channel
            inputs_NIR = inputs[:, -1, ...]  # Extract the NIR channel -> [batch size, H, W] since it's only one channel
            inputs_NIR.unsqueeze_(1)  # add a channel to get the good size -> [:, 1, :, :]
            inputs = inputs[:, :-1, ...]  # take out the NIR channel and take only the RGB for the inputs
            inputs = [inputs, inputs_NIR]
        output_lst = []
        for transformer in transforms:
            # augment inputs
            augmented_input = transformer.augment_image(inputs)
            with torch.cuda.amp.autocast():
                augmented_output = model(augmented_input)
            if isinstance(augmented_output, OrderedDict) and 'out' in augmented_output.keys():
                augmented_output = augmented_output['out']
            logging.debug(f'Shape of augmented output: {augmented_output.shape}')
            # reverse augmentation for outputs
            deaugmented_output = transformer.deaugment_mask(augmented_output)
            if single_class_mode:
                deaugmented_output = deaugmented_output.squeeze(dim=0)
            else:
                deaugmented_output = F.softmax(deaugmented_output, dim=1).squeeze(dim=0)
            output_lst.append(deaugmented_output)
        outputs = torch.stack(output_lst)
        outputs = torch.mul(outputs, WINDOW_SPLINE_2D)
        outputs = outputs.permute(0, 2, 3, 1)
        outputs = outputs.reshape(tf_len, pad, pad, num_classes).cpu().numpy().astype('float16')
        outputs = outputs[:, dist_samples:-dist_samples, dist_samples:-dist_samples, :]
        fp[:, row:row + chunk_size, col:col + chunk_size, :] += outputs
    fp.flush()
    del fp

    fp = np.memmap(tp_mem, dtype='float16', mode='r', shape=(tf_len, h_padded, w_padded, num_classes))
    pred_heatmap = np.zeros((h_padded, w_padded, num_classes), dtype=heatmap_dtype)
    for row, col in tqdm(itertools.product(range(0, input_image.height, chunk_size),
                                           range(0, input_image.width, chunk_size)),
                         leave=False, total=total_inf_windows, desc="Writing to array"):
        arr1 = (fp[:, row:row + chunk_size, col:col + chunk_size, :]).max(axis=0)
        if single_class_mode:
            arr1 = sigmoid(arr1)

        heatmap_max = np.iinfo(heatmap_dtype).max
        arr1 = stretch_heatmap(heatmap_arr=arr1, out_max=heatmap_max)

        pred_heatmap[row:row + chunk_size, col:col + chunk_size, :] = arr1.astype(heatmap_dtype)

    end_seg = time.time() - start_seg
    logging.info('Segmentation operation completed in {:.0f}m {:.0f}s'.format(end_seg // 60, end_seg % 60))

    if debug:
        logging.debug(f'Bin count of final output: {np.unique(pred_heatmap, return_counts=True)}')
    input_image.close()
    return pred_heatmap[:h_padded-pad, :w_padded-pad]


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
    logging.info(f'Data will be split into chunks of {max_chunk_size_rd}')
    return max_chunk_size_rd


def override_model_params_from_checkpoint(
        params: DictConfig,
        checkpoint_params) -> DictConfig:
    """
    Overrides model-architecture related parameters from provided checkpoint parameters
    @param params: Original parameters as inputted through hydra
    @param checkpoint_params: Checkpoint parameters as saved during checkpoint creation when training
    @return:
    """
    bands = get_key_def('bands', params['dataset'], expected_type=Sequence)
    classes = get_key_def('classes_dict', params['dataset'], expected_type=(dict, DictConfig))
    clip_limit = get_key_def('clahe_clip_limit', params['tiling'], expected_type=int)
    normalization = get_key_def('normalization', params['augmentation'], expected_type=DictConfig)
    scale_data = get_key_def('scale_data', params['augmentation'], expected_type=ListConfig)

    bands_ckpt = get_key_def('bands', checkpoint_params['dataset'], expected_type=Sequence)
    classes_ckpt = get_key_def('classes_dict', checkpoint_params['dataset'], expected_type=(dict, DictConfig))
    model_ckpt = get_key_def('model', checkpoint_params, expected_type=(dict, DictConfig))
    clip_limit_ckpt = get_key_def('clahe_clip_limit', checkpoint_params['tiling'], expected_type=int)
    normalization_ckpt = get_key_def('normalization', checkpoint_params['augmentation'], expected_type=(dict, DictConfig))
    # Workaround for "omegaconf.errors.UnsupportedValueType: Value 'CommentedSeq' is not a supported primitive type"
    if normalization_ckpt is not None and isinstance(list(normalization_ckpt.values())[0], CommentedSeq):
        normalization_ckpt = {k: [float(val) for val in v] for k, v in normalization_ckpt.items()}
    scale_data_ckpt = get_key_def('scale_data', checkpoint_params['augmentation'], expected_type=(List, ListConfig))
    scale_data_ckpt = list(scale_data_ckpt)

    if model_ckpt != params.model or classes_ckpt != classes or bands_ckpt != bands \
            or clip_limit != clip_limit_ckpt:
        logging.info(f"\nParameters from checkpoint will override inputted parameters."
                     f"\n\t\t\t Inputted | Overriden"
                     f"\nModel:\t\t {params.model} | {model_ckpt}"
                     f"\nInput bands:\t\t{bands} | {bands_ckpt}"
                     f"\nOutput classes:\t\t{classes} | {classes_ckpt}"
                     f"\nNormalization means and stds:\t\t{normalization} | {normalization_ckpt}"
                     f"\nScale data range:\t\t{scale_data} | {scale_data_ckpt}"
                     f"\nRaster enhance clip limit:\t\t{clip_limit} | {clip_limit_ckpt}")
        with open_dict(params):
            params['model'] = model_ckpt
            params['dataset']['bands'] = bands_ckpt
            params['dataset']['classes_dict'] = classes_ckpt
            params['augmentation']['normalization'] = normalization_ckpt
            params['augmentation']['scale_data'] = scale_data_ckpt
            params['tiling']['clahe_clip_limit'] = clip_limit_ckpt
    return params


def stac_input_to_temp_csv(input_stac_item: Union[str, Path]) -> Path:
    """Saves a stac item path or url to a temporary csv"""
    _, stac_temp_csv = mkstemp(suffix=".csv")
    with open(stac_temp_csv, "w", newline="") as fh:
        csv.writer(fh).writerow([str(input_stac_item), None, "inference", Path(input_stac_item).stem])
    return Path(stac_temp_csv)


def main(params: Union[DictConfig, dict]) -> None:
    """
    Function to manage details about the inference on segmentation task.
    1. Read the parameters from the config given.
    2. Read and load the state dict from the previous training or the given one.
    3. Make the inference on the data specified in the config.
    -------
    :param params: (dict) Parameters inputted during execution.
    """
    # Main params
    working_folder = get_key_def('root_dir', params['inference'], default="inference", to_path=True)
    working_folder.mkdir(exist_ok=True)
    logging.info(f'\nInferences will be saved to: {working_folder}\n\n')

    state_dict = get_key_def('state_dict_path', params['inference'], to_path=True,
                             validate_path_exists=True,
                             wildcard='*pth.tar')
    models_dir = get_key_def('checkpoint_dir', params['inference'], default=working_folder / 'checkpoints', to_path=True)
    models_dir.mkdir(exist_ok=True)
    data_dir = get_key_def('raw_data_dir', params['dataset'], default="data", to_path=True, validate_path_exists=True)
    download_data = get_key_def('download_data', params['inference'], default=False, expected_type=bool)

    # Override params from checkpoint
    checkpoint = read_checkpoint(state_dict, out_dir=models_dir)
    params = override_model_params_from_checkpoint(
        params=params,
        checkpoint_params=checkpoint['params']
    )

    # Dataset params
    bands_requested = get_key_def('bands', params['dataset'], default=[1, 2, 3], expected_type=Sequence)
    classes_dict = get_key_def('classes_dict', params['dataset'], expected_type=DictConfig)
    num_classes = len(classes_dict)
    num_classes = num_classes + 1 if num_classes > 1 else num_classes  # multiclass account for background
    num_bands = len(bands_requested)

    # Default input directory based on default output directory
    raw_data_csv = get_key_def('raw_data_csv', params['inference'], expected_type=str, to_path=True,
                               validate_path_exists=True)
    input_stac_item = get_key_def('input_stac_item', params['inference'], expected_type=str, to_path=True,
                                  validate_path_exists=True)
    prep_data_only = get_key_def('prep_data_only', params['inference'], default=False, expected_type=bool)

    # LOGGING PARAMETERS
    exper_name = get_key_def('project_name', params['general'], default='gdl-training')
    run_name = get_key_def(['tracker', 'run_name'], params, default='gdl')
    tracker_uri = get_key_def(['tracker', 'uri'], params, default=None, expected_type=str, to_path=True)
    set_tracker(mode='inference', type='mlflow', task='segmentation', experiment_name=exper_name, run_name=run_name,
                tracker_uri=tracker_uri, params=params, keys2log=['general', 'dataset', 'model', 'inference'])

    # OPTIONAL PARAMETERS
    num_devices = get_key_def('gpu', params['inference'], default=0, expected_type=(int, bool))
    if num_devices > 1:
        logging.warning(f"Inference is not yet implemented for multi-gpu use. Will request only 1 GPU.")
        num_devices = 1
    max_used_ram = get_key_def('max_used_ram', params['inference'], default=25, expected_type=int)
    if not (0 <= max_used_ram <= 100):
        raise ValueError(f'\nMax used ram parameter should be a percentage. Got {max_used_ram}.')
    max_used_perc = get_key_def('max_used_perc', params['inference'], default=25, expected_type=int)
    scale = get_key_def('scale_data', params['augmentation'], default=[0, 1], expected_type=ListConfig)
    raster_to_vec = get_key_def('ras2vec', params['inference'], default=False)
    debug = get_key_def('debug', params, default=False, expected_type=bool)
    if debug:
        logging.warning(f'\nDebug mode activated. Some debug features may mobilize extra disk space and '
                        f'cause delays in execution.')

    # list of GPU devices that are available and unused. If no GPUs, returns empty dict
    gpu_devices_dict = get_device_ids(num_devices, max_used_ram_perc=max_used_ram, max_used_perc=max_used_perc)
    max_pix_per_mb_gpu = get_key_def('max_pix_per_mb_gpu', params['inference'], default=25, expected_type=int)
    auto_chunk_size = calc_inference_chunk_size(gpu_devices_dict=gpu_devices_dict,
                                                max_pix_per_mb_gpu=max_pix_per_mb_gpu, default=512)
    chunk_size = get_key_def('chunk_size', params['inference'], default=auto_chunk_size, expected_type=int)
    device = set_device(gpu_devices_dict=gpu_devices_dict)

    clahe_clip_limit = get_key_def('clahe_clip_limit', params['tiling'], expected_type=Number, default=0)
    heatmap_dtype = get_key_def('heatmap_dtype', params['inference'], default=np.uint16)
    save_heatmap = get_key_def('save_heatmap', params['inference'], default=True, expected_type=bool)
    heatmap_threshold = get_key_def('heatmap_threshold', params['inference'], default=0.5, expected_type=float)

    if raw_data_csv and input_stac_item:
        raise ValueError(f"Input imagery should be either a csv of stac item. Got inputs from both \"raw_data_csv\" "
                         f"and \"input stac item\"")
    if input_stac_item:
        raw_data_csv = stac_input_to_temp_csv(input_stac_item)
        if not all([SingleBandItemEO.is_valid_cname(band) for band in bands_requested]):
            logging.warning(f"Requested bands are not valid stac item common names. Got: {bands_requested}")
            bands_requested = [SingleBandItemEO.band_to_cname(band) for band in bands_requested]
            logging.warning(f"Will request: {bands_requested}")

    model = define_model(
        net_params=params.model,
        in_channels=num_bands,
        out_classes=num_classes,
        main_device=device,
        devices=[list(gpu_devices_dict.keys())],
        checkpoint_dict=checkpoint,
    )

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

    # LOOP THROUGH LIST OF INPUT IMAGES
    for aoi in tqdm(list_aois, desc='Inferring from images', position=0, leave=True):
        inference_image = working_folder / f"{aoi.raster_name.stem}_inference.tif"
        inference_heatmap = working_folder / f"{aoi.raster_name.stem}_heatmap.tif"
        temp_file = working_folder / f"{aoi.raster_name.stem}.dat"
        logging.info(f'\nReading image: {aoi.raster_name.stem}')
        inf_meta = aoi.raster.meta

        pred_heatmap = segmentation(
            param=params,
            input_image=aoi.raster,
            num_classes=num_classes,
            model=model,
            chunk_size=chunk_size,
            device=device,
            scale=scale,
            tp_mem=temp_file,
            heatmap_dtype=heatmap_dtype,
            debug=debug
        )

        inf_meta.update({"driver": "GTiff",
                         "height": pred_heatmap.shape[1],
                         "width": pred_heatmap.shape[2],
                         "count": pred_heatmap.shape[0],
                         "dtype": 'uint8',
                         "compress": 'lzw'})
        logging.info(f'\nSuccessfully inferred on {aoi.raster_name}\nWriting to file: {inference_image}')

        pred_img = class_from_heatmap(heatmap_arr=pred_heatmap, heatmap_threshold=heatmap_threshold)

        if save_heatmap:
            logging.info(f"\nSaving heatmap...")
            pred_heatmap = reshape_as_raster(pred_heatmap)
            create_new_raster_from_base(
                input_raster=aoi.raster,
                output_raster=inference_heatmap,
                write_array=pred_heatmap,
                dtype=heatmap_dtype,
                checkpoint_path=state_dict,
            )
            logging.info(f'\nSaved heatmap to {inference_heatmap}')

        create_new_raster_from_base(
            input_raster=aoi.raster,
            output_raster=inference_image,
            write_array=pred_img,
            checkpoint_path=state_dict
            )
        del pred_heatmap

        try:
            temp_file.unlink()
        except OSError as e:
            logging.warning(f'File Error: {temp_file, e.strerror}')
        if raster_to_vec:
            start_vec = time.time()
            inference_vec = working_folder.joinpath(aoi.raster_name.parent.name,
                                                    f"{aoi.raster_name.split('.')[0]}_inference.gpkg")
            ras2vec(inference_image, inference_vec)
            end_vec = time.time() - start_vec
            logging.info('Vectorization completed in {:.0f}m {:.0f}s'.format(end_vec // 60, end_vec % 60))

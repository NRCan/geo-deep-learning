import itertools
from math import sqrt
from typing import List

import torch
import torch.nn.functional as F
# import torch should be first. Unclear issue, mentionned here: https://github.com/pytorch/pytorch/issues/2083
import numpy as np
import os
import csv
import time
import heapq
import fiona  # keep this import. it sets GDAL_DATA to right value
import rasterio
from PIL import Image
import torchvision
import ttach as tta
from collections import OrderedDict
from fiona.crs import to_string
from omegaconf.errors import ConfigKeyError
from tqdm import tqdm
from rasterio import features
from rasterio.windows import Window
from rasterio.plot import reshape_as_image
from pathlib import Path
from omegaconf.listconfig import ListConfig

from utils.logger import dict_path
from models.model_choice import net
from utils import augmentation
from utils.utils import load_from_checkpoint, get_device_ids, get_key_def, \
    list_input_images, add_metadata_from_raster_to_sample, _window_2D, read_modalities, find_first_file
from utils.verifications import validate_raster

try:
    import boto3
except ModuleNotFoundError:
    pass
# Set the logging file
from utils import utils
logging = utils.get_logger(__name__)


def _pad_diff(arr, w, h, arr_shape):
    """ Pads img_arr width or height < samples_size with zeros """
    w_diff = arr_shape - w
    h_diff = arr_shape - h

    if len(arr.shape) > 2:
        padded_arr = np.pad(arr, ((0, w_diff), (0, h_diff), (0, 0)), mode="constant", constant_values=np.nan)
    else:
        padded_arr = np.pad(arr, ((0, w_diff), (0, h_diff)), mode="constant", constant_values=np.nan)

    return padded_arr


def _pad(arr, chunk_size):
    """ Pads img_arr """
    aug = int(round(chunk_size * (1 - 1.0 / 2.0)))
    if len(arr.shape) > 2:
        padded_arr = np.pad(arr, ((aug, aug), (aug, aug), (0, 0)), mode="reflect")
    else:
        padded_arr = np.pad(arr, ((aug, aug), (aug, aug)), mode="reflect")

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

    # Create shapely polygon featyres
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
            if window_array.shape[0] < chunk_size or window_array.shape[1] < chunk_size:
                window_array = _pad_diff(window_array, window_array.shape[0], window_array.shape[1], chunk_size)
            window_array = _pad(window_array, chunk_size)

            yield window_array, row, column


@torch.no_grad()
def segmentation(param,
                 input_image,
                 num_classes: int,
                 model,
                 chunk_size: int,
                 device,
                 scale: List,
                 BGR_to_RGB: bool,
                 tp_mem,
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
        BGR_to_RGB: True/False
        tp_mem: memory temp file for saving numpy array to disk
        debug: True/False

    Returns:

    """
    padded = chunk_size * 2
    h, w = input_image.shape
    h_padded = h + padded
    w_padded = w + padded
    dist_samples = int(round(chunk_size * (1 - 1.0 / 2.0)))

    # switch to evaluate mode
    model.eval()

    # initialize test time augmentation
    transforms = tta.Compose([tta.HorizontalFlip(), ])
    # construct window for smoothing
    WINDOW_SPLINE_2D = _window_2D(window_size=padded, power=2.0)
    WINDOW_SPLINE_2D = torch.as_tensor(np.moveaxis(WINDOW_SPLINE_2D, 2, 0), ).type(torch.float)
    WINDOW_SPLINE_2D = WINDOW_SPLINE_2D.to(device)

    fp = np.memmap(tp_mem, dtype='float16', mode='w+', shape=(h_padded, w_padded, num_classes))
    sample = {'sat_img': None, 'map_img': None, 'metadata': None}
    cnt = 0
    subdiv = 2
    step = int(chunk_size / subdiv)
    total_inf_windows = int(np.ceil(input_image.height / step) * np.ceil(input_image.width / step))
    img_gen = gen_img_samples(src=input_image,
                              chunk_size=chunk_size,
                              step=step)
    single_class_mode = True
    threshold = 0.5
    if num_classes > 1:
        single_class_mode = False
    start_seg = time.time()
    print_log = True
    for img in tqdm(img_gen, position=1, leave=False,
                    desc=f'Inferring on window slices of size {chunk_size}',
                    total=total_inf_windows):
        row = img[1]
        col = img[2]
        sub_image = img[0]
        image_metadata = add_metadata_from_raster_to_sample(sat_img_arr=sub_image,
                                                            raster_handle=input_image,
                                                            raster_info={})

        sample['metadata'] = image_metadata
        totensor_transform = augmentation.compose_transforms(param,
                                                             dataset="tst",
                                                             input_space=BGR_to_RGB,
                                                             scale=scale,
                                                             aug_type='totensor',
                                                             print_log=print_log)
        sample['sat_img'] = sub_image
        sample = totensor_transform(sample)
        inputs = sample['sat_img'].unsqueeze_(0)
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
        outputs, _ = torch.max(outputs, dim=0)
        if single_class_mode:
            outputs = torch.sigmoid(outputs)
        outputs = outputs.permute(1, 2, 0)
        outputs = outputs.reshape(padded, padded, num_classes).cpu().numpy().astype('float16')
        outputs = outputs[dist_samples:-dist_samples, dist_samples:-dist_samples, :]
        fp[row:row + chunk_size, col:col + chunk_size, :] = \
            fp[row:row + chunk_size, col:col + chunk_size, :] + outputs
        cnt += 1
    fp.flush()
    del fp

    fp = np.memmap(tp_mem, dtype='float16', mode='r', shape=(h_padded, w_padded, num_classes))
    pred_img = np.zeros((h_padded, w_padded), dtype=np.uint8)
    for row, col in tqdm(itertools.product(range(0, input_image.height, step), range(0, input_image.width, step)),
                         leave=False,
                         total=total_inf_windows,
                         desc="Writing to array"):
        arr1 = fp[row:row + chunk_size, col:col + chunk_size, :] / (2 ** 2)
        if single_class_mode:
            arr1 = (arr1 > threshold)
            arr1 = np.squeeze(arr1, axis=2).astype(np.uint8)
        else:
            arr1 = arr1.argmax(axis=-1).astype('uint8')
        pred_img[row:row + chunk_size, col:col + chunk_size] = arr1
    pred_img = pred_img[:h, :w]
    end_seg = time.time() - start_seg
    logging.info('Segmentation operation completed in {:.0f}m {:.0f}s'.format(end_seg // 60, end_seg % 60))

    if debug:
        logging.debug(f'Bin count of final output: {np.unique(pred_img, return_counts=True)}')
    input_image.close()

    return pred_img


def classifier(params, img_list, model, device, working_folder):
    """
    Classify images by class
    :param params:
    :param img_list:
    :param model:
    :param device:
    :return:
    """
    weights_file_name = params['inference']['state_dict_path']
    num_classes = params['global']['num_classes']
    bucket = params['global']['bucket_name']

    classes_file = weights_file_name.split('/')[:-1]
    if bucket:
        class_csv = ''
        for folder in classes_file:
            class_csv = os.path.join(class_csv, folder)
        bucket.download_file(os.path.join(class_csv, 'classes.csv'), 'classes.csv')
        with open('classes.csv', 'rt') as file:
            reader = csv.reader(file)
            classes = list(reader)
    else:
        class_csv = ''
        for c in classes_file:
            class_csv = class_csv + c + '/'
        with open(class_csv + 'classes.csv', 'rt') as f:
            reader = csv.reader(f)
            classes = list(reader)

    classified_results = np.empty((0, 2 + num_classes))

    for image in img_list:
        img_name = os.path.basename(image['tif'])  # TODO: pathlib
        model.eval()
        if bucket:
            img = Image.open(f"Images/{img_name}").resize((299, 299), resample=Image.BILINEAR)
        else:
            img = Image.open(image['tif']).resize((299, 299), resample=Image.BILINEAR)
        to_tensor = torchvision.transforms.ToTensor()

        img = to_tensor(img)
        img = img.unsqueeze(0)
        with torch.no_grad():
            img = img.to(device)
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)

        top5 = heapq.nlargest(5, outputs.cpu().numpy()[0])
        top5_loc = []
        for i in top5:
            top5_loc.append(np.where(outputs.cpu().numpy()[0] == i)[0][0])
        logging.info(f"Image {img_name} classified as {classes[0][predicted]}")
        logging.info('Top 5 classes:')
        for i in range(0, 5):
            logging.info(f"\t{classes[0][top5_loc[i]]} : {top5[i]}")
        classified_results = np.append(classified_results, [np.append([image['tif'], classes[0][predicted]],
                                                                      outputs.cpu().numpy()[0])], axis=0)
    csv_results = 'classification_results.csv'
    if bucket:
        np.savetxt(csv_results, classified_results, fmt='%s', delimiter=',')
        bucket.upload_file(csv_results, os.path.join(working_folder, csv_results))  # TODO: pathlib
    else:
        np.savetxt(os.path.join(working_folder, csv_results), classified_results, fmt='%s',  # TODO: pathlib
                   delimiter=',')


def calc_inference_chunk_size(gpu_devices_dict: dict, max_pix_per_mb_gpu: int = 200):
    """
    Calculate maximum chunk_size that could fit on GPU during inference based on thumb rule with hardcoded
    "pixels per MB of GPU RAM" as threshold. Threshold based on inference with a large model (Deeplabv3_resnet101)
    :param gpu_devices_dict: dictionary containing info on GPU devices as returned by lst_device_ids (utils.py)
    :param max_pix_per_mb_gpu: Maximum number of pixels that can fit on each MB of GPU (better to underestimate)
    :return: returns a downgraded evaluation batch size if the original batch size is considered too high
    """
    # get max ram for smallest gpu
    smallest_gpu_ram = min(gpu_info['max_ram'] for _, gpu_info in gpu_devices_dict.items())
    # rule of thumb to determine max chunk size based on approximate max pixels a gpu can handle during inference
    max_chunk_size = sqrt(max_pix_per_mb_gpu * smallest_gpu_ram)
    max_chunk_size_rd = int(max_chunk_size - (max_chunk_size % 256))
    logging.info(f'Images will be split into chunks of {max_chunk_size_rd}')
    return max_chunk_size_rd


def main(params: dict) -> None:
    """
    Function to manage details about the inference on segmentation task.
    1. Read the parameters from the config given.
    2. Read and load the state dict from the previous training or the given one.
    3. Make the inference on the data specifies in the config.
    -------
    :param params: (dict) Parameters found in the yaml config file.
    """
    # PARAMETERS
    mode = get_key_def('mode', params, expected_type=str)
    task = get_key_def('task_name', params['task'], expected_type=str)
    model_name = get_key_def('model_name', params['model'], expected_type=str).lower()
    num_classes = len(get_key_def('classes_dict', params['dataset']).keys())
    modalities = read_modalities(get_key_def('modalities', params['dataset'], expected_type=str))
    BGR_to_RGB = get_key_def('BGR_to_RGB', params['dataset'], expected_type=bool)
    num_bands = len(modalities)
    debug = get_key_def('debug', params, default=False, expected_type=bool)
    # SETTING OUTPUT DIRECTORY
    try:
        state_dict = Path(params['inference']['state_dict_path']).resolve(strict=True)
    except FileNotFoundError:
        logging.info(
            f"\nThe state dict path directory '{params['inference']['state_dict_path']}' don't seem to be find," +
            f"we will try to locate a state dict path in the '{params['general']['save_weights_dir']}' " +
            f"specify during the training phase"
        )
        try:
            state_dict = Path(params['general']['save_weights_dir']).resolve(strict=True)
        except FileNotFoundError:
            raise logging.critical(
                f"\nThe state dict path directory '{params['general']['save_weights_dir']}'" +
                f" don't seem to be find either, please specify the path to a state dict"
            )
    # TODO add more detail in the parent folder
    working_folder = state_dict.parent.joinpath(f'inference_{num_bands}bands')
    logging.info("\nThe state dict path directory used '{}'".format(working_folder))
    Path.mkdir(working_folder, parents=True, exist_ok=True)

    # LOGGING PARAMETERS TODO put option not just mlflow
    experiment_name = get_key_def('project_name', params['general'], default='gdl-training')
    try:
        tracker_uri = get_key_def('uri', params['tracker'], default=None, expected_type=str)
        Path(tracker_uri).mkdir(exist_ok=True)
        run_name = get_key_def('run_name', params['tracker'], default='gdl')  # TODO change for something meaningful
        run_name = '{}_{}_{}'.format(run_name, mode, task)
        logging.info(f'\nInference and log files will be saved to: {working_folder}')
        # TODO change to fit whatever inport
        from mlflow import log_params, set_tracking_uri, set_experiment, start_run, log_artifact, log_metrics
        # tracking path + parameters logging
        set_tracking_uri(tracker_uri)
        set_experiment(experiment_name)
        start_run(run_name=run_name)
        log_params(dict_path(params, 'general'))
        log_params(dict_path(params, 'dataset'))
        log_params(dict_path(params, 'data'))
        log_params(dict_path(params, 'model'))
        log_params(dict_path(params, 'inference'))
    # meaning no logging tracker as been assigned or it doesnt exist in config/logging
    except ConfigKeyError:
        logging.info(
            "\nNo logging tracker as been assigned or the yaml config doesnt exist in 'config/tracker'."
            "\nNo tracker file will be save in that case."
        )

    # MANDATORY PARAMETERS
    img_dir_or_csv = get_key_def(
        'img_dir_or_csv_file', params['inference'], default=params['general']['raw_data_csv'], expected_type=str
    )
    if not (Path(img_dir_or_csv).is_dir() or Path(img_dir_or_csv).suffix == '.csv'):
        raise logging.critical(
            FileNotFoundError(
                f'\nCouldn\'t locate .csv file or directory "{img_dir_or_csv}" containing imagery for inference'
            )
        )
    # load the checkpoint
    try:
        # Sort by modification time (mtime) descending
        sorted_by_mtime_descending = sorted(
            [os.path.join(state_dict, x) for x in os.listdir(state_dict)], key=lambda t: -os.stat(t).st_mtime
        )
        last_checkpoint_save = find_first_file('checkpoint.pth.tar', sorted_by_mtime_descending)
        if last_checkpoint_save is None:
            raise FileNotFoundError
        # change the state_dict
        state_dict = last_checkpoint_save
    except FileNotFoundError as e:
        logging.error(f"\nNo file name 'checkpoint.pth.tar' as been found at '{state_dict}'")
        raise e

    task = get_key_def('task_name', params['task'], expected_type=str)
    # TODO change it next version for all task
    if task not in ['classification', 'segmentation']:
        raise logging.critical(
            ValueError(f'\nTask should be either "classification" or "segmentation". Got {task}')
        )

    # OPTIONAL PARAMETERS
    dontcare_val = get_key_def("ignore_index", params["training"], default=-1, expected_type=int)
    num_devices = get_key_def('num_gpus', params['training'], default=0, expected_type=int)
    default_max_used_ram = 25
    max_used_ram = get_key_def('max_used_ram', params['training'], default=default_max_used_ram, expected_type=int)
    max_used_perc = get_key_def('max_used_perc', params['training'], default=25, expected_type=int)
    scale = get_key_def('scale_data', params['augmentation'], default=[0, 1], expected_type=ListConfig)
    raster_to_vec = get_key_def('ras2vec', params['inference'], False) # FIXME not implemented with hydra

    if debug:
        logging.warning(f'\nDebug mode activated. Some debug features may mobilize extra disk space and '
                        f'cause delays in execution.')

    logging.info(f'\nInferences will be saved to: {working_folder}\n\n')
    if not (0 <= max_used_ram <= 100):
        logging.warning(f'\nMax used ram parameter should be a percentage. Got {max_used_ram}. '
                        f'Will set default value of {default_max_used_ram} %')
        max_used_ram = default_max_used_ram

    # AWS
    bucket = None
    bucket_file_cache = []
    bucket_name = get_key_def('bucket_name', params['AWS'])

    # list of GPU devices that are available and unused. If no GPUs, returns empty dict
    gpu_devices_dict = get_device_ids(num_devices,
                                      max_used_ram_perc=max_used_ram,
                                      max_used_perc=max_used_perc)
    if gpu_devices_dict:
        chunk_size = calc_inference_chunk_size(gpu_devices_dict=gpu_devices_dict, max_pix_per_mb_gpu=50)
        logging.info(f"\nNumber of cuda devices requested: {num_devices}. "
                     f"\nCuda devices available: {gpu_devices_dict}. "
                     f"\nUsing {list(gpu_devices_dict.keys())[0]}\n\n")
        device = torch.device(f'cuda:{list(range(len(gpu_devices_dict.keys())))[0]}')
    else:
        chunk_size = get_key_def('chunk_size', params['inference'], default=512, expected_type=int)
        logging.warning(f"\nNo Cuda device available. This process will only run on CPU")
        device = torch.device('cpu')

    # CONFIGURE MODEL
    if num_classes > 1:
        num_classes = num_classes + 1  # multiclass account for background
    model, loaded_checkpoint, model_name = net(model_name=model_name,
                                               num_bands=num_bands,
                                               num_channels=num_classes,
                                               num_devices=1,
                                               net_params=params,
                                               inference_state_dict=state_dict)
    try:
        model.to(device)
    except RuntimeError:
        logging.info(f"\nUnable to use device. Trying device 0")
        device = torch.device(f'cuda' if gpu_devices_dict else 'cpu')
        model.to(device)

    # CREATE LIST OF INPUT IMAGES FOR INFERENCE
    try:
        # check if the data folder exist
        raw_data_dir = get_key_def('raw_data_dir', params['dataset'])
        my_data_path = Path(raw_data_dir).resolve(strict=True)
        logging.info("\nImage directory used '{}'".format(my_data_path))
        data_path = Path(my_data_path)
    except FileNotFoundError:
        raw_data_dir = get_key_def('raw_data_dir', params['dataset'])
        raise logging.critical(
            "\nImage directory '{}' doesn't exist, please change the path".format(raw_data_dir)
        )
    list_img = list_input_images(img_dir_or_csv, bucket_name, glob_patterns=["*.tif", "*.TIF"],
                                 in_case_of_path=str(data_path))

    # VALIDATION: anticipate problems with imagery before entering main for loop
    for info in tqdm(list_img, desc='Validating imagery'):
        validate_raster(info['tif'], num_bands, None)
    logging.info('\nSuccessfully validated imagery')

    if task == 'classification':
        classifier(params, list_img, model, device,
                   working_folder)  # FIXME: why don't we load from checkpoint in classification?

    elif task == 'segmentation':
        # TODO: Add verifications?
        if bucket:
            bucket.download_file(loaded_checkpoint, "saved_model.pth.tar")  # TODO: is this still valid?
            model, _ = load_from_checkpoint("saved_model.pth.tar", model)
        else:
            model, _ = load_from_checkpoint(loaded_checkpoint, model)

        # Save tracking TODO put option not just mlflow
        if 'tracker_uri' in locals() and 'run_name' in locals():
            mode = get_key_def('mode', params, expected_type=str)
            task = get_key_def('task_name', params['task'], expected_type=str)
            run_name = '{}_{}_{}'.format(run_name, mode, task)
            # tracking path + parameters logging
            set_tracking_uri(tracker_uri)
            set_experiment(experiment_name)
            start_run(run_name=run_name)
            log_params(dict_path(params, 'inference'))
            log_params(dict_path(params, 'dataset'))
            log_params(dict_path(params, 'model'))

        # LOOP THROUGH LIST OF INPUT IMAGES
        for info in tqdm(list_img, desc='Inferring from images', position=0, leave=True):
            img_name = Path(info['tif']).name
            if bucket:
                local_img = f"Images/{img_name}"
                bucket.download_file(info['tif'], local_img)
                inference_image = f"Classified_Images/{img_name.split('.')[0]}_inference.tif"
            else:
                local_img = Path(info['tif'])
                Path.mkdir(working_folder.joinpath(local_img.parent.name), parents=True, exist_ok=True)
                inference_image = working_folder.joinpath(local_img.parent.name,
                                                          f"{img_name.split('.')[0]}_inference.tif")
            temp_file = working_folder.joinpath(local_img.parent.name, f"{img_name.split('.')[0]}.dat")
            raster = rasterio.open(local_img, 'r')
            logging.info(f'\nReading original image: {raster.name}')
            inf_meta = raster.meta

            pred = segmentation(param=params,
                                     input_image=raster,
                                     num_classes=num_classes,
                                     model=model,
                                     chunk_size=chunk_size,
                                     device=device,
                                     scale=scale,
                                     BGR_to_RGB=BGR_to_RGB,
                                     tp_mem=temp_file,
                                     debug=debug)

            pred = pred[np.newaxis, :, :].astype(np.uint8)
            inf_meta.update({"driver": "GTiff",
                             "height": pred.shape[1],
                             "width": pred.shape[2],
                             "count": pred.shape[0],
                             "dtype": 'uint8',
                             "compress": 'lzw'})
            logging.info(f'\nSuccessfully inferred on {img_name}\nWriting to file: {inference_image}')
            with rasterio.open(inference_image, 'w+', **inf_meta) as dest:
                dest.write(pred)
            del pred
            try:
                temp_file.unlink()
            except OSError as e:
                logging.warning(f'File Error: {temp_file, e.strerror}')
            if raster_to_vec:
                start_vec = time.time()
                inference_vec = working_folder.joinpath(local_img.parent.name,
                                                        f"{img_name.split('.')[0]}_inference.gpkg")
                ras2vec(inference_image, inference_vec)
                end_vec = time.time() - start_vec
                logging.info('Vectorization completed in {:.0f}m {:.0f}s'.format(end_vec // 60, end_vec % 60))

import logging
import warnings
from typing import List

import torch
import torch.nn.functional as F
# import torch should be first. Unclear issue, mentionned here: https://github.com/pytorch/pytorch/issues/2083
import numpy as np
import os
import csv
import time
import argparse
import heapq
import fiona  # keep this import. it sets GDAL_DATA to right value
import rasterio
from PIL import Image
import torchvision
import ttach as tta
from collections import OrderedDict, defaultdict
import pandas as pd
import geopandas as gpd
from fiona.crs import to_string
from tqdm import tqdm
from rasterio import features
from shapely.geometry import Polygon
from rasterio.windows import Window
from rasterio.plot import reshape_as_image
from pathlib import Path

from utils.metrics import ComputePixelMetrics
from models.model_choice import net, load_checkpoint
from utils import augmentation
from utils.geoutils import vector_to_raster, clip_raster_with_gpkg
from utils.utils import load_from_checkpoint, get_device_ids, get_key_def, \
    list_input_images, add_metadata_from_raster_to_sample, _window_2D, is_url, checkpoint_url_download, \
    compare_config_yamls
from utils.readers import read_parameters
from utils.verifications import add_background_to_num_class, validate_num_classes, assert_crs_match

try:
    import boto3
except ModuleNotFoundError:
    pass

logging.getLogger(__name__)


def _pad_diff(arr, w, h, arr_shape):
    """ Pads img_arr width or height < samples_size with zeros """
    w_diff = arr_shape - w
    h_diff = arr_shape - h

    if len(arr.shape) > 2:
        padded_arr = np.pad(arr, ((0, w_diff), (0, h_diff), (0, 0)), "constant", constant_values=np.nan)
    else:
        padded_arr = np.pad(arr, ((0, w_diff), (0, h_diff)), "constant", constant_values=np.nan)

    return padded_arr


def _pad(arr, chunk_size):
    """ Pads img_arr """
    aug = int(round(chunk_size * (1 - 1.0 / 2.0)))
    if len(arr.shape) > 2:
        padded_arr = np.pad(arr, ((aug, aug), (aug, aug), (0, 0)), mode='reflect')
    else:
        padded_arr = np.pad(arr, ((aug, aug), (aug, aug)), mode='reflect')

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


def gen_img_samples(src, chunk_size, *band_order):
    """

    Args:
        src: input image (rasterio object)
        chunk_size: image tile size
        *band_order: ignore

    Returns: generator object

    """
    subdiv = 2.0
    step = int(chunk_size / subdiv)
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
                 label_arr,
                 num_classes: int,
                 gpkg_name,
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
        label_arr: numpy array of label if available
        num_classes: number of classes
        gpkg_name: geo-package name if available
        model: model weights
        chunk_size: image tile size
        device: cuda/cpu device
        scale: scale range
        BGR_to_RGB: True/False
        tp_mem: memory temp file for saving numpy array to disk
        debug: True/False

    Returns:

    """
    xmin, ymin, xmax, ymax = (input_image.bounds.left,
                              input_image.bounds.bottom,
                              input_image.bounds.right,
                              input_image.bounds.top)
    xres, yres = (abs(input_image.transform.a), abs(input_image.transform.e))
    mx = chunk_size * xres
    my = chunk_size * yres
    padded = chunk_size * 2
    h = input_image.height
    w = input_image.width
    h_ = h + padded
    w_ = w + padded
    dist_samples = int(round(chunk_size * (1 - 1.0 / 2.0)))

    # switch to evaluate mode
    model.eval()

    # initialize test time augmentation
    transforms = tta.Compose([tta.HorizontalFlip(), ])
    # construct window for smoothing
    WINDOW_SPLINE_2D = _window_2D(window_size=padded, power=2.0)
    WINDOW_SPLINE_2D = torch.as_tensor(np.moveaxis(WINDOW_SPLINE_2D, 2, 0), ).type(torch.float)
    WINDOW_SPLINE_2D = WINDOW_SPLINE_2D.to(device)

    fp = np.memmap(tp_mem, dtype='float16', mode='w+', shape=(h_, w_, num_classes))
    sample = {'sat_img': None, 'map_img': None, 'metadata': None}
    cnt = 0
    img_gen = gen_img_samples(input_image, chunk_size)
    start_seg = time.time()
    for img in tqdm(img_gen, position=1, leave=False, desc='inferring on window slices'):
        row = img[1]
        col = img[2]
        sub_image = img[0]
        image_metadata = add_metadata_from_raster_to_sample(sat_img_arr=sub_image,
                                                            raster_handle=input_image,
                                                            meta_map={},
                                                            raster_info={})

        sample['metadata'] = image_metadata
        totensor_transform = augmentation.compose_transforms(param,
                                                             dataset="tst",
                                                             input_space=BGR_to_RGB,
                                                             scale=scale,
                                                             aug_type='totensor')
        sample['sat_img'] = sub_image
        sample = totensor_transform(sample)
        inputs = sample['sat_img'].unsqueeze_(0)
        inputs = inputs.to(device)
        if inputs.shape[1] == 4 and any("module.modelNIR" in s for s in model.state_dict().keys()):
            ############################
            # Test Implementation of the NIR
            ############################
            # Init NIR   TODO: make a proper way to read the NIR channel
            #                  and put an option to be able to give the idex of the NIR channel
            # Extract the NIR channel -> [batch size, H, W] since it's only one channel
            inputs_NIR = inputs[:, -1, ...]
            # add a channel to get the good size -> [:, 1, :, :]
            inputs_NIR.unsqueeze_(1)
            # take out the NIR channel and take only the RGB for the inputs
            inputs = inputs[:, :-1, ...]
            # Suggestion of implementation
            # inputs_NIR = data['NIR'].to(device)
            inputs = [inputs, inputs_NIR]
            # outputs = model(inputs, inputs_NIR)
            ############################
            # End of the test implementation module
            ############################
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
            deaugmented_output = F.softmax(deaugmented_output, dim=1).squeeze(dim=0)
            output_lst.append(deaugmented_output)
        outputs = torch.stack(output_lst)
        outputs = torch.mul(outputs, WINDOW_SPLINE_2D)
        outputs, _ = torch.max(outputs, dim=0)
        outputs = outputs.permute(1, 2, 0)
        outputs = outputs.reshape(padded, padded, num_classes).cpu().numpy().astype('float16')
        outputs = outputs[dist_samples:-dist_samples, dist_samples:-dist_samples, :]
        fp[row:row + chunk_size, col:col + chunk_size, :] = \
            fp[row:row + chunk_size, col:col + chunk_size, :] + outputs
        cnt += 1
    fp.flush()
    del fp

    fp = np.memmap(tp_mem, dtype='float16', mode='r', shape=(h_, w_, num_classes))
    subdiv = 2.0
    step = int(chunk_size / subdiv)
    pred_img = np.zeros((h_, w_), dtype=np.uint8)
    for row in tqdm(range(0, input_image.height, step), position=2, leave=False):
        for col in tqdm(range(0, input_image.width, step), position=3, leave=False):
            arr1 = fp[row:row + chunk_size, col:col + chunk_size, :] / (2 ** 2)
            arr1 = arr1.argmax(axis=-1).astype('uint8')
            pred_img[row:row + chunk_size, col:col + chunk_size] = arr1
    pred_img = pred_img[:h, :w]
    end_seg = time.time() - start_seg
    logging.info('Segmentation operation completed in {:.0f}m {:.0f}s'.format(end_seg // 60, end_seg % 60))

    if debug:
        logging.debug(f'Bin count of final output: {np.unique(pred_img, return_counts=True)}')
    gdf = None
    if label_arr is not None:
        start_seg_ = time.time()
        feature = defaultdict(list)
        cnt = 0
        for row in tqdm(range(0, h, chunk_size), position=2, leave=False):
            for col in tqdm(range(0, w, chunk_size), position=3, leave=False):
                label = label_arr[row:row + chunk_size, col:col + chunk_size]
                pred = pred_img[row:row + chunk_size, col:col + chunk_size]
                pixelMetrics = ComputePixelMetrics(label.flatten(), pred.flatten(), num_classes)
                eval = pixelMetrics.update(pixelMetrics.iou)
                feature['id_image'].append(gpkg_name)
                for c_num in range(num_classes):
                    feature['L_count_' + str(c_num)].append(int(np.count_nonzero(label == c_num)))
                    feature['P_count_' + str(c_num)].append(int(np.count_nonzero(pred == c_num)))
                    feature['IoU_' + str(c_num)].append(eval['iou_' + str(c_num)])
                feature['mIoU'].append(eval['macro_avg_iou'])
                x_1, y_1 = (xmin + (col * xres)), (ymax - (row * yres))
                x_2, y_2 = (xmin + ((col * xres) + mx)), y_1
                x_3, y_3 = x_2, (ymax - ((row * yres) + my))
                x_4, y_4 = x_1, y_3
                geom = Polygon([(x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4)])
                feature['geometry'].append(geom)
                feature['length'].append(geom.length)
                feature['pointx'].append(geom.centroid.x)
                feature['pointy'].append(geom.centroid.y)
                feature['area'].append(geom.area)
                cnt += 1
        gdf = gpd.GeoDataFrame(feature, crs=input_image.crs)
        gdf.to_crs(crs="EPSG:4326", inplace=True)
        end_seg_ = time.time() - start_seg_
        logging.info('Benchmark operation completed in {:.0f}m {:.0f}s'.format(end_seg_ // 60, end_seg_ % 60))
    input_image.close()
    return pred_img, gdf


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


def main(params: dict):
    """
    Identify the class to which each image belongs.
    :param params: (dict) Parameters found in the yaml config file.

    """
    since = time.time()

    # MANDATORY PARAMETERS
    img_dir_or_csv = get_key_def('img_dir_or_csv_file', params['inference'], expected_type=str)
    state_dict = get_key_def('state_dict_path', params['inference'])
    task = get_key_def('task', params['global'], expected_type=str)
    if task not in ['classification', 'segmentation']:
        raise ValueError(f'Task should be either "classification" or "segmentation". Got {task}')
    model_name = get_key_def('model_name', params['global'], expected_type=str).lower()
    num_classes = get_key_def('num_classes', params['global'], expected_type=int)
    num_bands = get_key_def('number_of_bands', params['global'], expected_type=int)
    chunk_size = get_key_def('chunk_size', params['inference'], default=512, expected_type=int)
    BGR_to_RGB = get_key_def('BGR_to_RGB', params['global'], expected_type=bool)

    # OPTIONAL PARAMETERS
    dontcare_val = get_key_def("ignore_index", params["training"], default=-1, expected_type=int)
    num_devices = get_key_def('num_gpus', params['global'], default=0, expected_type=int)
    default_max_used_ram = 25
    max_used_ram = get_key_def('max_used_ram', params['global'], default=default_max_used_ram, expected_type=int)
    max_used_perc = get_key_def('max_used_perc', params['global'], default=25, expected_type=int)
    scale = get_key_def('scale_data', params['global'], default=[0, 1], expected_type=List)
    debug = get_key_def('debug_mode', params['global'], default=False, expected_type=bool)
    raster_to_vec = get_key_def('ras2vec', params['inference'], False)

    # benchmark (ie when gkpgs are inputted along with imagery)
    dontcare = get_key_def("ignore_index", params["training"], -1)
    targ_ids = get_key_def('target_ids', params['sample'], None, expected_type=List)

    # SETTING OUTPUT DIRECTORY
    working_folder = Path(params['inference']['state_dict_path']).parent.joinpath(f'inference_{num_bands}bands')
    Path.mkdir(working_folder, parents=True, exist_ok=True)

    # mlflow logging
    mlflow_uri = get_key_def('mlflow_uri', params['global'], default=None, expected_type=str)
    if mlflow_uri and not Path(mlflow_uri).is_dir():
        warnings.warn(f'Mlflow uri path is not valid: {mlflow_uri}')
        mlflow_uri = None
    # SETUP LOGGING
    import logging.config  # See: https://docs.python.org/2.4/lib/logging-config-fileformat.html
    if mlflow_uri:
        log_config_path = Path('utils/logging.conf').absolute()
        logfile = f'{working_folder}/info.log'
        logfile_debug = f'{working_folder}/debug.log'
        console_level_logging = 'INFO' if not debug else 'DEBUG'
        logging.config.fileConfig(log_config_path, defaults={'logfilename': logfile,
                                                             'logfilename_debug': logfile_debug,
                                                             'console_level': console_level_logging})

        # import only if mlflow uri is set
        from mlflow import log_params, set_tracking_uri, set_experiment, start_run, log_artifact, log_metrics
        if not Path(mlflow_uri).is_dir():
            logging.warning(f"Couldn't locate mlflow uri directory {mlflow_uri}. Directory will be created.")
            Path(mlflow_uri).mkdir()
        set_tracking_uri(mlflow_uri)
        exp_name = get_key_def('mlflow_experiment_name', params['global'], default='gdl-inference', expected_type=str)
        set_experiment(f'{exp_name}/{working_folder.name}')
        run_name = get_key_def('mlflow_run_name', params['global'], default='gdl', expected_type=str)
        start_run(run_name=run_name)
        log_params(params['global'])
        log_params(params['inference'])
    else:
        # set a console logger as default
        logging.basicConfig(level=logging.DEBUG)
        logging.info('No logging folder set for mlflow. Logging will be limited to console')

    if debug:
        logging.warning(f'Debug mode activated. Some debug features may mobilize extra disk space and '
                        f'cause delays in execution.')

    # Assert that all items in target_ids are integers (ex.: to benchmark single-class model with multi-class labels)
    if targ_ids:
        for item in targ_ids:
            if not isinstance(item, int):
                raise ValueError(f'Target id "{item}" in target_ids is {type(item)}, expected int.')

    logging.info(f'Inferences will be saved to: {working_folder}\n\n')
    if not (0 <= max_used_ram <= 100):
        logging.warning(f'Max used ram parameter should be a percentage. Got {max_used_ram}. '
                        f'Will set default value of {default_max_used_ram} %')
        max_used_ram = default_max_used_ram

    # AWS
    bucket = None
    bucket_file_cache = []
    bucket_name = get_key_def('bucket_name', params['global'])

    # list of GPU devices that are available and unused. If no GPUs, returns empty dict
    gpu_devices_dict = get_device_ids(num_devices,
                                      max_used_ram_perc=max_used_ram,
                                      max_used_perc=max_used_perc)
    if gpu_devices_dict:
        logging.info(f"Number of cuda devices requested: {num_devices}. Cuda devices available: {gpu_devices_dict}. "
                     f"Using {list(gpu_devices_dict.keys())[0]}\n\n")
        device = torch.device(f'cuda:{list(range(len(gpu_devices_dict.keys())))[0]}')
    else:
        logging.warning(f"No Cuda device available. This process will only run on CPU")
        device = torch.device('cpu')

    # CONFIGURE MODEL
    num_classes_backgr = add_background_to_num_class(task, num_classes)
    model, loaded_checkpoint, model_name = net(model_name=model_name,
                                               num_bands=num_bands,
                                               num_channels=num_classes_backgr,
                                               dontcare_val=dontcare_val,
                                               num_devices=1,
                                               net_params=params,
                                               inference_state_dict=state_dict)
    try:
        model.to(device)
    except RuntimeError:
        logging.info(f"Unable to use device 0")
        device = torch.device(f'cuda' if gpu_devices_dict else 'cpu')
        model.to(device)

    # CREATE LIST OF INPUT IMAGES FOR INFERENCE
    list_img = list_input_images(img_dir_or_csv, bucket_name, glob_patterns=["*.tif", "*.TIF"])

    # VALIDATION: anticipate problems with imagery and label (if provided) before entering main for loop
    valid_gpkg_set = set()
    for info in tqdm(list_img, desc='Validating imagery'):
        # validate_raster(info['tif'], num_bands, meta_map)
        if 'gpkg' in info.keys() and info['gpkg'] and info['gpkg'] not in valid_gpkg_set:
            validate_num_classes(vector_file=info['gpkg'],
                                 num_classes=num_classes,
                                 attribute_name=info['attribute_name'],
                                 ignore_index=dontcare,
                                 target_ids=targ_ids)
            assert_crs_match(info['tif'], info['gpkg'])
            valid_gpkg_set.add(info['gpkg'])

    logging.info('Successfully validated imagery')
    if valid_gpkg_set:
        logging.info('Successfully validated label data for benchmarking')

    if task == 'classification':
        classifier(params, list_img, model, device,
                   working_folder)  # FIXME: why don't we load from checkpoint in classification?

    elif task == 'segmentation':
        gdf_ = []
        gpkg_name_ = []

        # TODO: Add verifications?
        if bucket:
            bucket.download_file(loaded_checkpoint, "saved_model.pth.tar")  # TODO: is this still valid?
            model, _ = load_from_checkpoint("saved_model.pth.tar", model)
        else:
            model, _ = load_from_checkpoint(loaded_checkpoint, model)
        # LOOP THROUGH LIST OF INPUT IMAGES
        for info in tqdm(list_img, desc='Inferring from images', position=0, leave=True):
            with start_run(run_name=Path(info['tif']).name, nested=True):
                img_name = Path(info['tif']).name
                local_gpkg = Path(info['gpkg']) if 'gpkg' in info.keys() and info['gpkg'] else None
                gpkg_name = local_gpkg.stem if local_gpkg else None
                if bucket:
                    local_img = f"Images/{img_name}"
                    bucket.download_file(info['tif'], local_img)
                    inference_image = f"Classified_Images/{img_name.split('.')[0]}_inference.tif"
                    if info['meta']:
                        if info['meta'] not in bucket_file_cache:
                            bucket_file_cache.append(info['meta'])
                            bucket.download_file(info['meta'], info['meta'].split('/')[-1])
                        info['meta'] = info['meta'].split('/')[-1]
                else:  # FIXME: else statement should support img['meta'] integration as well.
                    local_img = Path(info['tif'])
                    Path.mkdir(working_folder.joinpath(local_img.parent.name), parents=True, exist_ok=True)
                    inference_image = working_folder.joinpath(local_img.parent.name,
                                                              f"{img_name.split('.')[0]}_inference.tif")
                temp_file = working_folder.joinpath(local_img.parent.name, f"{img_name.split('.')[0]}.dat")
                raster = rasterio.open(local_img, 'r')
                logging.info(f'Reading original image: {raster.name}')
                inf_meta = raster.meta
                label = None
                if local_gpkg:
                    logging.info(f'Burning label as raster: {local_gpkg}')
                    local_img = clip_raster_with_gpkg(raster, local_gpkg)
                    raster.close()
                    raster = rasterio.open(local_img, 'r')
                    logging.info(f'Reading clipped image: {raster.name}')
                    inf_meta = raster.meta
                    label = vector_to_raster(vector_file=local_gpkg,
                                             input_image=raster,
                                             out_shape=(inf_meta['height'], inf_meta['width']),
                                             attribute_name=info['attribute_name'],
                                             fill=0,  # background value in rasterized vector.
                                             target_ids=targ_ids)
                    if debug:
                        logging.debug(f'Unique values in loaded label as raster: {np.unique(label)}\n'
                                      f'Shape of label as raster: {label.shape}')
                pred, gdf = segmentation(param=params,
                                         input_image=raster,
                                         label_arr=label,
                                         num_classes=num_classes_backgr,
                                         gpkg_name=gpkg_name,
                                         model=model,
                                         chunk_size=chunk_size,
                                         device=device,
                                         scale=scale,
                                         BGR_to_RGB=BGR_to_RGB,
                                         tp_mem=temp_file,
                                         debug=debug)
                if gdf is not None:
                    gdf_.append(gdf)
                    gpkg_name_.append(gpkg_name)
                if local_gpkg:
                    pixelMetrics = ComputePixelMetrics(label, pred, num_classes_backgr)
                    log_metrics(pixelMetrics.update(pixelMetrics.iou))
                    log_metrics(pixelMetrics.update(pixelMetrics.dice))
                pred = pred[np.newaxis, :, :].astype(np.uint8)
                inf_meta.update({"driver": "GTiff",
                                 "height": pred.shape[1],
                                 "width": pred.shape[2],
                                 "count": pred.shape[0],
                                 "dtype": 'uint8',
                                 "compress": 'lzw'})
                logging.info(f'Successfully inferred on {img_name}\nWriting to file: {inference_image}')
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

        if len(gdf_) >= 1:
            if not len(gdf_) == len(gpkg_name_):
                raise ValueError('benchmarking unable to complete')
            all_gdf = pd.concat(gdf_)  # Concatenate all geo data frame into one geo data frame
            all_gdf.reset_index(drop=True, inplace=True)
            gdf_x = gpd.GeoDataFrame(all_gdf)
            bench_gpkg = working_folder / "benchmark.gpkg"
            gdf_x.to_file(bench_gpkg, driver="GPKG", index=False)
            logging.info(f'Successfully wrote benchmark geopackage to: {bench_gpkg}')
        # log_artifact(working_folder)
    time_elapsed = time.time() - since
    logging.info('Inference Script completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    print('\n\nStart:\n\n')
    parser = argparse.ArgumentParser(usage="%(prog)s [-h] [-p YAML] [-i MODEL IMAGE] ",
                                     description='Inference and Benchmark on images using trained model')

    parser.add_argument('-p', '--param', metavar='yaml_file', nargs=1,
                        help='Path to parameters stored in yaml')
    parser.add_argument('-i', '--input', metavar='model_pth img_dir', nargs=2,
                        help='model_path and image_dir')
    args = parser.parse_args()

    # if a yaml is inputted, get those parameters and get model state_dict to overwrite global parameters afterwards
    if args.param:
        input_params = read_parameters(args.param[0])
        model_ckpt = get_key_def('state_dict_path', input_params['inference'], expected_type=str)
        # load checkpoint
        checkpoint = load_checkpoint(model_ckpt)
        if 'params' in checkpoint.keys():
            params = checkpoint['params']
            # overwrite with inputted parameters
            compare_config_yamls(yaml1=params, yaml2=input_params, update_yaml1=True)
        else:
            warnings.warn('No parameters found in checkpoint. Defaulting to parameters from inputted yaml.'
                          'Use GDL version 1.3 or more.')
            params = input_params
        del checkpoint
        del input_params

    # elif input is a model checkpoint and an image directory, we'll rely on the yaml saved inside the model (pth.tar)
    elif args.input:
        if is_url(args.input[0]):
            url = args.input[0]
            checkpoint_path = checkpoint_url_download(url)
            args.input[0] = checkpoint_path
        else:
            checkpoint_path = Path(args.input[0])
        image = args.input[1]
        checkpoint = load_checkpoint(checkpoint_path)
        if 'params' not in checkpoint.keys():
            raise KeyError('No parameters found in checkpoint. Use GDL version 1.3 or more.')
        else:
            params = checkpoint['params']
            params['inference']['state_dict_path'] = args.input[0]
            params['inference']['img_dir_or_csv_file'] = args.input[1]

        del checkpoint
    else:
        print('use the help [-h] option for correct usage')
        raise SystemExit

    main(params)

import gc
import logging
import warnings
from math import sqrt
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

from tqdm import tqdm
from shapely.geometry import Polygon, box
from pathlib import Path

from utils.metrics import ComputePixelMetrics
from models.model_choice import net, load_checkpoint
from utils import augmentation
from utils.geoutils import vector_to_raster
from utils.utils import load_from_checkpoint, get_device_ids, get_key_def, \
    list_input_images, pad, add_metadata_from_raster_to_sample, _window_2D, defaults_from_params, compare_config_yamls
from utils.readers import read_parameters, image_reader_as_array
from utils.verifications import add_background_to_num_class, validate_raster, validate_num_classes, assert_crs_match

try:
    import boto3
except ModuleNotFoundError:
    pass

logging.getLogger(__name__)


def calc_inference_chunk_size(gpu_devices_dict: dict, max_pix_per_mb_gpu: int = 350):
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


@torch.no_grad()
def segmentation(img_array,
                 input_image,
                 label_arr,
                 num_classes: int,
                 gpkg_name,
                 model,
                 chunk_size: int,
                 num_bands: int,
                 device,
                 scale: List,
                 BGR_to_RGB: bool,
                 debug=False):

    # switch to evaluate mode
    model.eval()
    transforms = tta.Compose([tta.HorizontalFlip(), ])

    WINDOW_SPLINE_2D = _window_2D(window_size=chunk_size, power=2.0)
    WINDOW_SPLINE_2D = torch.as_tensor(np.moveaxis(WINDOW_SPLINE_2D, 2, 0), ).type(torch.float)
    WINDOW_SPLINE_2D = WINDOW_SPLINE_2D.to(device)


    metadata = add_metadata_from_raster_to_sample(img_array,
                                                  input_image,
                                                  meta_map=None,
                                                  raster_info=None)

    xmin, ymin, xmax, ymax = (input_image.bounds.left,
                              input_image.bounds.bottom,
                              input_image.bounds.right,
                              input_image.bounds.top)

    xres, yres = (abs(input_image.transform.a), abs(input_image.transform.e))
    h, w, bands = img_array.shape
    if num_bands < bands and debug:
        logging.warning(f'Expected {bands} bands, for {num_bands}. Slicing off excess bands. Problems may occur')
        img_array = img_array[:, :, :num_bands]
    padding = int(round(chunk_size * (1 - 1.0 / 2.0)))
    step = int(chunk_size / 2.0)
    img_array = pad(img_array, padding=padding, fill=np.nan)
    h_, w_, bands_ = img_array.shape
    mx = chunk_size * xres
    my = chunk_size * yres
    X_points = np.arange(0, w_ - chunk_size + 1, step)
    Y_points = np.arange(0, h_ - chunk_size + 1, step)
    pred_img = np.empty((h_, w_), dtype=np.uint8)
    sample = {'sat_img': None, 'map_img': None, 'metadata': None}

    for row in tqdm(Y_points, position=1, leave=False, desc=f'Inferring rows (chunk size: {chunk_size})'):
        for col in tqdm(X_points, position=2, leave=False, desc='Inferring columns'):
            sample['metadata'] = metadata
            totensor_transform = augmentation.compose_transforms(params=params,
                                                                 dataset="tst",
                                                                 input_space=BGR_to_RGB,
                                                                 scale=scale,
                                                                 aug_type='totensor')
            sample['sat_img'] = img_array[row:row + chunk_size, col:col + chunk_size, :]
            logging.debug(f"Sample shape: {sample['sat_img'].shape}")
            sample = totensor_transform(sample)
            inputs = sample['sat_img'].unsqueeze_(0)
            inputs = inputs.to(device)
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
            outputs = outputs.permute(1, 2, 0).argmax(dim=-1)
            outputs = outputs.reshape(chunk_size, chunk_size).cpu().numpy()
            if debug:
                logging.debug(f'Bin count of final output: {np.unique(outputs, return_counts=True)}')
            pred_img[row:row + chunk_size, col:col + chunk_size] = outputs
    pred_img = pred_img[padding:-padding, padding:-padding]
    gdf = None
    if label_arr is not None:
        feature = defaultdict(list)
        cnt = 0
        for row in tqdm(range(0, h, chunk_size), position=1, leave=False, desc='Inferring rows'):
            for col in tqdm(range(0, w, chunk_size), position=2, leave=False, desc='Inferring columns'):
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
    weights_file_name = get_key_def('state_dict_path', params['inference'],
                                    defaults_from_params(params, 'state_dict_path'))
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
    
    # mlflow logging
    exp_name = get_key_def('mlflow_experiment_name', params['global'], default='gdl-inference', expected_type=str)
    mlflow_uri = get_key_def('mlflow_uri', params['global'], default=None, expected_type=str)
    if mlflow_uri and not Path(mlflow_uri).is_dir():
        warnings.warn(f'Mlflow uri path is not valid: {mlflow_uri}')
        mlflow_uri = None
    
    # MANDATORY PARAMETERS
    # sets default csv path
    default_csv_file = Path(get_key_def('preprocessing_path', params['global'], ''),
                            exp_name, f"inference_sem_seg_{exp_name}.csv")
    img_dir_or_csv = get_key_def('img_dir_or_csv_file', params['inference'], default_csv_file, expected_type=str)
    if not (Path(img_dir_or_csv).is_dir() or Path(img_dir_or_csv).suffix == '.csv'):
        raise FileNotFoundError(f'Couldn\'t locate .csv file or directory "{img_dir_or_csv}" '
                                f'containing imagery for inference')
    state_dict = get_key_def('state_dict_path', params['inference'],
                                  defaults_from_params(params, 'state_dict_path'), expected_type=str)
    if not Path(state_dict).is_file():
        raise FileNotFoundError(f'Couldn\'t locate state_dict of model "{state_dict}" to be used for inference')
    task = get_key_def('task', params['global'], expected_type=str)
    if task not in ['classification', 'segmentation']:
        raise ValueError(f'Task should be either "classification" or "segmentation". Got {task}')
    model_name = get_key_def('model_name', params['global'], expected_type=str).lower()
    num_classes = get_key_def('num_classes', params['global'], expected_type=int)
    num_bands = get_key_def('number_of_bands', params['global'], expected_type=int)
    BGR_to_RGB = get_key_def('BGR_to_RGB', params['global'], expected_type=bool)

    # OPTIONAL PARAMETERS
    dontcare_val = get_key_def("ignore_index", params["training"], default=-1, expected_type=int)
    num_devices = get_key_def('num_gpus', params['global'], default=0, expected_type=int)
    default_max_used_ram = 25
    max_used_ram = get_key_def('max_used_ram', params['global'], default=default_max_used_ram, expected_type=int)
    max_used_perc = get_key_def('max_used_perc', params['global'], default=25, expected_type=int)
    meta_map = get_key_def('meta_map', params['global'], default={})
    scale = get_key_def('scale_data', params['global'], default=[0, 1], expected_type=List)
    debug = get_key_def('debug_mode', params['global'], default=False, expected_type=bool)

    # benchmark (ie when gkpgs are inputted along with imagery)
    dontcare = get_key_def("ignore_index", params["training"], -1)
    targ_ids = get_key_def('target_ids', params['sample'], None, expected_type=List)

    # SETTING OUTPUT DIRECTORY
    working_folder = Path(params['inference']['state_dict_path']).parent.joinpath(f'inference_{num_bands}bands')
    Path.mkdir(working_folder, parents=True, exist_ok=True)

    # SETUP DEFAULT LOGGING
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
    # TODO: test this thumbrule on different GPUs
    if gpu_devices_dict:
        chunk_size = calc_inference_chunk_size(gpu_devices_dict=gpu_devices_dict, max_pix_per_mb_gpu=350)
    else:
        chunk_size = get_key_def('chunk_size', params['inference'], default=512, expected_type=int)
    device = torch.device(f'cuda:0' if gpu_devices_dict else 'cpu')

    if gpu_devices_dict:
        logging.info(f"Number of cuda devices requested: {num_devices}. Cuda devices available: {gpu_devices_dict}. "
                     f"Using {list(gpu_devices_dict.keys())[0]}\n\n")
    else:
        logging.warning(f"No Cuda device available. This process will only run on CPU")

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
        logging.info(f"Unable to use device. Trying device 0")
        device = torch.device(f'cuda:0' if gpu_devices_dict else 'cpu')
        model.to(device)

    # CREATE LIST OF INPUT IMAGES FOR INFERENCE
    list_img = list_input_images(img_dir_or_csv, bucket_name, glob_patterns=["*.tif", "*.TIF"])

    # VALIDATION: anticipate problems with imagery and label (if provided) before entering main for loop
    valid_gpkg_set = set()
    for info in tqdm(list_img, desc='Validating imagery'):
        validate_raster(info['tif'], num_bands, meta_map)
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
        for info in tqdm(list_img, desc='Inferring from images', position=0):
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
            with rasterio.open(local_img, 'r') as raster:
                logging.info(f'Reading image as array: {raster.name}')
                img_array, raster, _ = image_reader_as_array(input_image=raster, clip_gpkg=local_gpkg)
                if debug:
                    logging.debug(f'Unique values in loaded raster: {np.unique(img_array)}\n'
                                  f'Shape of raster: {img_array.shape}')
                inf_meta = raster.meta
                label = None
                if local_gpkg:
                    logging.info(f'Burning label as raster: {local_gpkg}')
                    label = vector_to_raster(vector_file=local_gpkg,
                                             input_image=raster,
                                             out_shape=(inf_meta['height'], inf_meta['width']),
                                             attribute_name=info['attribute_name'],
                                             fill=0,  # background value in rasterized vector.
                                             target_ids=targ_ids)
                    if debug:
                        logging.debug(f'Unique values in loaded label as raster: {np.unique(label)}\n'
                                      f'Shape of label as raster: {label.shape}')

                pred, gdf = segmentation(img_array=img_array,
                                         input_image=raster,
                                         label_arr=label,
                                         num_classes=num_classes_backgr,
                                         gpkg_name=gpkg_name,
                                         model=model,
                                         chunk_size=chunk_size,
                                         num_bands=num_bands,
                                         device=device,
                                         scale=scale,
                                         BGR_to_RGB=BGR_to_RGB,
                                         debug=debug)
                if gdf is not None:
                    gdf_.append(gdf)
                    gpkg_name_.append(gpkg_name)
                if local_gpkg:
                    with start_run(run_name=img_name, nested=True):
                        pixelMetrics= ComputePixelMetrics(label, pred, num_classes_backgr)
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
    logging.info('Inference and Benchmarking completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


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
        yaml_file = args.param[0]
        input_params = read_parameters(yaml_file)
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
        params['self'] = {'config_file': yaml_file}
        del checkpoint
        del input_params

    # elif input is a model checkpoint and an image directory, we'll rely on the yaml saved inside the model (pth.tar)
    elif args.input:
        model_ckpt = Path(args.input[0])
        image = args.input[1]
        # load checkpoint
        checkpoint = load_checkpoint(model_ckpt)
        if 'params' not in checkpoint.keys():
            raise KeyError('No parameters found in checkpoint. Use GDL version 1.3 or more.')
        else:
            # set parameters for inference from those contained in checkpoint.pth.tar
            params = checkpoint['params']
            del checkpoint
        # overwrite with inputted parameters
        params['inference']['state_dict_path'] = args.input[0]
        params['inference']['img_dir_or_csv_file'] = args.input[1]
    else:
        print('use the help [-h] option for correct usage')
        raise SystemExit

    main(params)

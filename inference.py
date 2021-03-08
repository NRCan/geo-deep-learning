import torch
import torch.nn.functional as F
# import torch should be first. Unclear issue, mentionned here: https://github.com/pytorch/pytorch/issues/2083
import numpy as np
import os
import csv
import time
import argparse
import heapq
import rasterio
from PIL import Image
import torchvision
import math
from collections import OrderedDict
import warnings

from tqdm import tqdm
from pathlib import Path
from utils.metrics import ComputePixelMetrics
from models.model_choice import net
from utils import augmentation
from utils.geoutils import vector_to_raster
from utils.utils import load_from_checkpoint, get_device_ids, get_key_def, \
list_input_images, pad, pad_diff, ind2rgb, add_metadata_from_raster_to_sample, _window_2D
from utils.readers import read_parameters, image_reader_as_array
from utils.verifications import add_background_to_num_class
from mlflow import log_params, set_tracking_uri, set_experiment, start_run, log_artifact, log_metrics

try:
    import boto3
except ModuleNotFoundError:
    pass

colors = {0: [0, 0, 0],  # Background
          1: [27, 120, 55],  # Vegetation
          2: [116, 173, 209], # Hydro
          3: [223, 194, 125],  # Roads
          4: [150, 150, 150]}  # Buildings     

@torch.no_grad()
def segmentation_with_smoothing(raster, clip_gpkg, model, sample_size, overlap, num_bands, device):
    # switch to evaluate mode
    model.eval()
    img_array, input_image, dataset_nodata = image_reader_as_array(input_image=raster,
                                                                   clip_gpkg=clip_gpkg)
    metadata = add_metadata_from_raster_to_sample(img_array,
                                                    input_image,
                                                    meta_map=None,
                                                    raster_info=None)
    h, w, bands = img_array.shape
    assert num_bands <= bands, f"Num of specified bands is not compatible with image shape {img_array.shape}"
    if num_bands < bands:
       img_array = img_array[:, :, :num_bands]

    padding = int(round(sample_size * (1 - 1.0/overlap)))
    padded_img = pad(img_array, padding=padding, fill=0)
    WINDOW_SPLINE_2D = _window_2D(window_size=sample_size, power=1)
    WINDOW_SPLINE_2D = np.moveaxis(WINDOW_SPLINE_2D, 2, 0)
    step = int(sample_size/overlap)
    h_, w_ = padded_img.shape[:2]
    pred_img = np.empty((h_, w_), dtype=np.uint8)
    for row in tqdm(range(0, h_ - sample_size + 1, step), position=1, leave=False, desc='Inferring rows'):
        with tqdm(range(0, w_ - sample_size + 1, step), position=2, leave=False, desc='Inferring columns') as _tqdm:
            for col in _tqdm:
                sample = {'sat_img': None, 'metadata': None}
                sample['metadata'] = metadata
                totensor_transform = augmentation.compose_transforms(
                    params, dataset="tst", type='totensor'
                )
                sub_images = padded_img[row:row+sample_size, col:col+sample_size, :]
                sample['sat_img'] = sub_images
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
                    inputs_NIR = inputs[:,-1,...]
                    # add a channel to get the good size -> [:, 1, :, :]
                    inputs_NIR.unsqueeze_(1)
                    # take out the NIR channel and take only the RGB for the inputs
                    inputs = inputs[:,:-1, ...]
                    # Suggestion of implementation
                    #inputs_NIR = data['NIR'].to(device)
                    inputs = [inputs, inputs_NIR]
                    #outputs = model(inputs, inputs_NIR)
                    ############################
                    # End of the test implementation module
                    ############################

                outputs = model(inputs)
                # torchvision models give output in 'out' key.
                # May cause problems in future versions of torchvision.
                if isinstance(outputs, OrderedDict) and 'out' in outputs.keys():
                    outputs = outputs['out']
                outputs = F.softmax(outputs, dim=1).squeeze(dim=0).cpu().numpy()
                outputs = WINDOW_SPLINE_2D * outputs
                outputs = outputs.argmax(axis=0)
                pred_img[row:row + sample_size, col:col + sample_size] = outputs
    pred_img = pred_img[padding:-padding, padding:-padding]
    return pred_img[:h, :w]

@torch.no_grad()
def segmentation(raster, clip_gpkg, model, sample_size, num_bands, device):
    # switch to evaluate mode
    model.eval()
    img_array, input_image, dataset_nodata = image_reader_as_array(input_image=raster,
                                                                   clip_gpkg=clip_gpkg)
    metadata = add_metadata_from_raster_to_sample(img_array,
                                                    input_image,
                                                    meta_map=None,
                                                    raster_info=None)
    h, w, bands = img_array.shape
    assert num_bands <= bands, f"Num of specified bands is not compatible with image shape {img_array.shape}"
    if num_bands < bands:
       img_array = img_array[:, :, :num_bands]
    h_ = sample_size * math.ceil(h / sample_size)
    w_ = sample_size * math.ceil(w / sample_size)
    pred_img = np.empty((h_, w_), dtype=np.uint8)
    for row in tqdm(range(0, h, sample_size), position=1, leave=False, desc='Inferring rows'):
        with tqdm(range(0, w, sample_size), position=2, leave=False, desc='Inferring columns') as _tqdm:
            for column in _tqdm:
                sample = {'sat_img': None, 'metadata': None}
                sample['metadata'] = metadata
                totensor_transform = augmentation.compose_transforms(params, dataset="tst", type='totensor')
                sub_images = img_array[row:row + sample_size, column:column + sample_size, :]
                sub_images_row = sub_images.shape[0]
                sub_images_col = sub_images.shape[1]

                if sub_images_row < sample_size or sub_images_col < sample_size:
                    padding = pad_diff(actual_height=sub_images_row,
                                        actual_width=sub_images_col,
                                        desired_shape=sample_size)
                    sub_images = pad(sub_images, padding, fill=0) # FIXME combine pad and pad_diff into one function
                sample['sat_img'] = sub_images
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
                    inputs_NIR = inputs[:,-1,...]
                    # add a channel to get the good size -> [:, 1, :, :]
                    inputs_NIR.unsqueeze_(1)
                    # take out the NIR channel and take only the RGB for the inputs
                    inputs = inputs[:,:-1, ...]
                    # Suggestion of implementation
                    #inputs_NIR = data['NIR'].to(device)
                    inputs = [inputs, inputs_NIR]
                    #outputs = model(inputs, inputs_NIR)
                    ############################
                    # End of the test implementation module
                    ############################

                outputs = model(inputs)
                # torchvision models give output in 'out' key. May cause problems in future versions of torchvision.
                if isinstance(outputs, OrderedDict) and 'out' in outputs.keys():
                    outputs = outputs['out']
                outputs = F.softmax(outputs, dim=1).argmax(dim=1).squeeze(dim=0).cpu().numpy()

                pred_img[row:row + sample_size, column:column + sample_size] = outputs

    return pred_img[:h, :w]

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
        print(f"Image {img_name} classified as {classes[0][predicted]}")
        print('Top 5 classes:')
        for i in range(0, 5):
            print(f"\t{classes[0][top5_loc[i]]} : {top5[i]}")
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
    # SET BASIC VARIABLES AND PATHS
    since = time.time()
    task = params['global']['task']
    img_dir_or_csv = params['inference']['img_dir_or_csv_file']
    chunk_size = get_key_def('chunk_size', params['inference'], 512)
    prediction_with_smoothing = get_key_def('smooth_prediction', params['inference'], False)
    overlap = get_key_def('overlap', params['inference'], 2)
    num_classes = params['global']['num_classes']
    num_classes_corrected = add_background_to_num_class(task, num_classes)
    num_bands = params['global']['number_of_bands']
    working_folder = Path(params['inference']['state_dict_path']).parent.joinpath(f'inference_{num_bands}bands')
    num_devices = params['global']['num_gpus'] if params['global']['num_gpus'] else 0
    Path.mkdir(working_folder, parents=True, exist_ok=True)
    print(f'Inferences will be saved to: {working_folder}\n\n')

    bucket = None
    bucket_file_cache = []
    bucket_name = get_key_def('bucket_name', params['global'])

    # list of GPU devices that are available and unused. If no GPUs, returns empty list
    lst_device_ids = get_device_ids(num_devices) if torch.cuda.is_available() else []
    device = torch.device(f'cuda:{lst_device_ids[0]}' if torch.cuda.is_available() and lst_device_ids else 'cpu')

    if lst_device_ids:
        print(f"Number of cuda devices requested: {num_devices}. Cuda devices available: {lst_device_ids}. Using {lst_device_ids[0]}\n\n")
    else:
        warnings.warn(f"No Cuda device available. This process will only run on CPU")

    # CONFIGURE MODEL
    model, state_dict_path, model_name = net(params, num_channels=num_classes_corrected, inference=True)

    try:
        model.to(device)
    except RuntimeError:
        print(f"Unable to use device. Trying device 0")
        device = torch.device(f'cuda:0' if torch.cuda.is_available() and lst_device_ids else 'cpu')
        model.to(device)

    # mlflow tracking path + parameters logging
    set_tracking_uri(get_key_def('mlflow_uri', params['global'], default="./mlruns"))
    set_experiment('gdl-benchmarking/' + working_folder.name)
    log_params(params['global'])
    log_params(params['inference'])

    # CREATE LIST OF INPUT IMAGES FOR INFERENCE
    list_img = list_input_images(img_dir_or_csv, bucket_name, glob_patterns=["*.tif", "*.TIF"])

    if task == 'classification':
        # FIXME: why don't we load from checkpoint in classification?
        classifier(params, list_img, model, device, working_folder)

    elif task == 'segmentation':
        # TODO: Add verifications?
        if bucket:
            bucket.download_file(state_dict_path, "saved_model.pth.tar")  # TODO: is this still valid?
            model, _ = load_from_checkpoint("saved_model.pth.tar", model)
        else:
            model, _ = load_from_checkpoint(state_dict_path, model)
        # LOOP THROUGH LIST OF INPUT IMAGES
        with tqdm(list_img, desc='image list', position=0) as _tqdm:
            for info in _tqdm:
                img_name = Path(info['tif']).name
                local_gpkg = info['gpkg'] if 'gpkg' in info.keys() else None
                if local_gpkg:
                    local_gpkg = Path(local_gpkg)
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
                    inference_image = working_folder.joinpath(f"{img_name.split('.')[0]}_inference.tif")
                    print(inference_image)
                assert local_img.is_file(), f"Could not locate raster file at {local_img}"
                with rasterio.open(local_img, 'r') as raster:
                    inf_meta= raster.meta
                    if prediction_with_smoothing:
                        print('Smoothening Predictions with 2D interpolation')
                        pred = segmentation_with_smoothing(
                            raster, local_gpkg, model, chunk_size, overlap, num_bands, device
                        )
                    else:
                        pred = segmentation(raster, local_gpkg, model, chunk_size, num_bands, device)
                    if local_gpkg:
                        assert local_gpkg.is_file(), f"Could not locate gkpg file at {local_gpkg}"
                        label = vector_to_raster(vector_file=local_gpkg,
                                                 input_image=raster,
                                                 out_shape=pred.shape[:2],
                                                 attribute_name=info['attribute_name'],
                                                 fill=0)  # background value in rasterized vector.
                        with start_run(run_name=img_name, nested=True):
                            pixelMetrics= ComputePixelMetrics(label, pred, num_classes_corrected)
                            log_metrics(pixelMetrics.update(pixelMetrics.jaccard))
                            log_metrics(pixelMetrics.update(pixelMetrics.dice))
                            log_metrics(pixelMetrics.update(pixelMetrics.accuracy))
                            log_metrics(pixelMetrics.update(pixelMetrics.precision))
                            log_metrics(pixelMetrics.update(pixelMetrics.recall))
                            log_metrics(pixelMetrics.update(pixelMetrics.matthews))

                        label_classes = np.unique(label)
                        assert len(colors) >= len(label_classes), f'Not enough colors and class names for number of classes in output'
                        # FIXME: color mapping scheme is hardcoded for now because of memory constraint; To be fixed.
                        label_rgb = ind2rgb(label, colors)
                        pred_rgb = ind2rgb(pred, colors)
                        Image.fromarray((label_rgb).astype(np.uint8), mode='RGB').save(os.path.join(working_folder, 'label_rgb_' + inference_image.stem + '.png'))
                        Image.fromarray((pred_rgb).astype(np.uint8), mode='RGB').save(os.path.join(working_folder, 'pred_rgb_' + inference_image.stem + '.png'))
                        del label_rgb, pred_rgb
                    pred = pred[np.newaxis, :, :]
                    inf_meta.update({"driver": "GTiff",
                                     "height": pred.shape[1],
                                     "width": pred.shape[2],
                                     "count": pred.shape[0],
                                     "dtype": 'uint8'})

                    with rasterio.open(inference_image, 'w+', **inf_meta) as dest:
                        dest.write(pred)
        log_artifact(working_folder)
    time_elapsed = time.time() - since
    print('Inference and Benchmarking completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
    print('\n\nStart:\n\n')
    parser = argparse.ArgumentParser(description='Inference and Benchmark on images using trained model')
    parser.add_argument('param_file', metavar='file',
                        help='Path to parameters stored in yaml')
    args = parser.parse_args()
    params = read_parameters(args.param_file)

    main(params)

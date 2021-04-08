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
import ttach as tta
from collections import OrderedDict, defaultdict
import warnings
import pandas as pd
import geopandas as gpd

from tqdm import tqdm
from shapely.geometry import Polygon
from pathlib import Path
from utils.metrics import ComputePixelMetrics
from models.model_choice import net, load_checkpoint
from utils import augmentation
from utils.geoutils import vector_to_raster
from utils.utils import load_from_checkpoint, get_device_ids, get_key_def, \
    list_input_images, pad, add_metadata_from_raster_to_sample, _window_2D
from utils.readers import read_parameters, image_reader_as_array
from utils.verifications import add_background_to_num_class
from mlflow import log_params, set_tracking_uri, set_experiment, start_run, log_artifact, log_metrics

try:
    import boto3
except ModuleNotFoundError:
    pass


@torch.no_grad()
def segmentation(img_array, input_image, label_arr, num_classes, gpkg_name, model, sample_size, num_bands, device):

    # switch to evaluate mode
    model.eval()
    transforms = tta.Compose([tta.HorizontalFlip(), ])

    WINDOW_SPLINE_2D = _window_2D(window_size=sample_size, power=2.0)
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
    # assert num_bands <= bands, f"Num of specified bands is not compatible with image shape {img_array.shape}"
    if num_bands < bands:
        warnings.warn(F"Num of specified bands {num_bands} is < image shape {img_array.shape}")
        img_array = img_array[:, :, :num_bands]
    elif num_bands > bands:
        warnings.warn(F" Num of specified bands {num_bands} is > image shape {img_array.shape} ")
        for i in range(num_bands - bands):
            o_band = img_array[:, :, 0:i]
            img_array = np.append(img_array, o_band, axis=2)
    padding = int(round(sample_size * (1 - 1.0 / 2.0)))
    step = int(sample_size / 2.0)
    img_array = pad(img_array, padding=padding, fill=np.nan)
    h_, w_, bands_ = img_array.shape
    mx = sample_size * xres
    my = sample_size * yres
    X_points = np.arange(0, w_ - sample_size + 1, step)
    Y_points = np.arange(0, h_ - sample_size + 1, step)
    pred_img = np.empty((h_, w_), dtype=np.uint8)
    sample = {'sat_img': None, 'map_img': None, 'metadata': None}

    for row in tqdm(Y_points, position=1, leave=False, desc='Inferring rows'):
        with tqdm(X_points, position=2, leave=False, desc='Inferring columns') as _tqdm:
            for col in _tqdm:
                sample['metadata'] = metadata
                totensor_transform = augmentation.compose_transforms(params, dataset="tst", type='totensor')
                sample['sat_img'] = img_array[row:row + sample_size, col:col + sample_size, :]
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
                    # reverse augmentation for outputs
                    deaugmented_output = transformer.deaugment_mask(augmented_output)
                    deaugmented_output = F.softmax(deaugmented_output, dim=1).squeeze(dim=0)
                    output_lst.append(deaugmented_output)
                outputs = torch.stack(output_lst)
                outputs = torch.mul(outputs, WINDOW_SPLINE_2D)
                outputs, _ = torch.max(outputs, dim=0)
                outputs = outputs.permute(1, 2, 0).argmax(dim=-1)
                outputs = outputs.reshape(sample_size, sample_size).cpu().numpy()
                pred_img[row:row + sample_size, col:col + sample_size] = outputs
    pred_img = pred_img[padding:-padding, padding:-padding]
    gdf = None
    if label_arr is not None:
        feature = defaultdict(list)
        cnt = 0
        for row in tqdm(range(0, h, sample_size), position=1, leave=False, desc='Inferring rows'):
            with tqdm(range(0, w, sample_size), position=2, leave=False, desc='Inferring columns') as _tqdm:
                for col in _tqdm:
                    label = label_arr[row:row + sample_size, col:col + sample_size]
                    pred = pred_img[row:row + sample_size, col:col + sample_size]
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
    # set_tracking_uri(get_key_def('mlflow_uri', params['global'], default="./mlruns"))
    # set_experiment('gdl-benchmarking/' + working_folder.name)
    # log_params(params['global'])
    # log_params(params['inference'])

    # CREATE LIST OF INPUT IMAGES FOR INFERENCE
    list_img = list_input_images(img_dir_or_csv, bucket_name, glob_patterns=["*.tif", "*.TIF"])

    if task == 'classification':
        classifier(params, list_img, model, device,
                   working_folder)  # FIXME: why don't we load from checkpoint in classification?

    elif task == 'segmentation':
        gdf_ = []
        gpkg_name_ = []

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
                assert local_img.is_file(), f"Could not locate raster file at {local_img}"
                with rasterio.open(local_img, 'r') as raster:
                    img_array, raster, _ = image_reader_as_array(input_image=raster, clip_gpkg=local_gpkg)
                    inf_meta = raster.meta
                    label = None
                    if local_gpkg:
                        assert local_gpkg.is_file(), f"Could not locate gkpg file at {local_gpkg}"
                        label = vector_to_raster(vector_file=local_gpkg,
                                                 input_image=raster,
                                                 out_shape=(inf_meta['height'], inf_meta['width']),
                                                 attribute_name=info['attribute_name'],
                                                 fill=0)  # background value in rasterized vector.

                    pred, gdf = segmentation(img_array, raster, label, num_classes_corrected,
                                             gpkg_name, model, chunk_size, num_bands, device)
                    if gdf is not None:
                        gdf_.append(gdf)
                        gpkg_name_.append(gpkg_name)
                    if local_gpkg:
                        with start_run(run_name=img_name, nested=True):
                            pixelMetrics= ComputePixelMetrics(label, pred, num_classes_corrected)
                            log_metrics(pixelMetrics.update(pixelMetrics.iou))
                            log_metrics(pixelMetrics.update(pixelMetrics.dice))
                    pred = pred[np.newaxis, :, :].astype(np.uint8)
                    inf_meta.update({"driver": "GTiff",
                                     "height": pred.shape[1],
                                     "width": pred.shape[2],
                                     "count": pred.shape[0],
                                     "dtype": 'uint8'})
                    with rasterio.open(inference_image, 'w+', **inf_meta) as dest:
                        dest.write(pred)
        if len(gdf_) >= 1:
            assert len(gdf_) == len(gpkg_name_), 'benchmarking unable to complete'
            all_gdf = pd.concat(gdf_)  # Concatenate all geo data frame into one geo data frame
            all_gdf.reset_index(drop=True, inplace=True)
            gdf_x = gpd.GeoDataFrame(all_gdf)
            gdf_x.to_file(working_folder.joinpath("benchmark.gpkg"), driver="GPKG", index=False)
        # log_artifact(working_folder)
    time_elapsed = time.time() - since
    print('Inference and Benchmarking completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    print('\n\nStart:\n\n')
    parser = argparse.ArgumentParser(usage="%(prog)s [-h] [-p YAML] [-i MODEL IMAGE] ",
                                     description='Inference and Benchmark on images using trained model')

    parser.add_argument('-p', '--param', metavar='yaml_file', nargs=1,
                        help='Path to parameters stored in yaml')
    parser.add_argument('-i', '--input', metavar='model_pth img_dir', nargs=2,
                        help='model_path and image_dir')
    args = parser.parse_args()

    if args.param:
        params = read_parameters(args.param[0])
    elif args.input:
        model = Path(args.input[0])
        image = args.input[1]

        checkpoint = load_checkpoint(model)
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

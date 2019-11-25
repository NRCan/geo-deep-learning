import torch
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

from models.model_choice import net
from utils.utils import load_from_checkpoint, get_device_ids, gpu_stats, get_key_def
from utils.readers import read_parameters, image_reader_as_array, read_csv
from utils.CreateDataset import MetaSegmentationDataset

try:
    import boto3
except ModuleNotFoundError:
    pass


def create_new_raster_from_base(input_raster, output_raster, write_array):
    """Function to use info from input raster to create new one.
    Args:
        input_raster: input raster path and name
        output_raster: raster name and path to be created with info from input
        write_array (optional): array to write into the new raster

    Return:
        none
    """

    with rasterio.open(input_raster, 'r') as src:
        with rasterio.open(output_raster, 'w',
                           driver=src.driver,
                           width=src.width,
                           height=src.height,
                           count=1,
                           crs=src.crs,
                           dtype=np.uint8,
                           transform=src.transform) as dst:
            dst.write(write_array[:, :], 1)


def sem_seg_inference(model, nd_array, overlay, chunk_size, num_classes, device, meta_map=None, metadata=None, output_path=Path(os.getcwd())):
    """Inference on images using semantic segmentation
    Args:
        model: model to use for inference
        nd_array: nd_array
        overlay: amount of overlay to apply
        num_classes: number of different classes that may be predicted by the model
        device: device used by pytorch (cpu ou cuda)

        returns a numpy array of the same size (h,w) as the input image, where each value is the predicted output.
    """

    # switch to evaluate mode
    model.eval()

    if len(nd_array.shape) == 3:
        h, w, nb = nd_array.shape
        # Pad with overlay on left and top and pad with chunk_size on right and bottom
        padded_array = np.pad(nd_array, ((overlay, chunk_size), (overlay, chunk_size), (0, 0)), mode='constant')
    elif len(nd_array.shape) == 2:
        h, w = nd_array.shape
        padded_array = np.expand_dims(np.pad(nd_array, ((overlay, chunk_size), (overlay, chunk_size)),
                                             mode='constant'), axis=0)
    else:
        h = 0
        w = 0
        padded_array = None

    h_padded, w_padded = padded_array.shape[:2]
    # Create an empty array of dimensions (c x h x w): num_classes x height of padded array x width of padded array
    output_probs = np.empty([num_classes, h_padded, w_padded], dtype=np.float32)
    # Create identical 0-filled array without channels dimension to receive counts for number of outputs generated in specific area.
    output_counts = np.zeros([output_probs.shape[1], output_probs.shape[2]], dtype=np.int32)

    if padded_array.any():
        with torch.no_grad():
            for row in tqdm(range(overlay, h + chunk_size, chunk_size - overlay), position=1, leave=False,
                      desc=f'Inferring rows with "{device}"'):
                row_start = row - overlay
                row_end = row_start + chunk_size
                with tqdm(range(overlay, w + chunk_size, chunk_size - overlay), position=2, leave=False, desc='Inferring columns') as _tqdm:
                    for col in _tqdm:
                        col_start = col - overlay
                        col_end = col_start + chunk_size

                        chunk_input = padded_array[row_start:row_end, col_start:col_end, :]
                        if meta_map:
                            chunk_input = MetaSegmentationDataset.append_meta_layers(chunk_input, meta_map, metadata)
                        inputs = torch.from_numpy(np.float32(np.transpose(chunk_input, (2, 0, 1))))

                        inputs.unsqueeze_(0)

                        inputs = inputs.to(device)
                        # forward
                        outputs = model(inputs)

                        # torchvision models give output it 'out' key. May cause problems in future versions of torchvision.
                        if isinstance(outputs, OrderedDict) and 'out' in outputs.keys():
                            outputs = outputs['out']

                        output_counts[row_start:row_end, col_start:col_end] += 1

                        # Add inference on sub-image to all completed inferences on previous sub-images.
                        # FIXME: This operation need to be optimized. Using a lot of RAM on large images.
                        output_probs[:, row_start:row_end, col_start:col_end] += np.squeeze(outputs.cpu().numpy(),
                                                                                            axis=0)

                        if debug and device.type == 'cuda':
                            res, mem = gpu_stats(device=device.index)
                            _tqdm.set_postfix(OrderedDict(gpu_perc=f'{res.gpu} %',
                                                          gpu_RAM=f'{mem.used / (1024 ** 2):.0f}/{mem.total / (1024 ** 2):.0f} MiB',
                                                          inp_size=inputs.cpu().numpy().shape,
                                                          out_size=outputs.cpu().numpy().shape,
                                                          overlay=overlay))
            if debug:
                output_counts_PIL = Image.fromarray(output_counts.astype(np.uint8), mode='L')
                output_counts_PIL.save(output_path.joinpath(f'output_counts.png'))

            # Divide array according to output counts. Manages overlap and returns a softmax array as if only one forward pass had been done.
            output_mask_softmax = np.divide(output_probs, np.maximum(output_counts, 1))
            # Give value of class to band with highest value in final inference
            output_mask = np.argmax(output_mask_softmax, axis = 0)

            # Resize the output array to the size of the input image and write it
            output_mask_cropped = output_mask[overlay:(h + overlay), overlay:(w + overlay)].astype(np.uint8)
            return output_mask_cropped
    else:
        raise IOError(f"Error classifying image : Image shape of {len(nd_array.shape)} is not recognized")


def classifier(params, img_list, model, device):
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
        img_name = os.path.basename(image['tif']) #TODO: pathlib
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
        print()

    csv_results = 'classification_results.csv'
    if bucket:
        np.savetxt(csv_results, classified_results, fmt='%s', delimiter=',')
        bucket.upload_file(csv_results, os.path.join(params['inference']['working_folder'], csv_results)) #TODO: pathlib
    else:
        np.savetxt(os.path.join(params['inference']['working_folder'], csv_results), classified_results, fmt='%s', #TODO: pathlib
                   delimiter=',')  # FIXME create directories if don't exist


def main(params):
    """
    Identify the class to which each image belongs.
    :param params: (dict) Parameters found in the yaml config file.

    """
    # SET BASIC VARIABLES AND PATHS
    since = time.time()
    chunk_size = get_key_def('chunk_size', params['inference'], None)
    overlap = get_key_def('overlap', params['inference'], None)
    nbr_pix_overlap = int(math.floor(overlap / 100 * chunk_size))
    num_bands = params['global']['number_of_bands']

    img_dir_or_csv = params['inference']['img_dir_or_csv_file']
    working_folder = Path(params['inference']['working_folder']) if params['inference']['working_folder'] \
        else Path(params['inference']['state_dict_path']).parent.joinpath(f'inf_{chunk_size}_overlap{overlap}_{num_bands}bands')
    Path.mkdir(working_folder, exist_ok=True)
    print(f'Inferences will be saved to: {working_folder}\n\n')

    bucket = None
    bucket_file_cache = []
    bucket_name = params['global']['bucket_name']

    # CONFIGURE MODEL
    model, state_dict_path, model_name = net(params, inference=True)

    num_devices = params['global']['num_gpus'] if params['global']['num_gpus'] else 0
    # list of GPU devices that are available and unused. If no GPUs, returns empty list
    lst_device_ids = get_device_ids(num_devices) if torch.cuda.is_available() else []
    device = torch.device(f'cuda:{lst_device_ids[0]}' if torch.cuda.is_available() and lst_device_ids else 'cpu')

    if lst_device_ids:
        print(f"Number of cuda devices requested: {num_devices}. Cuda devices available: {lst_device_ids}. Using {lst_device_ids[0]}\n\n")
    else:
        warnings.warn(f"No Cuda device available. This process will only run on CPU")

    try:
        model.to(device)
    except RuntimeError:
        print(f"Unable to use device. Trying device 0")
        device = torch.device(f'cuda:0' if torch.cuda.is_available() and lst_device_ids else 'cpu')
        model.to(device)

    if bucket_name:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        if img_dir_or_csv.endswith('.csv'):
            bucket.download_file(img_dir_or_csv, 'img_csv_file.csv')
            list_img = read_csv('img_csv_file.csv', inference=True)
        else:
            raise NotImplementedError(
                'Specify a csv file containing images for inference. Directory input not implemented yet')
    else:
        if img_dir_or_csv.endswith('.csv'):
            list_img = read_csv(img_dir_or_csv, inference=True)
        else:
            img_dir = Path(img_dir_or_csv)
            assert img_dir.is_dir(), f'Could not find directory "{img_dir_or_csv}"'
            list_img_paths = sorted(img_dir.glob('*.tif'))
            list_img = []
            for img_path in list_img_paths:
                img = {}
                img['tif'] = img_path
                list_img.append(img)
            assert len(list_img) >= 0, f'No .tif files found in {img_dir_or_csv}'

    if params['global']['task'] == 'classification':
        classifier(params, list_img, model, device)

    elif params['global']['task'] == 'segmentation':
        if bucket:
            bucket.download_file(state_dict_path, "saved_model.pth.tar")
            model, _ = load_from_checkpoint("saved_model.pth.tar", model)
        else:
            model, _ = load_from_checkpoint(state_dict_path, model)

        num_classes = params['global']['num_classes']
        if num_classes == 1:
            # assume background is implicitly needed (makes no sense to predict with one class otherwise)
            # this will trigger some warnings elsewhere, but should succeed nonetheless
            num_classes = 2
        with tqdm(list_img, desc='image list', position=0) as _tqdm:
            for img in _tqdm:
                img_name = Path(img['tif']).name
                if bucket:
                    local_img = f"Images/{img_name}"
                    bucket.download_file(img['tif'], local_img)
                    inference_image = f"Classified_Images/{img_name.split('.')[0]}_inference.tif"
                    if img['meta']:
                        if img['meta'] not in bucket_file_cache:
                            bucket_file_cache.append(img['meta'])
                            bucket.download_file(img['meta'], img['meta'].split('/')[-1])
                        img['meta'] = img['meta'].split('/')[-1]
                else:
                    local_img = Path(img['tif'])
                    inference_image = working_folder.joinpath(f"{img_name.split('.')[0]}_inference.tif")

                assert local_img.is_file(), f"Could not open raster file at {local_img}"
                with rasterio.open(local_img, 'r') as raster:

                    np_input_image = image_reader_as_array(input_image=raster,
                                                           scale=get_key_def('scale_data', params['global'], None),
                                                           aux_vector_file=get_key_def('aux_vector_file',
                                                                                       params['global'], None),
                                                           aux_vector_attrib=get_key_def('aux_vector_attrib',
                                                                                         params['global'], None),
                                                           aux_vector_ids=get_key_def('aux_vector_ids',
                                                                                      params['global'], None),
                                                           aux_vector_dist_maps=get_key_def('aux_vector_dist_maps',
                                                                                            params['global'], True),
                                                           aux_vector_scale=get_key_def('aux_vector_scale',
                                                                                        params['global'], None))

                meta_map, metadata = get_key_def("meta_map", params["global"], {}), None
                if meta_map:
                    assert img['meta'] is not None and isinstance(img['meta'], str) and os.path.isfile(img['meta']), \
                        "global configuration requested metadata mapping onto loaded samples, but raster did not have available metadata"
                    metadata = read_parameters(img['meta'])

                if debug:
                    _tqdm.set_postfix(OrderedDict(img_name=img_name,
                                                  img=np_input_image.shape,
                                                  img_min_val=np.min(np_input_image),
                                                  img_max_val=np.max(np_input_image)))

                input_band_count = np_input_image.shape[2] + MetaSegmentationDataset.get_meta_layer_count(meta_map)
                assert input_band_count == params['global']['number_of_bands'], \
                    f"The number of bands in the input image ({input_band_count}) and the parameter" \
                    f"'number_of_bands' in the yaml file ({params['global']['number_of_bands']}) should be identical"

                # START INFERENCES ON SUB-IMAGES
                sem_seg_results = sem_seg_inference(model, np_input_image, nbr_pix_overlap, chunk_size, num_classes,
                                                    device, meta_map, metadata, output_path=working_folder)

                if debug:
                    _tqdm.set_postfix(
                        OrderedDict(result_min_val=np.min(sem_seg_results), result_max_val=np.max(sem_seg_results)))

                if debug and len(np.unique(sem_seg_results)) == 1:
                    print(
                        f'Something is wrong. Inference contains only one value. Make sure data scale is coherent with training domain values.')

                # CREATE GEOTIF FROM METADATA OF ORIGINAL IMAGE
                tqdm.write(f'Saving inference...')
                create_new_raster_from_base(local_img, inference_image, sem_seg_results)
                tqdm.write(f"\n\nSemantic segmentation of image {img_name} completed\n\n")
                if bucket:
                    bucket.upload_file(inference_image, os.path.join(params['inference']['working_folder'],
                                                                     f"{img_name.split('.')[0]}_inference.tif"))
    else:
        raise ValueError(
            f"The task should be either classification or segmentation. The provided value is {params['global']['task']}")

    time_elapsed = time.time() - since
    print('Inference completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    print('\n\nStart:\n\n')
    parser = argparse.ArgumentParser(description='Inference on images using trained model')
    parser.add_argument('param_file', metavar='file',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    params = read_parameters(args.param_file)

    debug = True if params['global']['debug_mode'] else False

    main(params)

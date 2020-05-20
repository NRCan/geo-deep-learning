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

from models.model_choice import net
from utils.utils import load_from_checkpoint, get_device_ids, gpu_stats, get_key_def
from utils.readers import read_parameters, image_reader_as_array, read_csv
from utils.CreateDataset import MetaSegmentationDataset
from utils.visualization import vis, vis_from_batch

try:
    import boto3
except ModuleNotFoundError:
    pass


def sem_seg_inference(model, nd_array, overlay, chunk_size, num_classes, device, meta_map=None, metadata=None, output_path=Path(os.getcwd()), index=0, debug=False):
    """Inference on images using semantic segmentation
    Args:
        model: model to use for inference
        nd_array: nd_array
        overlay: amount of overlay to apply
        num_classes: number of different classes that may be predicted by the model
        device: device used by pytorch (cpu ou cuda)
        meta_map:
        metadata:
        output_path: path to save debug files
        index: (int) index of array from list of images on which inference is performed

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

                        inputs.unsqueeze_(0) #Add dummy batch dimension

                        inputs = inputs.to(device)
                        # forward
                        outputs = model(inputs)

                        # torchvision models give output in 'out' key. May cause problems in future versions of torchvision.
                        if isinstance(outputs, OrderedDict) and 'out' in outputs.keys():
                            outputs = outputs['out']

                        if debug:
                            if index == 0:
                                tqdm.write(f'(debug mode) Visualizing inferred tiles...')
                            vis_from_batch(params, inputs, outputs, batch_index=0, vis_path=output_path,
                                        dataset=f'{row_start}_{col_start}_inf', ep_num=index, debug=True)

                        outputs = F.softmax(outputs, dim=1)

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
                tqdm.write(f'Dividing array according to output counts...\n')

            # Divide array according to output counts. Manages overlap and returns a softmax array as if only one forward pass had been done.
            output_mask_raw = np.divide(output_probs, np.maximum(output_counts, 1))  # , 1 is added to overwrite 0 values.

            # Resize the output array to the size of the input image and write it
            output_mask_raw_cropped = np.moveaxis(output_mask_raw, 0, -1)
            output_mask_raw_cropped = output_mask_raw_cropped[overlay:(h + overlay), overlay:(w + overlay), :]

            return output_mask_raw_cropped
    else:
        raise IOError(f"Error classifying image : Image shape of {len(nd_array.shape)} is not recognized")


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
        bucket.upload_file(csv_results, os.path.join(working_folder, csv_results)) #TODO: pathlib
    else:
        np.savetxt(os.path.join(working_folder, csv_results), classified_results, fmt='%s', #TODO: pathlib
                   delimiter=',')


def main(params):
    """
    Identify the class to which each image belongs.
    :param params: (dict) Parameters found in the yaml config file.

    """
    # SET BASIC VARIABLES AND PATHS
    since = time.time()

    debug = get_key_def('debug_mode', params['global'], False)
    if debug:
        warnings.warn(f'Debug mode activated. Some debug features may mobilize extra disk space and cause delays in execution.')

    num_classes = params['global']['num_classes']
    if params['global']['task'] == 'segmentation':
        # assume background is implicitly needed (makes no sense to predict with one class, for example.)
        # this will trigger some warnings elsewhere, but should succeed nonetheless
        num_classes_corrected = num_classes + 1 # + 1 for background # FIXME temporary patch for num_classes problem.
    elif params['global']['task'] == 'classification':
        num_classes_corrected = num_classes

    chunk_size = get_key_def('chunk_size', params['inference'], 512)
    overlap = get_key_def('overlap', params['inference'], 10)
    nbr_pix_overlap = int(math.floor(overlap / 100 * chunk_size))
    num_bands = params['global']['number_of_bands']

    img_dir_or_csv = params['inference']['img_dir_or_csv_file']

    default_working_folder = Path(params['inference']['state_dict_path']).parent.joinpath(f'inference_{num_bands}bands')
    working_folder = Path(get_key_def('working_folder', params['inference'], default_working_folder)) # TODO: remove working_folder parameter in all templates
    working_folder.mkdir(parents=True, exist_ok=True)
    print(f'Inferences will be saved to: {working_folder}\n\n')

    bucket = None
    bucket_file_cache = []
    bucket_name = params['global']['bucket_name']

    # CONFIGURE MODEL
    model, state_dict_path, model_name = net(params, num_channels=num_classes_corrected, inference=True)

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
            list_img_paths = sorted(img_dir.glob('*.tif'))  # FIXME: what if .tif is in caps (.TIF) ?
            list_img = []
            for img_path in list_img_paths:
                img = {}
                img['tif'] = img_path
                list_img.append(img)
            assert len(list_img) >= 0, f'No .tif files found in {img_dir_or_csv}'

    if params['global']['task'] == 'classification':
        classifier(params, list_img, model, device, working_folder)  # FIXME: why don't we load from checkpoint in classification?

    elif params['global']['task'] == 'segmentation':
        if bucket:
            bucket.download_file(state_dict_path, "saved_model.pth.tar")
            model, _ = load_from_checkpoint("saved_model.pth.tar", model, inference=True)
        else:
            model, _ = load_from_checkpoint(state_dict_path, model, inference=True)

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

                scale = get_key_def('scale_data', params['global'], None)
                with rasterio.open(local_img, 'r') as raster:

                    np_input_image = image_reader_as_array(input_image=raster,
                                                           scale=scale,
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
                if input_band_count > params['global']['number_of_bands']:
                    # FIXME: Following statements should be reconsidered to better manage inconsistencies between
                    #  provided number of band and image number of band.
                    warnings.warn(f"Input image has more band than the number provided in the yaml file ({params['global']['number_of_bands']}). "
                                  f"Will use the first {params['global']['number_of_bands']} bands of the input image.")
                    np_input_image = np_input_image[:, :, 0:params['global']['number_of_bands']]
                    print(f"Input image's new shape: {np_input_image.shape}")

                elif input_band_count < params['global']['number_of_bands']:
                    warnings.warn(f"Skipping image: The number of bands requested in the yaml file ({params['global']['number_of_bands']})"
                                  f"can not be larger than the number of band in the input image ({input_band_count}).")
                    continue

                # START INFERENCES ON SUB-IMAGES
                sem_seg_results_per_class = sem_seg_inference(model, np_input_image, nbr_pix_overlap, chunk_size, num_classes_corrected,
                                                    device, meta_map, metadata, output_path=working_folder, index=_tqdm.n, debug=debug)

                # CREATE GEOTIF FROM METADATA OF ORIGINAL IMAGE
                tqdm.write(f'Saving inference...\n')
                if get_key_def('heatmaps', params['inference'], False):
                    tqdm.write(f'Heatmaps will be saved.\n')
                vis(params, np_input_image, sem_seg_results_per_class, working_folder, inference_input_path=local_img, debug=debug)

                tqdm.write(f"\n\nSemantic segmentation of image {img_name} completed\n\n")
                if bucket:
                    bucket.upload_file(inference_image, os.path.join(working_folder, f"{img_name.split('.')[0]}_inference.tif"))
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

    main(params)

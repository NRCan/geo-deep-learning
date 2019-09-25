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
from torch import nn
from tqdm import tqdm

from models.model_choice import net
from utils.utils import read_parameters, assert_band_number, load_from_checkpoint, \
    image_reader_as_array, read_csv, get_device_ids, minmax_scale

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


def sem_seg_inference(model, nd_array, overlay, chunk_size, num_classes, device):
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
        padded_array = np.pad(nd_array, ((overlay, chunk_size), (overlay, chunk_size), (0, 0)), mode='constant')
    elif len(nd_array.shape) == 2:
        h, w = nd_array.shape
        padded_array = np.expand_dims(np.pad(nd_array, ((overlay, chunk_size), (overlay, chunk_size)),
                                             mode='constant'), axis=0)
    else:
        h = 0
        w = 0
        padded_array = None

    output_probs = np.empty([num_classes, h + overlay + chunk_size, w + overlay + chunk_size], dtype=np.float32)
    output_counts = np.zeros([output_probs.shape[1], output_probs.shape[2]], dtype=np.int32)

    if padded_array.any():
        with torch.no_grad():
            # TODO: BUG. tqdm's second loop printing on multiple lines.
            for row in tqdm(range(overlay, h, chunk_size - overlay), position=1, leave=False):
                row_start = row - overlay
                row_end = row_start + chunk_size
                for col in range(overlay, w, chunk_size - overlay):
                    col_start = col - overlay
                    col_end = col_start + chunk_size

                    chunk_input = padded_array[row_start:row_end, col_start:col_end, :]
                    inputs = torch.from_numpy(np.float32(np.transpose(chunk_input, (2, 0, 1))))

                    inputs.unsqueeze_(0)

                    inputs = inputs.to(device)
                    # forward
                    outputs = model(inputs)

                    # torchvision models give output it 'out' key. May cause problems in future versions of torchvision.
                    if isinstance(outputs, OrderedDict) and 'out' in outputs.keys():
                        outputs = outputs['out']

                    output_counts[row_start:row_end, col_start:col_end] += 1
                    output_probs[:, row_start:row_end, col_start:col_end] += np.squeeze(outputs.cpu().numpy(), axis=0)

            output_mask = np.argmax(np.divide(output_probs, np.maximum(output_counts, 1)), axis=0)
            # Resize the output array to the size of the input image and write it
            return output_mask[overlay:(h + overlay), overlay:(w + overlay)].astype(np.uint8)
    else:
        print("Error classifying image : Image shape of {:1} is not recognized".format(len(nd_array.shape)))


def classifier(params, img_list, model):
    """
    Classify images by class
    :param params:
    :param img_list:
    :param model:
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
        img_name = os.path.basename(image['tif'])
        model.eval()
        if bucket:
            img = Image.open(f"Images/{img_name}").resize((299, 299), resample=Image.BILINEAR)
        else:
            img = Image.open(image['tif']).resize((299, 299), resample=Image.BILINEAR)
        to_tensor = torchvision.transforms.ToTensor()

        img = to_tensor(img)
        img = img.unsqueeze(0)
        with torch.no_grad():
            if torch.cuda.is_available():
                img = img.cuda()
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
        bucket.upload_file(csv_results, os.path.join(params['inference']['working_folder'], csv_results))
    else:
        np.savetxt(os.path.join(params['inference']['working_folder'], csv_results), classified_results, fmt='%s', delimiter=',')


def calc_overlap(params):
    """
    Function to calculate the number of pixels requires in overlap, based on chunk_size and overlap percentage,
    if provided in the config file
    :param params: (dict) Parameters found in the yaml config file.
    :return: (int) number of pixel required for overlap
    """
    chunk_size = 512
    overlap = 10

    if params['inference']['chunk_size']:
        chunk_size = int(params['inference']['chunk_size'])
    if params['inference']['overlap']:
        overlap = int(params['inference']['overlap'])
    nbr_pix_overlap = int(math.floor(overlap / 100 * chunk_size))
    return chunk_size, nbr_pix_overlap


def main(params):
    """
    Identify the class to which each image belongs.
    :param params: (dict) Parameters found in the yaml config file.

    """
    since = time.time()
    csv_file = params['inference']['img_csv_file']

    bucket = None
    bucket_name = params['global']['bucket_name']

    model, state_dict_path, model_name = net(params, inference=True)

    num_devices = params['global']['num_gpus'] if params['global']['num_gpus'] else 0
    # list of GPU devices that are available and unused. If no GPUs, returns empty list
    lst_device_ids = get_device_ids(num_devices) if torch.cuda.is_available() else []
    device = torch.device(f'cuda:{lst_device_ids[0]}' if torch.cuda.is_available() and lst_device_ids else 'cpu')

    if lst_device_ids:
        print(f"Using Cuda device {lst_device_ids[0]}")
    else:
        warnings.warn(f"No Cuda device available. This process will only run on CPU")

    model.to(device)

    if bucket_name:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        bucket.download_file(csv_file, 'img_csv_file.csv')
        list_img = read_csv('img_csv_file.csv', inference=True)
    else:
        list_img = read_csv(csv_file, inference=True)

    if params['global']['task'] == 'classification':
        classifier(params, list_img, model)

    elif params['global']['task'] == 'segmentation':
        if bucket:
            bucket.download_file(state_dict_path, "saved_model.pth.tar")
            model = load_from_checkpoint("saved_model.pth.tar", model)
        else:
            model = load_from_checkpoint(state_dict_path, model)

        chunk_size, nbr_pix_overlap = calc_overlap(params)
        num_classes = params['global']['num_classes']
        for img in tqdm(list_img, desc='image list', position=0):
            img_name = os.path.basename(img['tif'])
            if bucket:
                local_img = f"Images/{img_name}"
                bucket.download_file(img['tif'], local_img)
                inference_image = f"Classified_Images/{img_name.split('.')[0]}_inference.tif"
            else:
                local_img = img['tif']
                inference_image = os.path.join(params['inference']['working_folder'],
                                               f"{img_name.split('.')[0]}_inference.tif")

            assert_band_number(local_img, params['global']['number_of_bands'])

            nd_array_tif = image_reader_as_array(local_img)
            # Scale arrays to values [0,1]
            if params['sample']['scale_data']:
                min, max = params['sample']['scale_data']
                nd_array_tif = minmax_scale(nd_array_tif,
                                              orig_range=(np.min(nd_array_tif), np.max(nd_array_tif)),
                                              scale_range=(min,max))
            sem_seg_results = sem_seg_inference(model, nd_array_tif, nbr_pix_overlap, chunk_size, num_classes, device)
            create_new_raster_from_base(local_img, inference_image, sem_seg_results)
            tqdm.write(f"Semantic segmentation of image {img_name} completed")
            #print(f"Semantic segmentation of image {img_name} completed")
            if bucket:
                bucket.upload_file(inference_image, os.path.join(params['inference']['working_folder'],
                                                                 f"{img_name.split('.')[0]}_inference.tif"))
    else:
        raise ValueError(f"The task should be either classification or segmentation. The provided value is {params['global']['task']}")

    time_elapsed = time.time() - since
    print('Inference completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    print('Start: ')
    parser = argparse.ArgumentParser(description='Inference on images using trained model')
    parser.add_argument('param_file', metavar='file',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    params = read_parameters(args.param_file)

    main(params)

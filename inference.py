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
from models.model_choice import net
from utils import read_parameters, assert_band_number, load_from_checkpoint, \
    image_reader_as_array, read_csv
from wcs import create_bbox_from_pol, cut_bbox_from_maxsize, wcs_request, merge_wcs_tiles

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
        with rasterio.open( output_raster, 'w',
                            driver=src.driver,
                            width=src.width,
                            height=src.height,
                            count=1,
                            crs=src.crs,
                            dtype=np.uint8,
                            transform=src.transform) as dst:
            dst.write(write_array[:,:,0], 1)


def sem_seg_inference(model, nd_array, overlay):
    """Inference on images using semantic segmentation
    Args:
        model: model to use for inference
        nd_array: nd_array
        overlay: amount of overlay to apply

        returns a numpy array of the same size (h,w) as the input image, where each value is the predicted output.
    """
    # Chunk size. Should not be modified often. We want the biggest chunk to be process at a time but,
    # a too large image chunk will bust the GPU memory when processing.
    chunk_size = 512  # TODO parametrize this.

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

    output_np = np.empty([h + overlay + chunk_size, w + overlay + chunk_size, 1], dtype=np.uint8)

    if padded_array.any():
        with torch.no_grad():
            for row in range(0, h, chunk_size - (2 * overlay)):
                for col in range(0, w, chunk_size - (2 * overlay)):

                    chunk_input = padded_array[row:row + chunk_size, col:col + chunk_size, :]
                    inputs = torch.from_numpy(np.float32(np.transpose(chunk_input, (2, 0, 1))))

                    inputs.unsqueeze_(0)

                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                    # forward
                    outputs = model(inputs)

                    a, pred = torch.max(outputs, dim=1)
                    segmentation = torch.squeeze(pred)

                    row_from = row + overlay
                    row_to = row + chunk_size - overlay
                    col_from = col + overlay
                    col_to = col + chunk_size - overlay

                    useful_sem_seg = segmentation[overlay:chunk_size - overlay, overlay:chunk_size - overlay]
                    output_np[row_from:row_to, col_from:col_to, 0] = useful_sem_seg.cpu()

            # Resize the output array to the size of the input image and write it
            output_np = output_np[overlay:h + overlay, overlay:w + overlay]
            return output_np
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


def main(params):
    """
    Identify the class to which each image belongs.
    :param params: (dict) Parameters found in the yaml config file.

    """
    since = time.time()
    csv_file = params['inference']['img_csv_file']

    model_depth = 5  # TODO: change fixed value for parameter in config file.
    nbr_pix_overlay = 2 ** (model_depth + 1)

    bucket = None
    bucket_name = params['global']['bucket_name']

    if bucket_name:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        bucket.download_file(csv_file, 'img_csv_file.csv')
        list_img = read_csv('img_csv_file.csv', inference=True)
    else:
        list_img = read_csv(csv_file, inference=True)

    model, state_dict_path, model_name = net(params, inference=True)
    if torch.cuda.is_available():
        model = model.cuda()

    if params['global']['task'] == 'classification':
        classifier(params, list_img, model)

    elif params['global']['task'] == 'segmentation':
        if bucket:
            bucket.download_file(state_dict_path, "saved_model.pth.tar")
            model = load_from_checkpoint("saved_model.pth.tar", model)
        else:
            model = load_from_checkpoint(state_dict_path, model)

        if params['global']['source'] == 'wcs':
            if bucket:
                bucket.download_file(params['wcs_parameters']['aoi_file'], "aoi_file.gpkg")
                lst_bbox = create_bbox_from_pol("aoi_file.gpkg")
            else:
                lst_bbox = create_bbox_from_pol(params['wcs_parameters']['aoi_file'])
            for elem in lst_bbox:
                lst_sub_bboxes = cut_bbox_from_maxsize(elem['bbox'],
                                                       params['wcs_parameters']['maxsize'],
                                                       params['wcs_parameters']['resx'],
                                                       params['wcs_parameters']['resy'])
                lst_tmpfiles = []
                for bbox in lst_sub_bboxes:
                    lst_tmpfiles.append(wcs_request(sub_bbox=bbox,
                                                    service_url=params['wcs_parameters']['service_url'],
                                                    version=params['wcs_parameters']['version'],
                                                    coverage=params['wcs_parameters']['coverage'],
                                                    epsg=params['wcs_parameters']['epsg'],
                                                    resx=params['wcs_parameters']['resx'],
                                                    resy=params['wcs_parameters']['resy'],
                                                    output_format=params['wcs_parameters']['format']))
                ndarray_wcs, out_trans = merge_wcs_tiles(lst_tmpfiles, params['wcs_parameters']['epsg'])

                sem_seg_results = sem_seg_inference(model, ndarray_wcs, nbr_pix_overlay)

                output_raster = os.path.join(params['inference']['working_folder'], f'{lst_bbox.index(elem)}.tif')
                with rasterio.open(output_raster, 'w',
                                   driver=params['wcs_parameters']['format'],
                                   width=ndarray_wcs.shape[1],
                                   height=ndarray_wcs.shape[2],
                                   count=1,
                                   crs=params['wcs_parameters']['epsg'],
                                   dtype=np.uint8,
                                   transform=out_trans) as dst:
                    dst.write(sem_seg_results[:, :, 0], 1)

        elif params['global']['source'] == 'local':
            for img in list_img:
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
                sem_seg_results = sem_seg_inference(model, nd_array_tif, nbr_pix_overlay)
                create_new_raster_from_base(local_img, inference_image, sem_seg_results)
                print(f"Semantic segmentation of image {img_name} completed")
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



import argparse
import os
import numpy as np
import h5py
import warnings
import fiona
import rasterio
from rasterio import features

from utils import read_parameters, assert_band_number, image_reader_as_array, \
    create_or_empty_folder, validate_num_classes, read_csv

try:
    import boto3
except ModuleNotFoundError:
    warnings.warn('The boto3 library counldn\'t be imported. Ignore if not using AWS s3 buckets', ImportWarning)
    pass


def mask_image(arrayA, arrayB):
    """Function to mask values of arrayB, based on 0 values from arrayA.

    >>> x1 = np.array([0, 2, 4, 6, 0, 3, 9, 8], dtype=np.uint8).reshape(2,2,2)
    >>> x2 = np.array([1.5, 1.2, 1.6, 1.2, 11., 1.1, 25.9, 0.1], dtype=np.float32).reshape(2,2,2)
    >>> mask_image(x1, x2)
    array([[[ 0. ,  0. ],
        [ 1.6,  1.2]],
        [[11. ,  1.1],
        [25.9,  0.1]]], dtype=float32)
    """

    # Handle arrayA of shapes (h,w,c) and (h,w)
    if len(arrayA.shape) == 3:
        mask = arrayA[:, :, 0] != 0
    else:
        mask = arrayA != 0

    ma_array = np.zeros(arrayB.shape, dtype=arrayB.dtype)
    # Handle arrayB of shapes (h,w,c) and (h,w)
    if len(arrayB.shape) == 3:
        for i in range(0, arrayB.shape[2]):
            ma_array[:, :, i] = mask * arrayB[:, :, i]
    else:
        ma_array = arrayB * mask
    return ma_array


def resize_datasets(hdf5_file):
    """Function to add one entry to both the datasets"""

    n = hdf5_file['sat_img'].shape[0]

    new_size = n + 1
    hdf5_file['sat_img'].resize(new_size, axis=0)
    hdf5_file['map_img'].resize(new_size, axis=0)


def samples_preparation(in_img_array, label_array, sample_size, dist_samples, samples_count, num_classes, samples_file,
                        dataset, background_switch):
    """Extract and write samples from input image and reference image
    Args:
        sat_img: num py array of to the input image
        ref_img: num py array the reference image
        sample_size: Size (in pixel) of the samples to create
        dist_samples: Distance (in pixel) between samples in both images
        samples_count: Current number of samples created (will be appended and return)
        num_classes: Number of classes in reference data (will be appended and return)
        samples_file: hdfs file where samples will be written
        dataset: Type of dataset where the samples will be written. Can be either of 'trn' or 'val'
        background_switch: Indicate if samples containing only background pixels will be written or discarded
    """

    # read input and reference images as array

    h, w, num_bands = in_img_array.shape

    if dataset == 'trn':
        idx_samples = samples_count['trn']
    elif dataset == 'val':
        idx_samples = samples_count['val']

    # half tile padding
    half_tile = int(sample_size / 2)
    pad_in_img_array = np.pad(in_img_array, ((half_tile, half_tile), (half_tile, half_tile), (0, 0)),
                              mode='constant')
    pad_label_array = np.pad(label_array, ((half_tile, half_tile), (half_tile, half_tile), (0, 0)), mode='constant')

    for row in range(0, h, dist_samples):
        for column in range(0, w, dist_samples):
            data = (pad_in_img_array[row:row + sample_size, column:column + sample_size, :])
            target = np.squeeze(pad_label_array[row:row + sample_size, column:column + sample_size, :], axis=2)

            target_class_num = max(target.ravel())

            if (background_switch and target_class_num != 0) or (not background_switch):
                resize_datasets(samples_file)
                samples_file["sat_img"][idx_samples, ...] = data
                samples_file["map_img"][idx_samples, ...] = target
                idx_samples += 1

            if num_classes < target_class_num:
                num_classes = target_class_num

    if dataset == 'trn':
        samples_count['trn'] = idx_samples
    elif dataset == 'val':
        samples_count['val'] = idx_samples

    # return the appended samples count and number of classes.
    return samples_count, num_classes


def vector_to_raster(vector_file, input_image, attribute_name):
    """Function to rasterize vector data.
    Args:
        vector_file: Path and name of reference GeoPackage
        input_image: Path and name of the input raster image
        attribute_name: Attribute containing the pixel value to write

    Return
        num py array of the burned image
    """

    # Extract vector features to burn in the raster image
    with fiona.open(vector_file, 'r') as src:
        lst_vector = [vector for vector in src]

    # Sort feature in order to priorize the burning in the raster image (ex: vegetation before roads...)
    lst_vector.sort(key=lambda vector : vector['properties'][attribute_name])
    lst_vector_tuple = [(vector['geometry'], int(vector['properties'][attribute_name])) for vector in lst_vector]

    # Open input raster image to have access to number of rows, column, crs...
    with rasterio.open(input_image, 'r') as src:
        burned_raster = rasterio.features.rasterize( (vector_tuple for vector_tuple in lst_vector_tuple),
                                    fill = 0,
                                    out_shape=src.shape,
                                    transform=src.transform,
                                    dtype=np.uint8)

    return burned_raster


def main( bucket_name, data_path, samples_size, num_classes, number_of_bands, csv_file, samples_dist,
          remove_background, mask_input_image, mask_reference):
    gpkg_file = []
    if bucket_name:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        bucket.download_file(csv_file, 'samples_prep.csv')
        list_data_prep = read_csv('samples_prep.csv')
        if data_path:
            final_samples_folder = os.path.join(data_path, "samples")
        else:
            final_samples_folder = "samples"
        samples_folder = "samples"
        out_label_folder = "label"

    else:
        list_data_prep = read_csv(csv_file)
        samples_folder = os.path.join(data_path, "samples")
        out_label_folder = os.path.join(data_path, "label")

    create_or_empty_folder(samples_folder)
    create_or_empty_folder(out_label_folder)

    number_samples = {'trn': 0, 'val': 0}
    number_classes = 0

    trn_hdf5 = h5py.File(os.path.join(samples_folder, "trn_samples.hdf5"), "w")
    val_hdf5 = h5py.File(os.path.join(samples_folder, "val_samples.hdf5"), "w")

    trn_hdf5.create_dataset("sat_img", (0, samples_size, samples_size, number_of_bands), np.float32,
                            maxshape=(None, samples_size, samples_size, number_of_bands))
    trn_hdf5.create_dataset("map_img", (0, samples_size, samples_size), np.uint8,
                            maxshape=(None, samples_size, samples_size))
    val_hdf5.create_dataset("sat_img", (0, samples_size, samples_size, number_of_bands), np.float32,
                            maxshape=(None, samples_size, samples_size, number_of_bands))
    val_hdf5.create_dataset("map_img", (0, samples_size, samples_size), np.uint8,
                            maxshape=(None, samples_size, samples_size))
    for info in list_data_prep:

        if bucket_name:
            bucket.download_file(info['tif'], "Images/" + info['tif'].split('/')[-1])
            info['tif'] = "Images/" + info['tif'].split('/')[-1]
            if info['gpkg'] not in gpkg_file:
                gpkg_file.append(info['gpkg'])
                bucket.download_file(info['gpkg'], info['gpkg'].split('/')[-1])
            info['gpkg'] = info['gpkg'].split('/')[-1]
        assert_band_number(info['tif'], number_of_bands)

        # Read the input raster image
        np_input_image = image_reader_as_array(info['tif'])

        # Validate the number of class in the vector file
        validate_num_classes(info['gpkg'], num_classes, info['attribute_name'])

        # Burn vector file in a raster file
        np_label_raster = vector_to_raster(info['gpkg'], info['tif'], info['attribute_name'])

        # Mask the zeros from input image into label raster.
        if mask_reference:
            np_label_raster = mask_image(np_input_image, np_label_raster)

        # Mask zeros from label raster into input image otherwise use original image
        if mask_input_image:
            np_input_image = mask_image(np_label_raster, np_input_image)

        if info['dataset'] == 'trn':
            out_file = trn_hdf5
        elif info['dataset'] == 'val':
            out_file = val_hdf5

        np_label_raster = np.reshape(np_label_raster, (np_label_raster.shape[0], np_label_raster.shape[1], 1))
        number_samples, number_classes = samples_preparation(np_input_image, np_label_raster, samples_size, samples_dist,
                                                             number_samples, number_classes, out_file, info['dataset'],
                                                             remove_background)

        print(info['tif'])
        print(number_samples)
        out_file.flush()

    trn_hdf5.close()
    val_hdf5.close()

    print("Number of samples created: ", number_samples)

    if bucket_name:
        print('Transfering Samples to the bucket')
        bucket.upload_file(samples_folder + "/trn_samples.hdf5", final_samples_folder + '/trn_samples.hdf5')
        bucket.upload_file(samples_folder + "/val_samples.hdf5", final_samples_folder + '/val_samples.hdf5')

    print("End of process")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample preparation')
    parser.add_argument('ParamFile', metavar='DIR',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    params = read_parameters(args.ParamFile)

    import time
    start_time = time.time()


    main(params['global']['bucket_name'],
         params['global']['data_path'],
         params['global']['samples_size'],
         params['global']['num_classes'],
         params['global']['number_of_bands'],
         params['sample']['prep_csv_file'],
         params['sample']['samples_dist'],
         params['sample']['remove_background'],
         params['sample']['mask_input_image'],
         params['sample']['mask_reference'])

    print ("Elapsed time:{}".format(time.time() - start_time))


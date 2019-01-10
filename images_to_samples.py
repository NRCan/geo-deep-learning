import argparse
import csv
import os
import numpy as np
import h5py
import warnings
from osgeo import gdal, osr, ogr
from utils import read_parameters, create_new_raster_from_base, assert_band_number, image_reader_as_array, \
    create_or_empty_folder, validate_num_classes

try:
    import boto3
except ModuleNotFoundError:
    warnings.warn('The boto3 library counldn\'t be imported. Ignore if not using AWS s3 buckets', ImportWarning)
    pass


def mask_image(arrayA, arrayB):
    """Function to mask values of arrayB, based on 0 values from arrayA."""

    # Handle arrayA of shapes (h,w,c) and (h,w)
    if len(arrayA.shape) == 3:
        mask = arrayA[:, :, 0] != 0
    else:
        mask = arrayA != 0

    ma_array = np.zeros(arrayB.shape, dtype=np.uint8)
    # Handle arrayB of shapes (h,w,c) and (h,w)
    if len(arrayB.shape) == 3:
        for i in range(0, arrayB.shape[2]):
            ma_array[:, :, i] = mask * arrayB[:, :, i]
    else:
        ma_array = arrayB * mask
    return ma_array


def read_csv(csv_file_name):
    """Open csv file and parse it."""

    list_values = []
    with open(csv_file_name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            list_values.append({'tif': row[0], 'gpkg': row[1], 'attribute_name': row[2], 'dataset': row[3]})

    return sorted(list_values, key=lambda k: k['dataset'])


def resize_datasets(hdf5_file):
    """Function to add one entry to both the datasets"""

    n = hdf5_file['sat_img'].shape[0]

    new_size = n + 1
    hdf5_file['sat_img'].resize(new_size, axis=0)
    hdf5_file['map_img'].resize(new_size, axis=0)


def samples_preparation(sat_img, ref_img, sample_size, dist_samples, samples_count, num_classes, samples_file, dataset,
                        background_switch):
    """Extract and write samples from input image and reference image
    Args:
        sat_img: Path and name to the input image
        ref_img: path and name to the reference image
        sample_size: Size (in pixel) of the samples to create
        dist_samples: Distance (in pixel) between samples in both images
        samples_count: Current number of samples created (will be appended and return)
        num_classes: Number of classes in reference data (will be appended and return)
        samples_file: hdfs file where samples will be written
        dataset: Type of dataset where the samples will be written. Can be either of 'trn' or 'val'
        background_switch: Indicate if samples containing only background pixels will be written or discarded
    """

    # read input and reference images as array
    in_img_array = image_reader_as_array(sat_img)
    label_array = image_reader_as_array(ref_img)

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


def vector_to_raster(vector_file, attribute_name, new_raster):
    """Function to rasterize vector data.
    Args:
        vector_file: Path and name of reference GeoPackage
        attribute_name: Attribute containing the pixel value to write
        new_raster: Raster file where the info will be written
    """
    source_ds = ogr.Open(vector_file)
    source_layer = source_ds.GetLayer()
    name_lyr = source_layer.GetLayerDefn().GetName()
    rev_lyr = source_ds.ExecuteSQL("SELECT * FROM " + name_lyr + " ORDER BY " + attribute_name + " ASC")

    gdal.RasterizeLayer(new_raster, [1], rev_lyr, options=["ATTRIBUTE=%s" % attribute_name])


def main(bucket_name, data_path, samples_size, num_classes, number_of_bands, csv_file, samples_dist,
         remove_background, mask_input_image, mask_reference):
    gpkg_file = []
    if bucket_name:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        bucket.download_file(csv_file, 'samples_prep.csv')
        list_data_prep = read_csv('samples_prep.csv')
        if data_path:
            final_samples_folder = os.path.join(data_path, "samples")
            final_out_label_folder = os.path.join(data_path, "label")
        else:
            final_samples_folder = "samples"
            final_out_label_folder = "label"
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
        img_name = os.path.basename(info['tif']).split('.')[0]
        tmp_label_name = os.path.join(out_label_folder, img_name + "_label_tmp.tif")
        label_name = os.path.join(out_label_folder, img_name + "_label.tif")

        if bucket_name:
            bucket.download_file(info['tif'], "Images/" + info['tif'].split('/')[-1])
            info['tif'] = "Images/" + info['tif'].split('/')[-1]
            if info['gpkg'] not in gpkg_file:
                gpkg_file.append(info['gpkg'])
                bucket.download_file(info['gpkg'], info['gpkg'].split('/')[-1])
            info['gpkg'] = info['gpkg'].split('/')[-1]
        assert_band_number(info['tif'], number_of_bands)

        value_field = info['attribute_name']
        validate_num_classes(info['gpkg'], num_classes, value_field)

        # Mask zeros from input image into label raster.
        if mask_reference:
            tmp_label_raster = create_new_raster_from_base(info['tif'], tmp_label_name, 1)
            vector_to_raster(info['gpkg'], info['attribute_name'], tmp_label_raster)
            tmp_label_raster = None

            masked_array = mask_image(image_reader_as_array(info['tif']), image_reader_as_array(tmp_label_name))
            create_new_raster_from_base(info['tif'], label_name, 1, masked_array)

            os.remove(tmp_label_name)

        else:
            label_raster = create_new_raster_from_base(info['tif'], label_name, 1)
            vector_to_raster(info['gpkg'], info['attribute_name'], label_raster)
            label_raster = None

        # Mask zeros from label raster into input image.
        if mask_input_image:
            masked_img = mask_image(image_reader_as_array(label_name), image_reader_as_array(info['tif']))
            create_new_raster_from_base(label_name, info['tif'], number_of_bands, masked_img)

        if info['dataset'] == 'trn':
            out_file = trn_hdf5
        elif info['dataset'] == 'val':
            out_file = val_hdf5

        number_samples, number_classes = samples_preparation(info['tif'], label_name, samples_size, samples_dist,
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
        try:
            bucket.put_object(Key=os.path.join(data_path, 'samples/', Body=''))
        except:
            pass
        try:
            bucket.put_object(Key='label/', Body='')
        except:
            pass

        trn_samples = open(samples_folder + "/trn_samples.hdf5", 'rb')
        bucket.put_object(Key=final_samples_folder + '/trn_samples.hdf5', Body=trn_samples)
        val_samples = open(samples_folder + "/val_samples.hdf5", 'rb')
        bucket.put_object(Key=final_samples_folder + '/val_samples.hdf5', Body=val_samples)
        # trn labels from out_label_folder
        for f in os.listdir(out_label_folder):
            label = open(os.path.join(out_label_folder, f), 'rb')
            bucket.put_object(Key=os.path.join(final_out_label_folder, f), Body=label)
            os.remove(os.path.join(out_label_folder, f))
        os.remove(samples_folder + "/trn_samples.hdf5")
        os.remove(samples_folder + "/val_samples.hdf5")
        os.remove(info['gpkg'])
        os.remove('samples_prep.csv')
    print("End of process")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample preparation')
    parser.add_argument('ParamFile', metavar='DIR',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    params = read_parameters(args.ParamFile)

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

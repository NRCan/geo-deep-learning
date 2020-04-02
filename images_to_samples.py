import argparse
import datetime
import os
from pathlib import Path

import fiona
import numpy as np
import warnings
import rasterio
import random
import time

from tqdm import tqdm
from collections import OrderedDict

from utils.CreateDataset import create_files_and_datasets, MetaSegmentationDataset
from utils.utils import vector_to_raster, get_key_def, lst_ids
from utils.readers import read_parameters, image_reader_as_array, read_csv
from utils.verifications import is_valid_geom, validate_num_classes

# from rasterio.features import is_valid_geom #FIXME: wait for https://github.com/mapbox/rasterio/issues/1815 to be solved

try:
    import boto3
except ModuleNotFoundError:
    warnings.warn("The boto3 library couldn't be imported. Ignore if not using AWS s3 buckets", ImportWarning)
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


def pad_diff(arr, w, h, arr_shape):
    w_diff = arr_shape - w
    h_diff = arr_shape - h

    if len(arr.shape) > 2:
        padded_arr = np.pad(arr, ((0, w_diff), (0, h_diff), (0, 0)), "constant", constant_values=(0, 0))
    else:
        padded_arr = np.pad(arr, ((0, w_diff), (0, h_diff)), "constant", constant_values=(0, 0))

    return padded_arr

def append_to_dataset(dataset, sample):
    old_size = dataset.shape[0]  # this function always appends samples on the first axis
    dataset.resize(old_size + 1, axis=0)
    dataset[old_size, ...] = sample
    return old_size  # the index to the newly added sample, or the previous size of the dataset


def samples_preparation(in_img_array,
                        label_array,
                        sample_size,
                        overlap,
                        samples_count,
                        num_classes,
                        samples_file,
                        val_sample_file,
                        dataset,
                        min_annotated_percent,
                        image_metadata):
    """
    Extract and write samples from input image and reference image
    :param in_img_array: numpy array of the input image
    :param label_array: numpy array of the annotation image
    :param sample_size: (int) Size (in pixel) of the samples to create #FIXME: could there be a different sample size for tst dataset? shows results closer to inference
    :param overlap: (int) Desired overlap between samples in %
    :param samples_count: (dict) Current number of samples created (will be appended and return)
    :param num_classes: (dict) Number of classes in reference data (will be appended and return)
    :param samples_file: (hdf5 dataset) hdfs file where samples will be written
    :param dataset: (str) Type of dataset where the samples will be written. Can be 'trn' or 'val' or 'tst'
    :param min_annotated_percent: (int) Minimum % of non background pixels in sample, in order to store it
    :param image_metadata: (Ruamel) list of optionnal metadata specified in the associated metadata file
    :return: updated samples count and number of classes.
    """

    # read input and reference images as array

    h, w, num_bands = in_img_array.shape
    if dataset == 'trn':
        idx_samples = samples_count['trn']
    # elif dataset == 'val':
    #     idx_samples = samples_count['val']

    elif dataset == 'tst':
        idx_samples = samples_count['tst']
        print('tst:', idx_samples)
    else:
        raise ValueError(f"Dataset value must be trn or val. Provided value is {dataset}")

    metadata_idx = -1
    if image_metadata:
        # there should be one set of metadata per raster
        # ...all samples created by tiling below will point to that metadata by index
        metadata_idx = append_to_dataset(samples_file["metadata"], repr(image_metadata))

    # half tile padding
    # half_tile = int(sample_size / 2)
    # print(half_tile)
    # pad_in_img_array = np.pad(in_img_array, ((half_tile, half_tile), (half_tile, half_tile), (0, 0)),
    #                           mode='constant')
    # pad_label_array = np.pad(label_array, ((half_tile, half_tile), (half_tile, half_tile), (0, 0)), mode='constant')

    # print('padded label shape:', pad_in_img_array.shape)
    # print('padded image shape:', pad_label_array.shape)

    dist_samples = round(sample_size*(1-(overlap/100)))
    added_samples = 0
    excl_samples = 0
    idx_samples_v = samples_count['val']
    with tqdm(range(0, h, dist_samples), position=1, leave=True, desc='testing slicing strategy' ) as _tqdm:
        for row in _tqdm:
            for column in range(0, w, dist_samples):
                data = (in_img_array[row:row + sample_size, column:column + sample_size, :])
                target = label_array[row:row + sample_size, column:column + sample_size]

                data_row = data.shape[0]
                data_col = data.shape[1]
                if data_row < sample_size or data_col < sample_size:
                    data = pad_diff(data, data_row, data_col, sample_size)

                target_row = target.shape[0]
                target_col = target.shape[1]
                if target_row < sample_size or target_col < sample_size:
                    target = pad_diff(target, target_row, target_col, sample_size)
                u, count = np.unique(target, return_counts=True)
                target_background_percent = count[0] / np.sum(count) * 100 if 0 in u else 0
                if target_background_percent <= 100 - min_annotated_percent: #FIXME: if min_annot_perc is >50%, samples on edges will be excluded

                    if dataset == 'trn':

                        random_val = random.randint(1, 100)

                        if random_val > 5:
                            _samples_file = samples_file
                        else:
                            _samples_file = val_sample_file
                            idx_samples_v += 1
                        append_to_dataset(_samples_file["sat_img"], data)
                        append_to_dataset(_samples_file["map_img"], target)
                        append_to_dataset(_samples_file["meta_idx"], metadata_idx)

                    else:
                        append_to_dataset(samples_file["sat_img"], data)
                        append_to_dataset(samples_file["map_img"], target)
                        append_to_dataset(samples_file["meta_idx"], metadata_idx)

                    idx_samples += 1
                    added_samples += 1
                else:
                    excl_samples += 1

                target_class_num = np.max(u)
                if num_classes < target_class_num:
                    num_classes = target_class_num

                _tqdm.set_postfix(Excld_samples=excl_samples, Added_samples=f'{added_samples}/{len(_tqdm)*len(range(0, w, dist_samples))}', Target_annot_perc=100-target_background_percent)

    if dataset == 'tst':
        print('tst:', idx_samples)
        samples_count['tst'] = idx_samples
    else:
        print('train:', idx_samples)
        print('validation:', idx_samples_v)
        samples_count['trn'] = idx_samples
        samples_count['val'] = idx_samples_v
    # elif dataset == 'val':
    #     print('val:', idx_samples)
    #     samples_count['val'] = idx_samples_v
    # return the appended samples count and number of classes.
    return samples_count, num_classes


def main(params):
    """
    Training and validation datasets preparation.
    :param params: (dict) Parameters found in the yaml config file.

    """
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    bucket_file_cache = []

    assert params['global']['task'] == 'segmentation', f"images_to_samples.py isn't necessary when performing classification tasks"

    # SET BASIC VARIABLES AND PATHS. CREATE OUTPUT FOLDERS.
    bucket_name = params['global']['bucket_name']
    data_path = Path(params['global']['data_path'])
    Path.mkdir(data_path, exist_ok=True, parents=True)
    csv_file = params['sample']['prep_csv_file']
    samples_size = params["global"]["samples_size"]
    overlap = params["sample"]["overlap"]
    min_annot_perc = params['sample']['min_annotated_percent']
    num_bands = params['global']['number_of_bands']
    debug = get_key_def('debug_mode', params['global'], False)
    if debug:
        warnings.warn(f'Debug mode activate. Execution may take longer...')

    final_samples_folder = None
    if bucket_name:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        bucket.download_file(csv_file, 'samples_prep.csv')
        list_data_prep = read_csv('samples_prep.csv')
        if data_path:
            final_samples_folder = os.path.join(data_path, "samples")
        else:
            final_samples_folder = "samples"
        samples_folder = f'samples{samples_size}_overlap{overlap}_min-annot{min_annot_perc}_{num_bands}bands' # TODO: validate this is preferred name structure

    else:
        list_data_prep = read_csv(csv_file)
        samples_folder = data_path.joinpath(f'samples{samples_size}_overlap{overlap}_min-annot{min_annot_perc}_{num_bands}bands')

    if samples_folder.is_dir():
        warnings.warn(f'Data path exists: {samples_folder}. Suffix will be added to directory name.')
        samples_folder = Path(str(samples_folder) + '_' + now)
    else:
        tqdm.write(f'Writing samples to {samples_folder}')
    Path.mkdir(samples_folder, exist_ok=False)    #FIXME: what if we want to append samples to existing hdf5?
    tqdm.write(f'Samples will be written to {samples_folder}\n\n')

    tqdm.write(f'\nSuccessfully read csv file: {Path(csv_file).stem}\nNumber of rows: {len(list_data_prep)}\nCopying first entry:\n{list_data_prep[0]}\n')
    ignore_index = get_key_def('ignore_index', params['training'], -1)

    for info in tqdm(list_data_prep, position=0, desc=f'Asserting existence of tif and gpkg files in csv'):
        assert Path(info['tif']).is_file(), f'Could not locate "{info["tif"]}". ' \
                                            f'Make sure file exists in this directory.'
        assert Path(info['gpkg']).is_file(), f'Could not locate "{info["gpkg"]}". ' \
                                             f'Make sure file exists in this directory.'
    if debug:
        for info in tqdm(list_data_prep, position=0, desc=f"Validating presence of {params['global']['num_classes']} "
                                                          f"classes in attribute \"{info['attribute_name']}\" for vector "
                                                          f"file \"{Path(info['gpkg']).stem}\""):
            validate_num_classes(info['gpkg'], params['global']['num_classes'], info['attribute_name'], ignore_index)
        with tqdm(list_data_prep, position=0, desc=f"Checking validity of features in vector files") as _tqdm:
            invalid_features = {}
            for info in _tqdm:
                # Extract vector features to burn in the raster image
                with fiona.open(info['gpkg'], 'r') as src:  # TODO: refactor as independent function
                    lst_vector = [vector for vector in src]
                shapes = lst_ids(list_vector=lst_vector, attr_name=info['attribute_name'])
                for index, item in enumerate(tqdm([v for vecs in shapes.values() for v in vecs], leave=False, position=1)):
                    # geom must be a valid GeoJSON geometry type and non-empty
                    geom, value = item
                    geom = getattr(geom, '__geo_interface__', None) or geom
                    if not is_valid_geom(geom):
                        gpkg_stem = str(Path(info['gpkg']).stem)
                        if gpkg_stem not in invalid_features.keys(): # create key with name of gpkg
                            invalid_features[gpkg_stem] = []
                        if lst_vector[index]["id"] not in invalid_features[gpkg_stem]: # ignore feature is already appended
                            invalid_features[gpkg_stem].append(lst_vector[index]["id"])
            assert len(invalid_features.values()) == 0, f'Invalid geometry object(s) for "gpkg:ids": \"{invalid_features}\"'

    number_samples = {'trn': 0, 'val': 0, 'tst': 0}
    number_classes = 0

    trn_hdf5, val_hdf5, tst_hdf5 = create_files_and_datasets(params, samples_folder)

    # For each row in csv: (1) burn vector file to raster, (2) read input raster image, (3) prepare samples
    with tqdm(list_data_prep, position=0, leave=False, desc=f'Preparing samples') as _tqdm:
        for info in _tqdm:
            _tqdm.set_postfix(
                OrderedDict(tif=f'{Path(info["tif"]).stem}', sample_size=params['global']['samples_size']))
            # try:
            if bucket_name:
                bucket.download_file(info['tif'], "Images/" + info['tif'].split('/')[-1])
                info['tif'] = "Images/" + info['tif'].split('/')[-1]
                if info['gpkg'] not in bucket_file_cache:
                    bucket_file_cache.append(info['gpkg'])
                    bucket.download_file(info['gpkg'], info['gpkg'].split('/')[-1])
                info['gpkg'] = info['gpkg'].split('/')[-1]
                if info['meta']:
                    if info['meta'] not in bucket_file_cache:
                        bucket_file_cache.append(info['meta'])
                        bucket.download_file(info['meta'], info['meta'].split('/')[-1])
                    info['meta'] = info['meta'].split('/')[-1]

            with rasterio.open(info['tif'], 'r') as raster:
                # Burn vector file in a raster file
                np_label_raster = vector_to_raster(vector_file=info['gpkg'],
                                                   input_image=raster,
                                                   attribute_name=info['attribute_name'],
                                                   fill=get_key_def('ignore_idx', get_key_def('training', params, {}), 0))

                # Read the input raster image
                np_input_image = image_reader_as_array(input_image=raster,
                                                       scale=get_key_def('scale_data', params['global'], None),
                                                       aux_vector_file=get_key_def('aux_vector_file', params['global'], None),
                                                       aux_vector_attrib=get_key_def('aux_vector_attrib', params['global'], None),
                                                       aux_vector_ids=get_key_def('aux_vector_ids', params['global'], None),
                                                       aux_vector_dist_maps=get_key_def('aux_vector_dist_maps', params['global'], True),
                                                       aux_vector_dist_log=get_key_def('aux_vector_dist_log', params['global'], True),
                                                       aux_vector_scale=get_key_def('aux_vector_scale', params['global'], None))

                print('label shape:', np_label_raster.shape)
                print('image shape:', np_input_image.shape)

            # Mask the zeros from input image into label raster.
            # print(params['sample']['mask_reference'])
            if params['sample']['mask_reference']:
                np_label_raster = mask_image(np_input_image, np_label_raster)

            if info['dataset'] == 'trn':
                out_file = trn_hdf5
                val_file = val_hdf5
            # elif info['dataset'] == 'val':
            #     out_file = val_hdf5
            elif info['dataset'] == 'tst':
                out_file = tst_hdf5
            else:
                raise ValueError(f"Dataset value must be trn or val or tst. Provided value is {info['dataset']}")

            meta_map, metadata = get_key_def("meta_map", params["global"], {}), None
            if info['meta'] is not None and isinstance(info['meta'], str) and Path(info['meta']).is_file():
                metadata = read_parameters(info['meta'])

            number_samples, number_classes = samples_preparation(np_input_image,
                                                                 np_label_raster,
                                                                 samples_size,
                                                                 overlap,
                                                                 number_samples,
                                                                 number_classes,
                                                                 out_file,
                                                                 val_file,
                                                                 info['dataset'],
                                                                 min_annot_perc,
                                                                 metadata)

    trn_hdf5.close()
    val_hdf5.close()
    tst_hdf5.close()

    print("Number of samples created: ", number_samples)

    if bucket_name and final_samples_folder:
        print('Transfering Samples to the bucket')
        bucket.upload_file(samples_folder + "/trn_samples.hdf5", final_samples_folder + '/trn_samples.hdf5')
        bucket.upload_file(samples_folder + "/val_samples.hdf5", final_samples_folder + '/val_samples.hdf5')
        bucket.upload_file(samples_folder + "/tst_samples.hdf5", final_samples_folder + '/tst_samples.hdf5')

    print("End of process")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample preparation')
    parser.add_argument('ParamFile', metavar='DIR',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    params = read_parameters(args.ParamFile)
    start_time = time.time()
    tqdm.write(f'\n\nStarting images to samples preparation with {args.ParamFile}\n\n')
    main(params)
    print("Elapsed time:{}".format(time.time() - start_time))

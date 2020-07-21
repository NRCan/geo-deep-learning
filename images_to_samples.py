import argparse
import datetime
import os
import numpy as np
np.random.seed(1234)  # Set random seed for reproducibility
import warnings
import rasterio
import time

from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict

from utils.CreateDataset import create_files_and_datasets
from utils.utils import get_key_def, pad, pad_diff, read_csv, add_metadata_from_raster_to_sample
from utils.geoutils import vector_to_raster
from utils.readers import read_parameters, image_reader_as_array
from utils.verifications import validate_num_classes, assert_num_bands, assert_crs_match, \
    validate_features_from_gpkg

from rasterio.features import is_valid_geom

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


def append_to_dataset(dataset, sample):
    """
    Append a new sample to a provided dataset. The dataset has to be expanded before we can add value to it.
    :param dataset:
    :param sample: data to append
    :return: Index of the newly added sample.
    """
    old_size = dataset.shape[0]  # this function always appends samples on the first axis
    dataset.resize(old_size + 1, axis=0)
    dataset[old_size, ...] = sample
    return old_size


def validate_class_prop_dict(actual_classes_dict, config_dict):
    """
    Populate dictionary containing class values found in vector data with values (thresholds) from sample/class_prop
    parameter in config file

    actual_classes_dict: dict
        Dictionary where each key is a class found in vector data. Value is not relevant (should be 0)

    config_dict:
        Dictionary with class ids (keys and thresholds (values) from config file

    """
    # Validation of class proportion parameters (assert types).
    if not isinstance(config_dict, dict):
        warnings.warn(f"Class_proportion parameter should be a dictionary. Got type {type(config_dict)}. "
                      f"Ignore if parameter was omitted)")
        return None

    for key, value in config_dict.items():
        try:
            assert isinstance(key, str)
            int(key)
        except (ValueError, AssertionError):
            f"Class should be a string castable as an integer. Got {key} of type {type(key)}"
        assert isinstance(value, int), f"Class value should be an integer, got {value} of type {type(value)}"

    # Populate actual classes dictionary with values from config
    for key, value in config_dict.items():
        if int(key) in actual_classes_dict.keys():
            actual_classes_dict[int(key)] = value
        else:
            warnings.warn(f"Class {key} not found in provided vector data.")

    return actual_classes_dict.copy()


def minimum_annotated_percent(target_background_percent, min_annotated_percent):
    if not min_annotated_percent:
        return True
    elif float(target_background_percent) <= 100 - min_annotated_percent:
        return True

    return False


def class_proportion(target, sample_size: int, class_min_prop: dict):
    if not class_min_prop:
        return True
    sample_total = sample_size ** 2
    for key, value in class_min_prop.items():
        if key not in np.unique(target):
            target_prop_classwise = 0
        else:
            target_prop_classwise = (round((np.bincount(target.clip(min=0).flatten())[key] / sample_total) * 100, 1))
        if target_prop_classwise < value:
            return False
    return True


def add_to_datasets(dataset,
                    samples_file,
                    val_percent,
                    val_sample_file,
                    data,
                    target,
                    sample_metadata,
                    metadata_idx,
                    dict_classes):
    """ Add sample to Hdf5 (trn, val or tst) and computes pixel classes(%). """
    val = False
    if dataset == 'trn':
        random_val = np.random.randint(1, 100)
        if random_val > val_percent:
            pass
        else:
            val = True
            samples_file = val_sample_file
    append_to_dataset(samples_file["sat_img"], data)
    append_to_dataset(samples_file["map_img"], target)
    append_to_dataset(samples_file["sample_metadata"], repr(sample_metadata))
    append_to_dataset(samples_file["meta_idx"], metadata_idx)

    # adds pixel count to pixel_classes dict for each class in the image
    for key, value in enumerate(np.bincount(target.clip(min=0).flatten())):
        cls_keys = dict_classes.keys()
        if key in cls_keys:
            dict_classes[key] += value
        elif key not in cls_keys and value > 0:
            raise ValueError(f"A class value was written ({key}) that was not defined in the classes ({cls_keys}).")

    return val


def samples_preparation(in_img_array,
                        label_array,
                        sample_size,
                        overlap,
                        samples_count,
                        num_classes,
                        samples_file,
                        val_percent,
                        val_sample_file,
                        dataset,
                        pixel_classes,
                        image_metadata=None,
                        dontcare=0,
                        min_annot_perc=None,
                        class_prop=None):
    """
    Extract and write samples from input image and reference image
    :param in_img_array: numpy array of the input image
    :param label_array: numpy array of the annotation image
    :param sample_size: (int) Size (in pixel) of the samples to create # TODO: could there be a different sample size for tst dataset? shows results closer to inference
    :param overlap: (int) Desired overlap between samples in %
    :param samples_count: (dict) Current number of samples created (will be appended and return)
    :param num_classes: (dict) Number of classes in reference data (will be appended and return)
    :param samples_file: (hdf5 dataset) hdfs file where samples will be written
    :param val_percent: (int) percentage of validation samples
    :param val_sample_file: (hdf5 dataset) hdfs file where samples will be written (val)
    :param dataset: (str) Type of dataset where the samples will be written. Can be 'trn' or 'val' or 'tst'
    :param pixel_classes: (dict) samples pixel statistics
    :param image_metadata: (dict) metadata associated to source raster
    :param dontcare: Value in gpkg features that will ignored during training
    :param min_annot_perc: optional, minimum annotated percent required for sample to be created
    :param class_prop: optional, minimal proportion of pixels for each class required for sample to be created
    :return: updated samples count and number of classes.
    """

    # read input and reference images as array
    h, w, num_bands = in_img_array.shape
    if dataset == 'trn':
        idx_samples = samples_count['trn']
        append_to_dataset(val_sample_file["metadata"], repr(image_metadata))
    elif dataset == 'tst':
        idx_samples = samples_count['tst']
    else:
        raise ValueError(f"Dataset value must be trn or tst. Provided value is {dataset}")

    idx_samples_v = samples_count['val']

    # Adds raster metadata to the dataset. All samples created by tiling below will point to that metadata by index
    metadata_idx = append_to_dataset(samples_file["metadata"], repr(image_metadata))

    dist_samples = round(sample_size * (1 - (overlap / 100)))
    added_samples = 0
    excl_samples = 0

    with tqdm(range(0, h, dist_samples), position=1, leave=True,
              desc=f'Writing samples. Dataset currently contains {idx_samples} '
                   f'samples') as _tqdm:

        for row in _tqdm:
            for column in range(0, w, dist_samples):
                data = (in_img_array[row:row + sample_size, column:column + sample_size, :])
                target = np.squeeze(label_array[row:row + sample_size, column:column + sample_size, :], axis=2)
                data_row = data.shape[0]
                data_col = data.shape[1]
                if data_row < sample_size or data_col < sample_size:
                    padding = pad_diff(data_row, data_col, sample_size)  # array, actual height, actual width, desired size
                    data = pad(data, padding, fill=np.nan)  # don't fill with 0 if possible. Creates false min value when scaling.

                target_row = target.shape[0]
                target_col = target.shape[1]
                if target_row < sample_size or target_col < sample_size:
                    padding = pad_diff(target_row, target_col, sample_size)  # array, actual height, actual width, desired size
                    target = pad(target, padding, fill=dontcare)
                u, count = np.unique(target, return_counts=True)
                target_background_percent = round(count[0] / np.sum(count) * 100 if 0 in u else 0, 1)

                sample_metadata = {'sample_indices': (row, column)}

                val = False
                if minimum_annotated_percent(target_background_percent, min_annot_perc) and \
                        class_proportion(target, sample_size, class_prop):
                    val = add_to_datasets(dataset=dataset,
                                          samples_file=samples_file,
                                          val_percent=val_percent,
                                          val_sample_file=val_sample_file,
                                          data=data,
                                          target=target,
                                          sample_metadata=sample_metadata,
                                          metadata_idx=metadata_idx,
                                          dict_classes=pixel_classes)
                    if val:
                        idx_samples_v += 1
                    else:
                        idx_samples += 1
                    added_samples += 1
                else:
                    excl_samples += 1

                target_class_num = np.max(u)
                if num_classes < target_class_num:
                    num_classes = target_class_num

                final_dataset = 'val' if val else dataset
                _tqdm.set_postfix(Dataset=final_dataset,
                                  Excld_samples=excl_samples,
                                  Added_samples=f'{added_samples}/{len(_tqdm) * len(range(0, w, dist_samples))}',
                                  Target_annot_perc=100 - target_background_percent)

    assert added_samples > 0, "No sample added for current raster. Problems may occur with use of metadata"
    if dataset == 'tst':
        samples_count['tst'] = idx_samples
    else:
        samples_count['trn'] = idx_samples
        samples_count['val'] = idx_samples_v
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
    bucket_name = get_key_def('bucket_name', params['global'])
    data_path = Path(params['global']['data_path'])
    Path.mkdir(data_path, exist_ok=True, parents=True)
    csv_file = params['sample']['prep_csv_file']
    val_percent = params['sample']['val_percent']
    samples_size = params["global"]["samples_size"]
    overlap = params["sample"]["overlap"]
    min_annot_perc = get_key_def('min_annotated_percent', params['sample']['sampling_method'], None, expected_type=int)
    num_bands = params['global']['number_of_bands']
    debug = get_key_def('debug_mode', params['global'], False)
    if debug:
        warnings.warn(f'Debug mode activate. Execution may take longer...')

    final_samples_folder = None

    sample_path_name = f'samples{samples_size}_overlap{overlap}_min-annot{min_annot_perc}_{num_bands}bands'

    # AWS
    if bucket_name:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        bucket.download_file(csv_file, 'samples_prep.csv')
        list_data_prep = read_csv('samples_prep.csv')
        if data_path:
            final_samples_folder = data_path.joinpath("samples")
        else:
            final_samples_folder = "samples"
        samples_folder = sample_path_name

    else:
        list_data_prep = read_csv(csv_file)
        samples_folder = data_path.joinpath(sample_path_name)

    if samples_folder.is_dir():
        warnings.warn(f'Data path exists: {samples_folder}. Suffix will be added to directory name.')
        samples_folder = Path(str(samples_folder) + '_' + now)
    else:
        tqdm.write(f'Writing samples to {samples_folder}')
    Path.mkdir(samples_folder, exist_ok=False)  # TODO: what if we want to append samples to existing hdf5?
    tqdm.write(f'Samples will be written to {samples_folder}\n\n')

    tqdm.write(f'\nSuccessfully read csv file: {Path(csv_file).stem}\n'
               f'Number of rows: {len(list_data_prep)}\n'
               f'Copying first entry:\n{list_data_prep[0]}\n')
    ignore_index = get_key_def('ignore_index', params['training'], -1)
    meta_map, metadata = get_key_def("meta_map", params["global"], {}), None

    # VALIDATION: (1) Assert num_classes parameters == num actual classes in gpkg and (2) check CRS match (tif and gpkg)
    valid_gpkg_set = set()
    for info in tqdm(list_data_prep, position=0):
        assert_num_bands(info['tif'], num_bands, meta_map)
        if info['gpkg'] not in valid_gpkg_set:
            gpkg_classes = validate_num_classes(info['gpkg'], params['global']['num_classes'], info['attribute_name'],
                                                ignore_index)
            assert_crs_match(info['tif'], info['gpkg'])
            valid_gpkg_set.add(info['gpkg'])

    if debug:
        # VALIDATION (debug only): Checking validity of features in vector files
        for info in tqdm(list_data_prep, position=0, desc=f"Checking validity of features in vector files"):
            invalid_features = validate_features_from_gpkg(info['gpkg'], info['attribute_name'])  # TODO: test this with invalid features.
            assert not invalid_features, f"{info['gpkg']}: Invalid geometry object(s) '{invalid_features}'"

    number_samples = {'trn': 0, 'val': 0, 'tst': 0}
    number_classes = 0

    class_prop = get_key_def('class_proportion', params['sample']['sampling_method'], None, expected_type=dict)

    trn_hdf5, val_hdf5, tst_hdf5 = create_files_and_datasets(params, samples_folder)

    # Set dontcare (aka ignore_index) value
    dontcare = get_key_def("ignore_index", params["training"], -1)  # TODO: deduplicate with train_segmentation, l300
    if dontcare == 0:
        warnings.warn("The 'dontcare' value (or 'ignore_index') used in the loss function cannot be zero;"
                      " all valid class indices should be consecutive, and start at 0. The 'dontcare' value"
                      " will be remapped to -1 while loading the dataset, and inside the config from now on.")
        params["training"]["ignore_index"] = -1

    # creates pixel_classes dict and keys
    pixel_classes = {key: 0 for key in gpkg_classes}
    background_val = 0
    pixel_classes[background_val] = 0
    class_prop = validate_class_prop_dict(pixel_classes, class_prop)
    pixel_classes[dontcare] = 0

    # For each row in csv: (1) burn vector file to raster, (2) read input raster image, (3) prepare samples
    with tqdm(list_data_prep, position=0, leave=False, desc=f'Preparing samples') as _tqdm:
        for info in _tqdm:
            _tqdm.set_postfix(
                OrderedDict(tif=f'{Path(info["tif"]).stem}', sample_size=params['global']['samples_size']))
            try:
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
                    # 1. Read the input raster image
                    np_input_image, raster, dataset_nodata = image_reader_as_array(
                        input_image=raster,
                        clip_gpkg=info['gpkg'],
                        aux_vector_file=get_key_def('aux_vector_file', params['global'], None),
                        aux_vector_attrib=get_key_def('aux_vector_attrib', params['global'], None),
                        aux_vector_ids=get_key_def('aux_vector_ids', params['global'], None),
                        aux_vector_dist_maps=get_key_def('aux_vector_dist_maps', params['global'], True),
                        aux_vector_dist_log=get_key_def('aux_vector_dist_log', params['global'], True),
                        aux_vector_scale=get_key_def('aux_vector_scale', params['global'], None))

                    # 2. Burn vector file in a raster file
                    np_label_raster = vector_to_raster(vector_file=info['gpkg'],
                                                       input_image=raster,
                                                       out_shape=np_input_image.shape[:2],
                                                       attribute_name=info['attribute_name'],
                                                       fill=background_val)  # background value in rasterized vector.

                    if dataset_nodata is not None:
                        # 3. Set ignore_index value in label array where nodata in raster (only if nodata across all bands)
                        np_label_raster[dataset_nodata] = dontcare

                if debug:
                    out_meta = raster.meta.copy()
                    np_image_debug = np_input_image.transpose(2, 0, 1).astype(out_meta['dtype'])
                    out_meta.update({"driver": "GTiff",
                                     "height": np_image_debug.shape[1],
                                     "width": np_image_debug.shape[2]})
                    out_tif = samples_folder / f"np_input_image_{_tqdm.n}.tif"
                    print(f"DEBUG: writing clipped raster to {out_tif}")
                    with rasterio.open(out_tif, "w", **out_meta) as dest:
                        dest.write(np_image_debug)

                    out_meta = raster.meta.copy()
                    np_label_debug = np.expand_dims(np_label_raster, axis=2).transpose(2, 0, 1).astype(out_meta['dtype'])
                    out_meta.update({"driver": "GTiff",
                                     "height": np_label_debug.shape[1],
                                     "width": np_label_debug.shape[2],
                                     'count': 1})
                    out_tif = samples_folder / f"np_label_rasterized_{_tqdm.n}.tif"
                    print(f"DEBUG: writing final rasterized gpkg to {out_tif}")
                    with rasterio.open(out_tif, "w", **out_meta) as dest:
                        dest.write(np_label_debug)

                # Mask the zeros from input image into label raster.
                if params['sample']['mask_reference']:
                    np_label_raster = mask_image(np_input_image, np_label_raster)

                if info['dataset'] == 'trn':
                    out_file = trn_hdf5
                elif info['dataset'] == 'tst':
                    out_file = tst_hdf5
                else:
                    raise ValueError(f"Dataset value must be trn or tst. Provided value is {info['dataset']}")
                val_file = val_hdf5

                metadata = add_metadata_from_raster_to_sample(sat_img_arr=np_input_image,
                                                              raster_handle=raster,
                                                              meta_map=meta_map,
                                                              raster_info=info)
                # Save label's per class pixel count to image metadata
                metadata['source_label_bincount'] = {class_num: count for class_num, count in
                                                          enumerate(np.bincount(np_label_raster.clip(min=0).flatten()))
                                                     if count > 0}  # TODO: add this to add_metadata_from[...] function?

                np_label_raster = np.reshape(np_label_raster, (np_label_raster.shape[0], np_label_raster.shape[1], 1))
                # 3. Prepare samples!
                number_samples, number_classes = samples_preparation(in_img_array=np_input_image,
                                                                     label_array=np_label_raster,
                                                                     sample_size=samples_size,
                                                                     overlap=overlap,
                                                                     samples_count=number_samples,
                                                                     num_classes=number_classes,
                                                                     samples_file=out_file,
                                                                     val_percent=val_percent,
                                                                     val_sample_file=val_file,
                                                                     dataset=info['dataset'],
                                                                     pixel_classes=pixel_classes,
                                                                     image_metadata=metadata,
                                                                     dontcare=dontcare,
                                                                     min_annot_perc=min_annot_perc,
                                                                     class_prop=class_prop)

                _tqdm.set_postfix(OrderedDict(number_samples=number_samples))
                out_file.flush()
            except OSError as e:
                warnings.warn(f'An error occurred while preparing samples with "{Path(info["tif"]).stem}" (tiff) and '
                              f'{Path(info["gpkg"]).stem} (gpkg). Error: "{e}"')
                continue

    trn_hdf5.close()
    val_hdf5.close()
    tst_hdf5.close()

    pixel_total = 0
    # adds up the number of pixels for each class in pixel_classes dict
    for i in pixel_classes:
        pixel_total += pixel_classes[i]

    # prints the proportion of pixels of each class for the samples created
    for i in pixel_classes:
        prop = round((pixel_classes[i] / pixel_total) * 100, 1) if pixel_total > 0 else 0
        print('Pixels from class', i, ':', prop, '%')

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

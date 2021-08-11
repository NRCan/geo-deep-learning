# region imports
import argparse
from datetime import datetime
import logging
from typing import List

import numpy as np
np.random.seed(1234)  # Set random seed for reproducibility
import rasterio
import time
import shutil
import uuid

from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict

from utils.create_dataset import create_files_and_datasets, append_to_dataset
from utils.utils import get_key_def, pad, pad_diff, read_csv, add_metadata_from_raster_to_sample, get_git_hash
from utils.geoutils import vector_to_raster
from utils.readers import read_parameters, image_reader_as_array
from utils.verifications import validate_num_classes, validate_raster, assert_crs_match, \
    validate_features_from_gpkg

try:
    import boto3
except ModuleNotFoundError:
    logging.warning("The boto3 library couldn't be imported. Ignore if not using AWS s3 buckets", ImportWarning)
    pass

logging.getLogger(__name__)
# endregion

def mask_image(arrayA, arrayB):
    """Function to mask values of arrayB, based on 0 values from arrayA.

    >>> x1 = np.array([0, 2, 4, 6, 0, 3, 9, 8], dtype=np.uint8).reshape(2,2,2)
    >>> x2 = np.array([1.5, 1.2, 1.6, 1.2, 11., 1.1, 25.9, 0.1], dtype=np.float32).reshape(2,2,2)
    >>> mask_image(x1, x2)
    array([[[ 0. ,  0. ],
            [ 1.6,  1.2]],
    <BLANKLINE>
           [[ 0. ,  0. ],
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
    if not config_dict:
        return None
    elif not isinstance(config_dict, dict):
        logging.warning(f"Class_proportion parameter should be a dictionary. Got type {type(config_dict)}")
        return None

    for key, value in config_dict.items():
        if not isinstance(key, str):
            raise TypeError(f"Class should be a string. Got {key} of type {type(key)}")
        try:
            int(key)
        except ValueError:
            raise ValueError('Class should be castable to an integer')
        if not isinstance(value, int):
            raise ValueError(f"Class value should be an integer, got {value} of type {type(value)}")

    # Populate actual classes dictionary with values from config
    for key, value in config_dict.items():
        if int(key) in actual_classes_dict.keys():
            actual_classes_dict[int(key)] = value
        else:
            logging.warning(f"Class {key} not found in provided vector data.")

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
                    dict_classes,
                    stratification_bias=0,
                    stratification_dict=None): # TODO: what is stratification?
    """ Add sample to Hdf5 (trn, val or tst) and computes pixel classes(%). """
    to_val_set = False
    if dataset == 'trn':
        random_val = np.random.randint(1, 100)
        if random_val > val_percent + stratification_bias:
            if stratification_dict is not None:
                stratification_dict['latest_assignment'] = 'trn'
        else:
            to_val_set = True
            samples_file = val_sample_file
            if stratification_dict is not None:
                stratification_dict['latest_assignment'] = 'val'
    append_to_dataset(samples_file["sat_img"], data)
    append_to_dataset(samples_file["map_img"], target)
    append_to_dataset(samples_file["sample_metadata"], repr(sample_metadata))
    append_to_dataset(samples_file["meta_idx"], metadata_idx)

    # adds pixel count to pixel_classes dict for each class in the image
    class_vals, counts = np.unique(target, return_counts=True)
    for i in range(len(class_vals)):
        class_val = class_vals[i]
        count = counts[i]
        dict_classes[class_val] += count
        if class_val not in dict_classes.keys():
            logging.error(f'Sample contains value "{class_val}" not defined in the classes ({dict_classes.keys()}).')
    return to_val_set

def samples_preparation(coords,
                        tracker_hdf5,
                        in_img_array,
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
                        dontcare,
                        image_metadata=None,
                        min_annot_perc=None,
                        class_prop=None,
                        stratd=None):
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

    if overlap > 25:
         logging.warning("high overlap >25%, note that automatic train/val split creates very similar samples in both sets")
    dist_samples = round(sample_size * (1 - (overlap / 100)))
    added_samples = 0
    excl_samples = 0

    # region Calc Lat&Long for visualisation
    # Nrows = np.ceil(w / dist_samples) # = equivalent to len(range(0, h, dist_samples))
    # Ncols = np.ceil(w / dist_samples) # = equivalent to len(range(0, w, dist_samples))
    # half_overlap_dist = (256 * 0.25) / 2
    long_per_pxl = (coords['n'] - coords['s']) / h
    lat_per_pxl = (coords['e'] - coords['w']) / w
    # endregion


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
                    # TODO track padded samples
                    padding = pad_diff(data_row, data_col, sample_size, sample_size)  # array, actual height, actual width, desired size
                    data = pad(data, padding, fill=np.nan)  # don't fill with 0 if possible. Creates false min value when scaling.

                target_row = target.shape[0]
                target_col = target.shape[1]
                if target_row < sample_size or target_col < sample_size:
                    padding = pad_diff(target_row, target_col, sample_size, sample_size)  # array, actual height, actual width, desired size
                    target = pad(target, padding, fill=dontcare)
                # u, count = np.unique(target, return_counts=True)
                # target_background_percent = round(count[0] / np.sum(count) * 100 if 0 in u else 0, 1)
                backgr_ct = np.sum(target == 0)
                backgr_ct += np.sum(target == dontcare)
                target_background_percent = round(backgr_ct / target.size * 100, 1)

                sample_metadata = {'sample_indices': (row, column)}

                # region Stratification bias
                u, count = np.unique(target, return_counts=True)

                if (stratd is not None) and (dataset == 'trn'):
                    tile_size = target.size
                    tile_counts = {x: y for x, y in zip(u, count)}
                    tile_props = {x: y / tile_size for x, y in zip(u, count)}
                    for key in tile_props.keys():
                        if key not in stratd['trn']['total_counts']:
                            stratd['trn']['total_counts'][key] = 0
                        if key not in stratd['val']['total_counts']:
                            stratd['val']['total_counts'][key] = 0
                    if stratd['trn']['total_pixels'] == 0:
                        stratd['trn']['total_props'] = {key: 0.0 for key in stratd['trn']['total_counts'].keys()}
                    else:
                        stratd['trn']['total_props'] = {key: val / stratd['trn']['total_pixels']
                                                        for key, val in stratd['trn']['total_counts'].items()}
                    if stratd['val']['total_pixels'] == 0:
                        stratd['val']['total_props'] = {key: 0.0 for key in stratd['val']['total_counts'].keys()}
                    else:
                        stratd['val']['total_props'] = {key: val / stratd['val']['total_pixels']
                                                        for key, val in stratd['val']['total_counts'].items()}
                    distances_trn = {key: np.abs(val - stratd['trn']['total_props'][key])
                                     for key, val in tile_props.items()}
                    distances_val = {key: np.abs(val - stratd['val']['total_props'][key])
                                     for key, val in tile_props.items()}
                    dist_trn = np.mean(np.array(list(distances_trn.values()))**2)
                    dist_val = np.mean(np.array(list(distances_val.values()))**2)
                    dist = dist_val - dist_trn
                    stratification_bias = stratd['strat_factor'] * np.sign(dist)
                else:
                    stratification_bias = 0.0
                # endregion

                val = False
                if minimum_annotated_percent(target_background_percent, min_annot_perc) and class_proportion(target, sample_size, class_prop):
                    val = add_to_datasets(dataset=dataset,
                                          samples_file=samples_file,
                                          val_percent=val_percent,
                                          val_sample_file=val_sample_file,
                                          data=data,
                                          target=target,
                                          sample_metadata=sample_metadata,
                                          metadata_idx=metadata_idx,
                                          dict_classes=pixel_classes,
                                          stratification_bias=stratification_bias,
                                          stratification_dict=stratd)
                    if val:
                        tracker_hdf5['val/projection'].resize(tracker_hdf5['val/projection'].shape[0]+1, axis=0)
                        tracker_hdf5['val/projection'][tracker_hdf5['val/projection'].shape[0]-1, ...] = coords['projection'] # TODO: get rid of projection for less data-intence method ->mb add in 'csv row' dataset
                        tracker_hdf5['val/coords'].resize(tracker_hdf5['val/coords'].shape[0]+1, axis=0)
                        tracker_hdf5['val/coords'][tracker_hdf5['val/coords'].shape[0]-1, ...] = (coords["w"] +(lat_per_pxl * column),
                                                                                                  coords["w"] +(lat_per_pxl * (column+sample_size)),
                                                                                                  coords["n"] -(long_per_pxl * row),
                                                                                                  coords["n"] -(long_per_pxl * (row+sample_size)))
                        # tracker.idx_samples_v += 1
                        idx_samples_v += 1
                    else:
                        tracker_hdf5[dataset+'/projection'].resize(tracker_hdf5[dataset+'/projection'].shape[0]+1, axis=0)
                        tracker_hdf5[dataset+'/projection'][tracker_hdf5[dataset+'/projection'].shape[0]-1, ...] = coords['projection']
                        tracker_hdf5[dataset+'/coords'].resize(tracker_hdf5[dataset+'/coords'].shape[0]+1, axis=0)
                        tracker_hdf5[dataset+'/coords'][tracker_hdf5[dataset+'/coords'].shape[0]-1, ...] = (coords["w"] +(lat_per_pxl * column),
                                                                                                            coords["w"] +(lat_per_pxl * (column+sample_size)),
                                                                                                            coords["n"] -(long_per_pxl * row),
                                                                                                            coords["n"] -(long_per_pxl * (row+sample_size)))
                        #
                        # if dataset == 'tst': tracker.tst_samples += 1
                        # else:                tracker.trn_samples += 1
                        idx_samples += 1
                    added_samples += 1

                    # region Stratification update
                    if (stratd is not None) and (dataset == 'trn'):
                        for key, val in tile_counts.items():
                            stratd[stratd['latest_assignment']]['total_counts'][key] += val
                        stratd[stratd['latest_assignment']]['total_pixels'] += tile_size
                    # endregion
                else:
                    # tracker.excl_samples += 1
                    excl_samples += 1

                target_class_num = np.max(target) # TODO: still good? old ver = np.max(u)
                if num_classes < target_class_num:
                    num_classes = target_class_num

                final_dataset = 'val' if val else dataset
                logging.debug(f'Dset={final_dataset}, '
                              f'Added samps={added_samples}/{len(_tqdm) * len(range(0, w, dist_samples))}, '
                              f'Excld samps={excl_samples}/{len(_tqdm) * len(range(0, w, dist_samples))}, ' 
                              f'Target annot perc={100 - target_background_percent:.1f}')

    if added_samples == 0:
        logging.warning(f"No sample added for current raster. Problems may occur with use of metadata")
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

    Process
    -------
    1. Read csv file and validate existence of all input files and GeoPackages.

    2. Do the following verifications:
        1. Assert number of bands found in raster is equal to desired number
           of bands.
        2. Check that `num_classes` is equal to number of classes detected in
           the specified attribute for each GeoPackage.
           Warning: this validation will not succeed if a Geopackage
                    contains only a subset of `num_classes` (e.g. 3 of 4).
        3. Assert Coordinate reference system between raster and gpkg match.

    3. Read csv file and for each line in the file, do the following:
        1. Read input image as array with utils.readers.image_reader_as_array().
             - If gpkg's extent is smaller than raster's extent,
              raster is clipped to gpkg's extent.
            - If gpkg's extent is bigger than raster's extent,
              gpkg is clipped to raster's extent.
        2. Convert GeoPackage vector information into the "label" raster with
           utils.utils.vector_to_raster(). The pixel value is determined by the
           attribute in the csv file.
        3. Create a new raster called "label" with the same properties as the
           input image.
        4. Read metadata and add to input as new bands (*more details to come*).
        5. Crop the arrays in smaller samples of the size `samples_size` of
           `your_conf.yaml`. Visual representation of this is provided at
            https://medium.com/the-downlinq/broad-area-satellite-imagery-semantic-segmentation-basiss-4a7ea2c8466f
        6. Write samples from input image and label into the "val", "trn" or
           "tst" hdf5 file, depending on the value contained in the csv file.
            Refer to samples_preparation().

    -------
    :param params: (dict) Parameters found in the yaml config file.
    """
    start_time = time.time()

    # region PARAMS
    # MANDATORY PARAMETERS
    num_classes = get_key_def('num_classes', params['global'], expected_type=int)
    num_bands = get_key_def('number_of_bands', params['global'], expected_type=int)
    csv_file = get_key_def('prep_csv_file', params['sample'], expected_type=str)

    # OPTIONAL PARAMETERS
    # basics
    debug = get_key_def('debug_mode', params['global'], False)

    task = get_key_def('task', params['global'], 'segmentation', expected_type=str)
    if task == 'classification':
        raise ValueError(f"Got task {task}. Expected 'segmentation'.")
    elif not task == 'segmentation':
        raise ValueError(f"images_to_samples.py isn't necessary for classification tasks")
    data_path = Path(get_key_def('data_path', params['global'], './data', expected_type=str))
    Path.mkdir(data_path, exist_ok=True, parents=True)
    val_percent = get_key_def('val_percent', params['sample'], default=10, expected_type=int)

    # mlflow logging
    mlflow_uri = get_key_def('mlflow_uri', params['global'], default="./mlruns")
    experiment_name = get_key_def('mlflow_experiment_name', params['global'], default='gdl-training', expected_type=str)

    # parameters to set hdf5 samples directory
    data_path = Path(get_key_def('data_path', params['global'], './data', expected_type=str))
    samples_size = get_key_def("samples_size", params["global"], default=1024, expected_type=int)
    overlap = get_key_def("overlap", params["sample"], default=5, expected_type=int)
    min_annot_perc = get_key_def('min_annotated_percent', params['sample']['sampling_method'], default=0,
                                 expected_type=int)
    if not data_path.is_dir():
        raise FileNotFoundError(f'Could not locate data path {data_path}')
    samples_folder_name = (f'samples{samples_size}_overlap{overlap}_min-annot{min_annot_perc}_{num_bands}bands'
                           f'_{experiment_name}')

    # other optional parameters
    dontcare = get_key_def("ignore_index", params["training"], -1)
    meta_map = get_key_def('meta_map', params['global'], default={})
    metadata = None
    targ_ids = get_key_def('target_ids', params['sample'], None, expected_type=List)
    class_prop = get_key_def('class_proportion', params['sample']['sampling_method'], None, expected_type=dict)
    mask_reference = get_key_def('mask_reference', params['sample'], default=False, expected_type=bool)

    if get_key_def('use_stratification', params['sample'], False) is not False:
        stratd = {'trn': {'total_pixels': 0, 'total_counts': {}, 'total_props': {}},
                  'val': {'total_pixels': 0, 'total_counts': {}, 'total_props': {}},
                  'strat_factor': params['sample']['use_stratification']}
    else:
        stratd = None

    # add git hash from current commit to parameters if available. Parameters will be saved to hdf5s
    params['global']['git_hash'] = get_git_hash()
    # endregion

    # region init SAMPLES folder & AWS
    # AWS
    final_samples_folder = None
    bucket_name = get_key_def('bucket_name', params['global'])
    bucket_file_cache = []
    if bucket_name:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        bucket.download_file(csv_file, 'samples_prep.csv')
        list_data_prep = read_csv('samples_prep.csv')
    else:
        list_data_prep = read_csv(csv_file)

    smpls_dir = data_path.joinpath(samples_folder_name)
    if smpls_dir.is_dir():
        if debug:
            # Move existing data folder with a random suffix.
            last_mod_time_suffix = datetime.fromtimestamp(smpls_dir.stat().st_mtime).strftime('%Y%m%d-%H%M%S')
            shutil.move(smpls_dir, data_path.joinpath(f'{str(smpls_dir)}_{last_mod_time_suffix}'))
        else:
            raise FileExistsError(f'Data path exists: {smpls_dir}. Remove it or use a different experiment_name.')
    Path.mkdir(smpls_dir, exist_ok=False)  # TODO: what if we want to append samples to existing hdf5?
    # endregion

    # region init logging
    import logging.config  # See: https://docs.python.org/2.4/lib/logging-config-fileformat.html
    log_config_path = Path('utils/logging.conf').absolute()
    # log_config_path = str(log_config_path).replace('\\', '\\\\')
    # for letter in str(log_config_path):
    #     print(letter,end='')
    console_level_logging = 'INFO' if not debug else 'DEBUG'
    if params['global']['my_comp']:
        logging.config.fileConfig(log_config_path, defaults={'logfilename': f'{"D:/NRCan_data/MECnet_implementation/runs/"}/{samples_folder_name}.log',
                                                             'logfilename_debug':
                                                                 f'{"D:/NRCan_data/MECnet_implementation/runs/"}/{samples_folder_name}_debug.log',
                                                             'console_level': console_level_logging})
    else:
        logging.config.fileConfig(log_config_path, defaults={'logfilename': f'{smpls_dir}/{samples_folder_name}.log',
                                                     'logfilename_debug':
                                                         f'{smpls_dir}/{samples_folder_name}_debug.log',
                                                     'console_level': console_level_logging})

    if debug:
        logging.warning(f'Debug mode activated. Some debug features may mobilize extra disk space and '
                        f'cause delays in execution.')

    logging.info(f'\n\tSuccessfully read csv file: {Path(csv_file).stem}\n'
                 f'\tNumber of rows: {len(list_data_prep)}\n'
                 f'\tCopying first entry:\n{list_data_prep[0]}\n')

    logging.info(f'Samples will be written to {smpls_dir}\n\n')

    # Set dontcare (aka ignore_index) value
    if dontcare == 0:
        logging.warning("The 'dontcare' value (or 'ignore_index') used in the loss function cannot be zero;"
                        " all valid class indices should be consecutive, and start at 0. The 'dontcare' value"
                        " will be remapped to -1 while loading the dataset, and inside the config from now on.")
        dontcare = -1
    # endregion

    # Assert that all items in target_ids are integers (ex.: single-class samples from multi-class label)
    if targ_ids:
        for item in targ_ids:
            if not isinstance(item, int):
                raise ValueError(f'Target id "{item}" in target_ids is {type(item)}, expected int.')

    # region VALIDATION
    # (1) Assert num_classes parameters == num actual classes in gpkg
    # (2) check CRS match (tif and gpkg)
    valid_gpkg_set = set()
    for info in tqdm(list_data_prep, position=0):
        validate_raster(info['tif'], num_bands, meta_map)
        if info['gpkg'] not in valid_gpkg_set:
            gpkg_classes = validate_num_classes(info['gpkg'],
                                                num_classes,
                                                info['attribute_name'],
                                                dontcare,
                                                target_ids=targ_ids)
            assert_crs_match(info['tif'], info['gpkg'])
            valid_gpkg_set.add(info['gpkg'])

    if debug:
        # VALIDATION (debug only): Checking validity of features in vector files
        for info in tqdm(list_data_prep, position=0, desc=f"Checking validity of features in vector files"):
            # TODO: make unit to test this with invalid features.
            invalid_features = validate_features_from_gpkg(info['gpkg'], info['attribute_name'])
            if invalid_features:
                logging.critical(f"{info['gpkg']}: Invalid geometry object(s) '{invalid_features}'")
    # endregion

    # region init dataset & class Variables
    number_samples = {'trn': 0, 'val': 0, 'tst': 0}
    number_classes = 0

    trn_hdf5, val_hdf5, tst_hdf5, tracker_hdf5 = create_files_and_datasets(samples_size=samples_size,
                                                                           number_of_bands=num_bands,
                                                                           meta_map=meta_map,
                                                                           samples_folder=smpls_dir,
                                                                           params=params)

    # creates pixel_classes dict and keys
    pixel_classes = {key: 0 for key in gpkg_classes}
    background_val = 0
    pixel_classes[background_val] = 0
    class_prop = validate_class_prop_dict(pixel_classes, class_prop)
    pixel_classes[dontcare] = 0
    # endregion

    # For each row in csv:
    # (1) burn vector file to raster
    # (2) read input raster image
    # (3) prepare samples
    logging.info(f"Preparing samples \n\tSamples_size: {samples_size} \n\tOverlap: {overlap} "
                 f"\n\tValidation set: {val_percent} % of created training samples")

    for rowN, info in enumerate(tqdm(list_data_prep, position=0, leave=False)):
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

                logging.info(f"\nReading as array: {info['tif']}")
            with rasterio.open(info['tif'], 'r') as raster:

# 1. Read the input raster image
                np_input_image, raster, dataset_nodata, coords = image_reader_as_array(smpls_dir,
                                                                                      input_image=raster,
                                                                                      clip_gpkg=info['gpkg'],
                                                                                      aux_vector_file=get_key_def('aux_vector_file', params['global'], None), # TODO: what does aux mean?
                                                                                      aux_vector_attrib=get_key_def('aux_vector_attrib', params['global'], None),
                                                                                      aux_vector_ids=get_key_def('aux_vector_ids', params['global'], None),
                                                                                      aux_vector_dist_maps=get_key_def('aux_vector_dist_maps', params['global'], True),
                                                                                      aux_vector_dist_log=get_key_def('aux_vector_dist_log', params['global'], True),
                                                                                      aux_vector_scale=get_key_def('aux_vector_scale', params['global'], None))



# 2. Burn vector file in a raster file
                logging.info(f"\nRasterizing vector file (attribute: {info['attribute_name']}): {info['gpkg']}")
                np_label_raster = vector_to_raster(vector_file=info['gpkg'],
                                                   input_image=raster,
                                                   out_shape=np_input_image.shape[:2],
                                                   attribute_name=info['attribute_name'],
                                                   fill=background_val,
                                                   target_ids=targ_ids)  # background value in rasterized vector.

                if dataset_nodata is not None:
                    # 3. Set ignore_index value in label array where nodata in raster (only if nodata across all bands)
                    np_label_raster[dataset_nodata] = dontcare

            if debug:
                out_meta = raster.meta.copy()
                np_image_debug = np_input_image.transpose(2, 0, 1).astype(out_meta['dtype'])
                out_meta.update({"driver": "GTiff",
                                 "height": np_image_debug.shape[1],
                                 "width": np_image_debug.shape[2]})
                out_tif = smpls_dir / f"{Path(info['tif']).stem}_clipped.tif"
                logging.debug(f"Writing clipped raster to {out_tif}")
                with rasterio.open(out_tif, "w", **out_meta) as dest:
                    dest.write(np_image_debug)

                out_meta = raster.meta.copy()
                np_label_debug = np.expand_dims(np_label_raster, axis=2).transpose(2, 0, 1).astype(out_meta['dtype'])
                out_meta.update({"driver": "GTiff",
                                 "height": np_label_debug.shape[1],
                                 "width": np_label_debug.shape[2],
                                 'count': 1})
                out_tif = smpls_dir / f"{Path(info['gpkg']).stem}_clipped.tif"
                logging.debug(f"Writing final rasterized gpkg to {out_tif}")
                with rasterio.open(out_tif, "w", **out_meta) as dest:
                    dest.write(np_label_debug)

            # Mask the zeros from input image into label raster.
            if mask_reference:
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
            number_samples, number_classes = samples_preparation(coords, tracker_hdf5,
                                                             in_img_array=np_input_image,
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
                                                             dontcare=dontcare,
                                                             image_metadata=metadata,
                                                             min_annot_perc=min_annot_perc,
                                                             class_prop=class_prop,
                                                             stratd=stratd)

            logging.info(f'Row {rowN} / {len(list_data_prep)} Number of samples={number_samples}')
            out_file.flush()
        except OSError:
            logging.exception(f'An error occurred while preparing samples with row {rowN}, "{Path(info["tif"]).stem}" (tiff) and "{Path(info["gpkg"]).stem}" (gpkg).')
            continue

    trn_hdf5.close()
    val_hdf5.close()
    tst_hdf5.close()
    tracker_hdf5.close()
    # print("Elapsed time:{}".format(time.time() - start_time))
    pixel_total = 0
    # adds up the number of pixels for each class in pixel_classes dict
    for i in pixel_classes:
        pixel_total += pixel_classes[i]

    # prints the proportion of pixels of each class for the samples created
    for i in pixel_classes:
        prop = round((pixel_classes[i] / pixel_total) * 100, 1) if pixel_total > 0 else 0
        logging.info(f'Pixels from class {i}: {prop} %')

    logging.info("Number of samples created: ", number_samples)

    if bucket_name and final_samples_folder:  # FIXME: final_samples_folder always None in current implementation
        logging.info('Transfering Samples to the bucket')
        bucket.upload_file(smpls_dir + "/trn_samples.hdf5", final_samples_folder + '/trn_samples.hdf5')
        bucket.upload_file(smpls_dir + "/val_samples.hdf5", final_samples_folder + '/val_samples.hdf5')
        bucket.upload_file(smpls_dir + "/tst_samples.hdf5", final_samples_folder + '/tst_samples.hdf5')

    logging.info(f"End of process. Elapsed time:{(time.time() - start_time)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample preparation')
    parser.add_argument('ParamFile', metavar='DIR',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    params = read_parameters(args.ParamFile)
    start_time = time.time()
    print(f'\n\nStarting images to samples preparation with {args.ParamFile}\n\n')
    main(params)
    print("Elapsed time:{}".format(time.time() - start_time))

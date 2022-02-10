import shutil
from typing import Sequence

import rasterio
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, open_dict

# Our modules
from utils.logger import get_logger
from utils.geoutils import vector_to_raster
from utils.readers import image_reader_as_array
from utils.create_dataset import create_files_and_datasets, append_to_dataset
from utils.utils import (
    get_key_def, pad, pad_diff, read_csv, add_metadata_from_raster_to_sample, get_git_hash,
    read_modalities,
)
from utils.verifications import (
    validate_num_classes, assert_crs_match, validate_features_from_gpkg, validate_input_imagery
)
# Set the logging file
logging = get_logger(__name__)  # import logging

try:
    import boto3
except ModuleNotFoundError:
    logging.warning("\nThe boto3 library couldn't be imported. Ignore if not using AWS s3 buckets", ImportWarning)

# Set random seed for reproducibility
np.random.seed(1234)


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
        logging.warning(f"\nClass_proportion parameter should be a dictionary. Got type {type(config_dict)}")
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
            logging.warning(f"\nClass {key} not found in provided vector data.")

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
                    stratification_dict=None):
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
        logging.warning(
            "\nhigh overlap >25%, note that automatic train/val split creates very similar samples in both sets"
        )
    dist_samples = round(sample_size * (1 - (overlap / 100)))
    added_samples = 0
    excl_samples = 0

    # with tqdm(range(0, h, dist_samples), position=1, leave=True,
    #           desc=f'Writing samples. Dataset currently contains {idx_samples} '
    #                f'samples') as _tqdm:
    with tqdm(range(0, h, dist_samples), position=1, leave=True) as _tqdm:
        for row in _tqdm:
            for column in range(0, w, dist_samples):
                data = (in_img_array[row:row + sample_size, column:column + sample_size, :])
                target = np.squeeze(label_array[row:row + sample_size, column:column + sample_size, :], axis=2)
                data_row = data.shape[0]
                data_col = data.shape[1]
                if data_row < sample_size or data_col < sample_size:
                    padding = pad_diff(
                        data_row, data_col, sample_size, sample_size  # array, actual height, actual width, desired size
                    )
                    # don't fill with 0 if possible. Creates false min value when scaling.
                    data = pad(data, padding, fill=np.nan)

                target_row = target.shape[0]
                target_col = target.shape[1]
                if target_row < sample_size or target_col < sample_size:
                    padding = pad_diff(target_row, target_col, sample_size,
                                       sample_size)  # array, actual height, actual width, desired size
                    target = pad(target, padding, fill=dontcare)
                backgr_ct = np.sum(target == 0)
                backgr_ct += np.sum(target == dontcare)
                target_background_percent = round(backgr_ct / target.size * 100, 1)

                sample_metadata = {'sample_indices': (row, column)}

                # Stratification bias
                if (stratd is not None) and (dataset == 'trn'):
                    tile_size = target.size
                    u, count = np.unique(target, return_counts=True)
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
                    dist_trn = np.mean(np.array(list(distances_trn.values())) ** 2)
                    dist_val = np.mean(np.array(list(distances_val.values())) ** 2)
                    dist = dist_val - dist_trn
                    stratification_bias = stratd['strat_factor'] * np.sign(dist)
                else:
                    stratification_bias = 0.0

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
                                          dict_classes=pixel_classes,
                                          stratification_bias=stratification_bias,
                                          stratification_dict=stratd)
                    if val:
                        idx_samples_v += 1
                    else:
                        idx_samples += 1
                    added_samples += 1

                    # Stratification update
                    if (stratd is not None) and (dataset == 'trn'):
                        for key, val in tile_counts.items():
                            stratd[stratd['latest_assignment']]['total_counts'][key] += val
                        stratd[stratd['latest_assignment']]['total_pixels'] += tile_size

                else:
                    excl_samples += 1

                target_class_num = np.max(target)
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


def main(cfg: DictConfig) -> None:
    """
    Function that create training, validation and testing datasets preparation.

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
    :param cfg: (dict) Parameters found in the yaml config file.
    """
    # PARAMETERS
    num_classes = len(cfg.dataset.classes_dict.keys())
    num_bands = len(cfg.dataset.modalities)
    modalities = read_modalities(cfg.dataset.modalities)  # TODO add the Victor module to manage the modalities
    debug = cfg.debug

    # RAW DATA PARAMETERS
    data_path = get_key_def('raw_data_dir', cfg['dataset'], to_path=True, validate_path_exists=True)
    csv_file = get_key_def('raw_data_csv', cfg['dataset'], to_path=True, validate_path_exists=True)
    out_path = get_key_def('sample_data_dir', cfg['dataset'], default=data_path, to_path=True, validate_path_exists=True)

    # SAMPLE PARAMETERS
    samples_size = get_key_def('input_dim', cfg['dataset'], default=256, expected_type=int)
    overlap = get_key_def('overlap', cfg['dataset'], default=0)
    min_annot_perc = get_key_def('min_annot_perc', cfg['dataset'], default=0)
    val_percent = get_key_def('train_val_percent', cfg['dataset'], default=0.3)['val'] * 100
    samples_folder_name = f'samples{samples_size}_overlap{overlap}_min-annot{min_annot_perc}' \
                          f'_{num_bands}bands_{cfg.general.project_name}'
    samples_dir = out_path.joinpath(samples_folder_name)
    if samples_dir.is_dir():
        if debug:
            # Move existing data folder with a random suffix.
            last_mod_time_suffix = datetime.fromtimestamp(samples_dir.stat().st_mtime).strftime('%Y%m%d-%H%M%S')
            shutil.move(samples_dir, out_path.joinpath(f'{str(samples_dir)}_{last_mod_time_suffix}'))
        else:
            logging.critical(
                f'Data path exists: {samples_dir}. Remove it or use a different experiment_name.'
            )
            raise FileExistsError()
    Path.mkdir(samples_dir, exist_ok=False)  # TODO: what if we want to append samples to existing hdf5?

    # LOGGING PARAMETERS  TODO see logging yaml
    experiment_name = cfg.general.project_name
    # mlflow_uri = get_key_def('mlflow_uri', params['global'], default="./mlruns")

    # OTHER PARAMETERS
    # TODO class_prop get_key_def('class_proportion', params['sample']['sampling_method'], None, expected_type=dict)
    class_prop = None
    mask_reference = False  # TODO get_key_def('mask_reference', params['sample'], default=False, expected_type=bool)
    # set dontcare (aka ignore_index) value
    dontcare = cfg.dataset.ignore_index if cfg.dataset.ignore_index is not None else -1
    if dontcare == 0:
        raise ValueError("\nThe 'dontcare' value (or 'ignore_index') used in the loss function cannot be zero.")
    attribute_field = get_key_def('attribute_field', cfg['dataset'], None, expected_type=str)
    # Assert that all items in attribute_values are integers (ex.: single-class samples from multi-class label)
    attr_vals = get_key_def('attribute_values', cfg['dataset'], None, expected_type=Sequence)
    if attr_vals is list:
        for item in attr_vals:
            if not isinstance(item, int):
                raise logging.critical(ValueError(f'\nAttribute value "{item}" is {type(item)}, expected int.'))

    # OPTIONAL
    use_stratification = cfg.dataset.use_stratification if cfg.dataset.use_stratification is not None else False
    if use_stratification:
        stratd = {
            'trn': {'total_pixels': 0, 'total_counts': {}, 'total_props': {}},
            'val': {'total_pixels': 0, 'total_counts': {}, 'total_props': {}},
            'strat_factor': cfg['dataset']['use_stratification']
        }
    else:
        stratd = None

    # ADD GIT HASH FROM CURRENT COMMIT TO PARAMETERS (if available and parameters will be saved to hdf5s).
    with open_dict(cfg):
        cfg.general.git_hash = get_git_hash()

    # AWS TODO
    bucket_name = cfg.AWS.bucket_name
    if bucket_name:
        final_samples_folder = None
        bucket_name = cfg.AWS.bucket_name
        bucket_file_cache = []
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        bucket.download_file(csv_file, 'samples_prep.csv')
        list_data_prep = read_csv('samples_prep.csv')
    else:
        list_data_prep = read_csv(csv_file)

    # IF DEBUG IS ACTIVATE
    if debug:
        logging.warning(
            f'\nDebug mode activated. Some debug features may mobilize extra disk space and cause delays in execution.'
        )

    # VALIDATION: (1) Assert num_classes parameters == num actual classes in gpkg and (2) check CRS match (tif and gpkg)
    valid_gpkg_set = set()
    for info in tqdm(list_data_prep, position=0):
        validate_input_imagery(info['tif'], num_bands)
        if info['gpkg'] not in valid_gpkg_set:
            gpkg_classes = validate_num_classes(
                info['gpkg'], num_classes, attribute_field, dontcare, attribute_values=attr_vals,
            )
            assert_crs_match(info['tif'], info['gpkg'])
            valid_gpkg_set.add(info['gpkg'])

    if debug:
        # VALIDATION (debug only): Checking validity of features in vector files
        for info in tqdm(list_data_prep, position=0, desc=f"Checking validity of features in vector files"):
            # TODO: make unit to test this with invalid features.
            invalid_features = validate_features_from_gpkg(info['gpkg'], attribute_field)
            if invalid_features:
                logging.critical(f"{info['gpkg']}: Invalid geometry object(s) '{invalid_features}'")

    number_samples = {'trn': 0, 'val': 0, 'tst': 0}
    number_classes = 0

    trn_hdf5, val_hdf5, tst_hdf5 = create_files_and_datasets(samples_size=samples_size,
                                                             number_of_bands=num_bands,
                                                             samples_folder=samples_dir,
                                                             cfg=cfg)

    # creates pixel_classes dict and keys
    pixel_classes = {key: 0 for key in gpkg_classes}
    background_val = 0
    pixel_classes[background_val] = 0
    class_prop = validate_class_prop_dict(pixel_classes, class_prop)
    pixel_classes[dontcare] = 0

    # For each row in csv: (1) burn vector file to raster, (2) read input raster image, (3) prepare samples
    logging.info(
        f"\nPreparing samples \n  Samples_size: {samples_size} \n  Overlap: {overlap} "
        f"\n  Validation set: {val_percent} % of created training samples"
    )
    for info in tqdm(list_data_prep, position=0, leave=False):
        try:
            if bucket_name:
                bucket.download_file(info['tif'], "Images/" + info['tif'].split('/')[-1])
                info['tif'] = "Images/" + info['tif'].split('/')[-1]
                if info['gpkg'] not in bucket_file_cache:
                    bucket_file_cache.append(info['gpkg'])
                    bucket.download_file(info['gpkg'], info['gpkg'].split('/')[-1])
                info['gpkg'] = info['gpkg'].split('/')[-1]

            logging.info(f"\nReading as array: {info['tif']}")
            with rasterio.open(info['tif'], 'r') as raster:
                # 1. Read the input raster image
                np_input_image, raster, dataset_nodata = image_reader_as_array(
                    input_image=raster,
                    clip_gpkg=info['gpkg']
                )

                # 2. Burn vector file in a raster file
                logging.info(f"\nRasterizing vector file (attribute: {attribute_field}): {info['gpkg']}")
                try:
                    np_label_raster = vector_to_raster(vector_file=info['gpkg'],
                                                       input_image=raster,
                                                       out_shape=np_input_image.shape[:2],
                                                       attribute_name=attribute_field,
                                                       fill=background_val,
                                                       attribute_values=attr_vals)  # background value in rasterized vector.
                except ValueError:
                    logging.error(f"No vector features found for {info['gpkg']} with provided configuration."
                                  f"Will skip to next AOI.")
                    continue

                if dataset_nodata is not None:
                    # 3. Set ignore_index value in label array where nodata in raster (only if nodata across all bands)
                    np_label_raster[dataset_nodata] = dontcare

            if debug:
                out_meta = raster.meta.copy()
                np_image_debug = np_input_image.transpose(2, 0, 1).astype(out_meta['dtype'])
                out_meta.update({"driver": "GTiff",
                                 "height": np_image_debug.shape[1],
                                 "width": np_image_debug.shape[2]})
                out_tif = samples_dir / f"{Path(info['tif']).stem}_clipped.tif"
                logging.debug(f"Writing clipped raster to {out_tif}")
                with rasterio.open(out_tif, "w", **out_meta) as dest:
                    dest.write(np_image_debug)

                out_meta = raster.meta.copy()
                np_label_debug = np.expand_dims(np_label_raster, axis=2).transpose(2, 0, 1).astype(out_meta['dtype'])
                out_meta.update({"driver": "GTiff",
                                 "height": np_label_debug.shape[1],
                                 "width": np_label_debug.shape[2],
                                 'count': 1})
                out_tif = samples_dir / f"{Path(info['gpkg']).stem}_clipped.tif"
                logging.debug(f"\nWriting final rasterized gpkg to {out_tif}")
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
                raise logging.critical(
                    ValueError(f"\nDataset value must be trn or tst. Provided value is {info['dataset']}")
                )
            val_file = val_hdf5

            metadata = add_metadata_from_raster_to_sample(sat_img_arr=np_input_image,
                                                          raster_handle=raster,
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
                                                                 dontcare=dontcare,
                                                                 image_metadata=metadata,
                                                                 min_annot_perc=min_annot_perc,
                                                                 class_prop=class_prop,
                                                                 stratd=stratd)

            # logging.info(f'\nNumber of samples={number_samples}')
            out_file.flush()
        except OSError:
            logging.exception(f'\nAn error occurred while preparing samples with "{Path(info["tif"]).stem}" (tiff) and '
                              f'{Path(info["gpkg"]).stem} (gpkg).')
            continue

    trn_hdf5.close()
    val_hdf5.close()
    tst_hdf5.close()

    pixel_total = 0
    # adds up the number of pixels for each class in pixel_classes dict
    for i in pixel_classes:
        pixel_total += pixel_classes[i]
    # calculate the proportion of pixels of each class for the samples created
    pixel_classes_dict = {}
    for i in pixel_classes:
        # prop = round((pixel_classes[i] / pixel_total) * 100, 1) if pixel_total > 0 else 0
        pixel_classes_dict[i] = round((pixel_classes[i] / pixel_total) * 100, 1) if pixel_total > 0 else 0
    # prints the proportion of pixels of each class for the samples created
    msg_pixel_classes = "\n".join("Pixels from class {}: {}%".format(k, v) for k, v in pixel_classes_dict.items())
    logging.info("\n" + msg_pixel_classes)

    logging.info(f"\nNumber of samples created: {number_samples}")

    if bucket_name and final_samples_folder:  # FIXME: final_samples_folder always None in current implementation
        logging.info('\nTransfering Samples to the bucket')
        bucket.upload_file(samples_dir + "/trn_samples.hdf5", final_samples_folder + '/trn_samples.hdf5')
        bucket.upload_file(samples_dir + "/val_samples.hdf5", final_samples_folder + '/val_samples.hdf5')
        bucket.upload_file(samples_dir + "/tst_samples.hdf5", final_samples_folder + '/tst_samples.hdf5')

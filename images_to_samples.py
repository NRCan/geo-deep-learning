import argparse
import datetime
import logging
from typing import List

import numpy as np
np.random.seed(1234)  # Set random seed for reproducibility
import warnings
import rasterio
import time

from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict

from utils.create_dataset import create_files_and_datasets, append_to_dataset
from utils.utils import get_key_def, pad, pad_diff, read_csv, add_metadata_from_raster_to_sample, get_git_hash
from utils.geoutils import vector_to_raster
from utils.readers import read_parameters, image_reader_as_array
from utils.verifications import validate_num_classes, assert_num_bands, assert_crs_match, \
    validate_features_from_gpkg

try:
    import boto3
except ModuleNotFoundError:
    logging.warning("The boto3 library couldn't be imported. Ignore if not using AWS s3 buckets", ImportWarning)
    pass

logging.getLogger(__name__)

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
                    dict_classes):
    """ Add sample to Hdf5 (trn, val or tst) and computes pixel classes(%). """
    to_val_set = False
    if dataset == 'trn':
        random_val = np.random.randint(1, 100)
        if random_val > val_percent:
            pass
        else:
            to_val_set = True
            samples_file = val_sample_file
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
        if i not in dict_classes.keys():
            raise ValueError(f'Sample contains value "{class_val}" not defined in the classes ({dict_classes.keys()}).')

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

    if overlap > 25:
         logging.warning("high overlap >25%, note that automatic train/val split creates very similar samples in both sets")
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
                    padding = pad_diff(data_row, data_col, sample_size,
                                       sample_size)  # array, actual height, actual width, desired size
                    data = pad(data, padding, fill=np.nan)  # don't fill with 0 if possible. Creates false min value when scaling.

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

                target_class_num = np.max(target)
                if num_classes < target_class_num:
                    num_classes = target_class_num

                final_dataset = 'val' if val else dataset
                logging.debug(f'Dset={final_dataset}, '
                              f'Added samps={added_samples}/{len(_tqdm) * len(range(0, w, dist_samples))}, '
                              f'Excld samps={excl_samples}/{len(_tqdm) * len(range(0, w, dist_samples))}, ' 
                              f'Target annot perc={100 - target_background_percent:.1f}')

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

    params['global']['git_hash'] = get_git_hash()
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

    final_samples_folder = None

    sample_path_name = f'samples{samples_size}_overlap{overlap}_min-annot{min_annot_perc}_{num_bands}bands'

    import logging.config  # See: https://docs.python.org/2.4/lib/logging-config-fileformat.html
    log_config_path = Path('utils/logging.conf').absolute()
    logging.config.fileConfig(log_config_path, defaults={'logfilename': f'logs/{sample_path_name}.log',
                                                         'logfilename_debug': f'logs/{sample_path_name}_debug.log'})

    if debug:
        logging.warning(f'Debug mode activate. Execution may take longer...')

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
        logging.warning(f'Data path exists: {samples_folder}. Suffix will be added to directory name.')
        samples_folder = Path(str(samples_folder) + '_' + now)
    else:
        logging.info(f'Writing samples to {samples_folder}')
    Path.mkdir(samples_folder, exist_ok=False)  # TODO: what if we want to append samples to existing hdf5?
    logging.info(f'Samples will be written to {samples_folder}\n\n')

    logging.info(f'\n\tSuccessfully read csv file: {Path(csv_file).stem}\n'
                 f'\tNumber of rows: {len(list_data_prep)}\n'
                 f'\tCopying first entry:\n{list_data_prep[0]}\n')

    # Set dontcare (aka ignore_index) value
    dontcare = get_key_def("ignore_index", params["training"], -1)  # TODO: deduplicate with train_segmentation, l300
    if dontcare == 0:
        logging.warning("The 'dontcare' value (or 'ignore_index') used in the loss function cannot be zero;"
                        " all valid class indices should be consecutive, and start at 0. The 'dontcare' value"
                        " will be remapped to -1 while loading the dataset, and inside the config from now on.")
        dontcare = -1

    meta_map, metadata = get_key_def("meta_map", params["global"], {}), None

    num_classes = get_key_def('num_classes', params['global'], expected_type=int)
    targ_ids = get_key_def('target_ids', params['sample'], None, expected_type=List)
    # Assert that all items in target_ids are integers
    for item in targ_ids:
        assert isinstance(item, int), f'Target id "{item}" in target_ids is {type(item)}, expected int.'
    assert len(targ_ids) == num_classes, f'Yaml parameters mismatch. \n' \
                                         f'Got target_ids {targ_ids} (sample sect) with length {len(targ_ids)}. ' \
                                         f'Expected match with num_classes {num_classes} (global sect))'

    # VALIDATION: (1) Assert num_classes parameters == num actual classes in gpkg and (2) check CRS match (tif and gpkg)
    valid_gpkg_set = set()
    for info in tqdm(list_data_prep, position=0):
        assert_num_bands(info['tif'], num_bands, meta_map)
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
            invalid_features = validate_features_from_gpkg(info['gpkg'], info['attribute_name'])  # TODO: test this with invalid features.
            assert not invalid_features, f"{info['gpkg']}: Invalid geometry object(s) '{invalid_features}'"

    number_samples = {'trn': 0, 'val': 0, 'tst': 0}
    number_classes = 0

    class_prop = get_key_def('class_proportion', params['sample']['sampling_method'], None, expected_type=dict)

    trn_hdf5, val_hdf5, tst_hdf5 = create_files_and_datasets(params, samples_folder)

    # creates pixel_classes dict and keys
    pixel_classes = {key: 0 for key in gpkg_classes}
    background_val = 0
    pixel_classes[background_val] = 0
    class_prop = validate_class_prop_dict(pixel_classes, class_prop)
    pixel_classes[dontcare] = 0

    # For each row in csv: (1) burn vector file to raster, (2) read input raster image, (3) prepare samples
    logging.info(f"Preparing samples \n\tSamples_size: {samples_size} \n\tOverlap: {overlap} "
                 f"\n\tValidation set: {val_percent} % of created training samples")
    for info in tqdm(list_data_prep, position=0, leave=False):
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
                out_tif = samples_folder / f"{Path(info['tif']).stem}_clipped.tif"
                logging.debug(f"Writing clipped raster to {out_tif}")
                with rasterio.open(out_tif, "w", **out_meta) as dest:
                    dest.write(np_image_debug)

                out_meta = raster.meta.copy()
                np_label_debug = np.expand_dims(np_label_raster, axis=2).transpose(2, 0, 1).astype(out_meta['dtype'])
                out_meta.update({"driver": "GTiff",
                                 "height": np_label_debug.shape[1],
                                 "width": np_label_debug.shape[2],
                                 'count': 1})
                out_tif = samples_folder / f"{Path(info['gpkg']).stem}_clipped.tif"
                logging.debug(f"Writing final rasterized gpkg to {out_tif}")
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
                                                                 dontcare=dontcare,
                                                                 image_metadata=metadata,
                                                                 min_annot_perc=min_annot_perc,
                                                                 class_prop=class_prop)

            logging.info(f'Number of samples={number_samples}')
            out_file.flush()
        except OSError as e:
            logging.warning(f'An error occurred while preparing samples with "{Path(info["tif"]).stem}" (tiff) and '
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
        logging.info(f'Pixels from class {i}: {prop} %')

    logging.info("Number of samples created: ", number_samples)

    if bucket_name and final_samples_folder:
        logging.info('Transfering Samples to the bucket')
        bucket.upload_file(samples_folder + "/trn_samples.hdf5", final_samples_folder + '/trn_samples.hdf5')
        bucket.upload_file(samples_folder + "/val_samples.hdf5", final_samples_folder + '/val_samples.hdf5')
        bucket.upload_file(samples_folder + "/tst_samples.hdf5", final_samples_folder + '/tst_samples.hdf5')

    logging.info("End of process")


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

import argparse
import datetime
from math import floor, ceil
import os
import fiona
import numpy as np
import warnings
import rasterio
import time

from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict

from utils.CreateDataset import create_files_and_datasets, MetaSegmentationDataset
from utils.utils import get_key_def, pad, pad_diff, BGR_to_RGB
from utils.geoutils import vector_to_raster, validate_features_from_gpkg, lst_ids, clip_raster_with_gpkg
from utils.readers import read_parameters, image_reader_as_array, read_csv
from utils.verifications import is_valid_geom, validate_num_classes

# from rasterio.features import is_valid_geom #FIXME: https://github.com/mapbox/rasterio/issues/1815 is solved. Update rasterio package.

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
    old_size = dataset.shape[0]  # this function always appends samples on the first axis
    dataset.resize(old_size + 1, axis=0)
    dataset[old_size, ...] = sample
    return old_size  # the index to the newly added sample, or the previous size of the dataset


def check_sampling_dict(dictionary):
    for key, value in dictionary.items():
        try:
            int(str(key))
        except ValueError:
            f"Class should be a string castable as an integer. Got {key} of type {type(key)}"
        assert isinstance(value, int), f"Class value should be an integer, got {value} of type {type(value)}"


def minimum_annotated_percent(target_background_percent, min_annotated_percent):
    if not min_annotated_percent:
        return True
    elif float(target_background_percent) <= 100 - min_annotated_percent:
        return True

    return False


def class_proportion(target, sample_size: int, class_min_prop: dict):
    if not class_min_prop:
        return True
    prop_classes = {}
    sample_total = (sample_size) ** 2
    for i in range(0, params['global']['num_classes'] + 1):
        prop_classes.update({str(i): 0})
        if i in np.unique(target.clip(min=0).flatten()):
            prop_classes[str(i)] = (round((np.bincount(target.clip(min=0).flatten())[i] / sample_total) * 100, 1))

    for key, value in class_min_prop.items():
        if prop_classes[key] < value:
            return False

    return True


def compute_classes(dataset,
                    samples_file,
                    val_percent,
                    val_sample_file,
                    data,
                    target,
                    metadata_idx,
                    dict_classes,
                    dtype="float32"
                    ):
    # TODO: rename this function?
    """ Creates Dataset (trn, val, tst) appended to Hdf5 and computes pixel classes(%) """
    val = False
    if dataset == 'trn':
        random_val = np.random.randint(1, 100)
        if random_val > val_percent:
            pass
        else:
            val = True
            samples_file = val_sample_file
    append_to_dataset(samples_file["sat_img"], data)
    append_to_dataset(samples_file["sat_img_dtype"], dtype)
    append_to_dataset(samples_file["map_img"], target)
    append_to_dataset(samples_file["meta_idx"], metadata_idx)

    # adds pixel count to pixel_classes dict for each class in the image
    for i in (np.unique(target)):
        dict_classes[i] += (np.bincount(target.clip(min=0).flatten()))[i]

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
                        dtype=np.float32):
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
    :param image_metadata: (Ruamel) list of optionnal metadata specified in the associated metadata file
    :return: updated samples count and number of classes.
    """

    # read input and reference images as array

    h, w, num_bands = in_img_array.shape
    if dataset == 'trn':
        idx_samples = samples_count['trn']
    elif dataset == 'tst':
        idx_samples = samples_count['tst']
    else:
        raise ValueError(f"Dataset value must be trn or val. Provided value is {dataset}")

    metadata_idx = -1
    idx_samples_v = samples_count['val']
    if image_metadata:
        # there should be one set of metadata per raster
        # ...all samples created by tiling below will point to that metadata by index
        metadata_idx = append_to_dataset(samples_file["metadata"], repr(image_metadata))

    dist_samples = round(sample_size * (1 - (overlap / 100)))
    added_samples = 0
    excl_samples = 0

    dontcare = get_key_def("ignore_index", params["training"], -1)  # TODO: deduplicate with train_segmentation, l128
    if dontcare == 0:
        warnings.warn("The 'dontcare' value (or 'ignore_index') used in the loss function cannot be zero;"
                      " all valid class indices should be consecutive, and start at 0. The 'dontcare' value"
                      " will be remapped to -1 while loading the dataset, and inside the config from now on.")
        params["training"]["ignore_index"] = -1

    with tqdm(range(0, h, dist_samples), position=1, leave=True,
              desc=f'Writing samples to "{dataset}" dataset. Dataset currently contains {idx_samples} '
                   f'samples.') as _tqdm:

        for row in _tqdm:
            for column in range(0, w, dist_samples):
                data = (in_img_array[row:row + sample_size, column:column + sample_size, :])
                target = np.squeeze(label_array[row:row + sample_size, column:column + sample_size, :], axis=2)
                data_row = data.shape[0]
                data_col = data.shape[1]
                if data_row < sample_size or data_col < sample_size:
                    h_diff, w_diff = pad_diff(data_row, data_col, sample_size) # array, actual height, actual width, desired size
                    padding = ((floor(w_diff/2), floor(h_diff/2), ceil(w_diff/2), ceil(h_diff/2))) # left, top, right, bottom
                    data = pad(data, padding, fill=np.nan)  # don't fill with 0 if possible. Creates false min value when scaling.

                target_row = target.shape[0]
                target_col = target.shape[1]
                if target_row < sample_size or target_col < sample_size:
                    h_diff, w_diff = pad_diff(target_row, target_col,
                                              sample_size)  # array, actual height, actual width, desired size
                    padding = ((floor(w_diff/2), floor(h_diff/2), ceil(w_diff/2), ceil(h_diff/2)))
                    target = pad(target, padding, fill=dontcare)
                u, count = np.unique(target, return_counts=True)
                target_background_percent = round(count[0] / np.sum(count) * 100 if 0 in u else 0, 1)

                min_annot_perc = get_key_def('min_annotated_percent', params['sample']['sampling'], None, expected_type=int)
                class_prop = get_key_def('class_proportion', params['sample']['sampling'], None, expected_type=dict)

                if minimum_annotated_percent(target_background_percent, min_annot_perc) and \
                        class_proportion(target, sample_size, class_prop):
                    val = compute_classes(dataset=dataset,
                                          samples_file=samples_file,
                                          val_percent=val_percent,
                                          val_sample_file=val_sample_file,
                                          data=data,
                                          target=target,
                                          metadata_idx=metadata_idx,
                                          dict_classes=pixel_classes,
                                          dtype=dtype)
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

                _tqdm.set_postfix(Excld_samples=excl_samples,
                                  Added_samples=f'{added_samples}/{len(_tqdm) * len(range(0, w, dist_samples))}',
                                  Target_annot_perc=100 - target_background_percent)

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
    bucket_name = params['global']['bucket_name']
    data_path = Path(params['global']['data_path'])
    Path.mkdir(data_path, exist_ok=True, parents=True)
    csv_file = params['sample']['prep_csv_file']
    val_percent = params['sample']['val_percent']
    samples_size = params["global"]["samples_size"]
    overlap = params["sample"]["overlap"]
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
        samples_folder = f'samples{samples_size}_overlap{overlap}_{num_bands}bands'  # TODO: check if this is preferred name structure

    else:
        list_data_prep = read_csv(csv_file)
        samples_folder = data_path.joinpath(f'samples{samples_size}_overlap{overlap}_{num_bands}bands')

    if samples_folder.is_dir():
        warnings.warn(f'Data path exists: {samples_folder}. Suffix will be added to directory name.')
        samples_folder = Path(str(samples_folder) + '_' + now)
    else:
        tqdm.write(f'Writing samples to {samples_folder}')
    Path.mkdir(samples_folder, exist_ok=False)  # TODO: what if we want to append samples to existing hdf5?
    tqdm.write(f'Samples will be written to {samples_folder}\n\n')

    tqdm.write(f'\nSuccessfully read csv file: {Path(csv_file).stem}\nNumber of rows: {len(list_data_prep)}\nCopying first entry:\n{list_data_prep[0]}\n')
    ignore_index = get_key_def('ignore_index', params['training'], -1)

    for info in tqdm(list_data_prep, position=0, desc=f'Asserting existence of tif and gpkg files in csv'):
        assert Path(info['tif']).is_file(), f'Could not locate "{info["tif"]}". Make sure file exists in this directory.'
        assert Path(info['gpkg']).is_file(), f'Could not locate "{info["gpkg"]}". Make sure file exists in this directory.'
    if debug:
        # Assert num_classes parameters == number of actual classes in gpkg
        for info in tqdm(list_data_prep, position=0, desc=f"Validating presence of {params['global']['num_classes']} "
                                                          f"classes in attribute \"{info['attribute_name']}\" for vector "
                                                          f"file \"{Path(info['gpkg']).stem}\""):
            gpkg_classes = validate_num_classes(info['gpkg'], params['global']['num_classes'], info['attribute_name'], ignore_index)

            meta_map, metadata = get_key_def("meta_map", params["global"], {}), None
            # FIXME: think this through. User will have to calculate the total number of bands including meta layers and
            #  specify it in yaml. Is this the best approach? What if metalayers are added on the fly ?
            with rasterio.open(info['tif'], 'r') as raster:
                input_band_count = raster.meta['count'] + MetaSegmentationDataset.get_meta_layer_count(meta_map)
            assert input_band_count == num_bands, \
                f"The number of bands in the input image ({input_band_count}) and the parameter" \
                f"'number_of_bands' in the yaml file ({params['global']['number_of_bands']}) should be identical"

        with tqdm(list_data_prep, position=0, desc=f"Checking validity of features in vector files") as _tqdm:
            for info in _tqdm:
                invalid_features = validate_features_from_gpkg(info['gpkg'], info['attribute_name'])  # TODO: test this.
                assert not invalid_features, f"{info['gpkg']}: Invalid geometry object(s) '{invalid_features}'"

    number_samples = {'trn': 0, 'val': 0, 'tst': 0}
    number_classes = 0

    # 'sampling' ordereddict validation
    # check_sampling_dict() # TODO replace with get_key_def(). Add type check to get_key_def.

    # creates pixel_classes dict and keys
    pixel_classes = {key: 0 for key in gpkg_classes}

    trn_hdf5, val_hdf5, tst_hdf5 = create_files_and_datasets(params, samples_folder)

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
                    dtype = raster.meta["dtype"]
                    np_input_image = image_reader_as_array(input_image=raster,
                                                           clip_gpkg=info['gpkg'],
                                                           aux_vector_file=get_key_def('aux_vector_file',
                                                                                       params['global'], None),
                                                           aux_vector_attrib=get_key_def('aux_vector_attrib',
                                                                                         params['global'], None),
                                                           aux_vector_ids=get_key_def('aux_vector_ids',
                                                                                      params['global'], None),
                                                           aux_vector_dist_maps=get_key_def('aux_vector_dist_maps',
                                                                                            params['global'], True),
                                                           aux_vector_dist_log=get_key_def('aux_vector_dist_log',
                                                                                           params['global'], True),
                                                           aux_vector_scale=get_key_def('aux_vector_scale',
                                                                                        params['global'], None))

                    bgr_to_rgb = get_key_def('BGR_to_RGB', params['global'], True)  # TODO: add to config
                    np_input_image = BGR_to_RGB(np_input_image) if bgr_to_rgb else np_input_image

                    # 2. Burn vector file in a raster file
                    np_label_raster = vector_to_raster(vector_file=info['gpkg'],
                                                       input_image=raster,
                                                       out_shape=np_input_image.shape[:2],
                                                       attribute_name=info['attribute_name'],
                                                       fill=0)  # This will become background value in raster.

                # Mask the zeros from input image into label raster.
                if params['sample']['mask_reference']:
                    np_label_raster = mask_image(np_input_image, np_label_raster)

                if info['dataset'] == 'trn':
                    out_file = trn_hdf5
                    val_file = val_hdf5
                elif info['dataset'] == 'tst':
                    out_file = tst_hdf5
                else:
                    raise ValueError(f"Dataset value must be trn or val or tst. Provided value is {info['dataset']}")

                if info['meta'] is not None and isinstance(info['meta'], str) and Path(info['meta']).is_file():
                    metadata = read_parameters(info['meta'])

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
                                                                     dtype=dtype)

                _tqdm.set_postfix(OrderedDict(number_samples=number_samples))
                out_file.flush()
            except IOError as e:
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

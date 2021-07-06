import argparse
import datetime
import os
import numpy as np

np.random.seed(1234)  # Set random seed for reproducibility
import warnings
import rasterio
import time
import json

from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict, Counter
from typing import List

from utils.create_dataset import create_files_and_datasets
from utils.utils import get_key_def, pad, pad_diff, read_csv, add_metadata_from_raster_to_sample
from utils.geoutils import vector_to_raster
# clip_raster_with_gpkg
from utils.readers import read_parameters, image_reader_as_array
from utils.verifications import validate_num_classes, assert_num_bands, assert_crs_match, \
    validate_features_from_gpkg
from rasterio.features import is_valid_geom
from rasterio.mask import mask
from rasterio.windows import Window
from rasterio.plot import reshape_as_image


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


def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


def clip_raster_with_gpkg(raster, gpkg, debug=False):
    """Clips input raster to limits of vector data in gpkg. Adapted from: https://automating-gis-processes.github.io/CSC18/lessons/L6/clipping-raster.html
    raster: Rasterio file handle holding the (already opened) input raster
    gpkg: Path and name of reference GeoPackage
    debug: if True, output raster as given by this function is saved to disk
    """
    from shapely.geometry import box  # geopandas and shapely become a project dependency only during sample creation
    import geopandas as gpd
    import fiona
    # Get extent of gpkg data with fiona
    with fiona.open(gpkg, 'r') as src:
        gpkg_crs = src.crs
        assert gpkg_crs == raster.crs
        minx, miny, maxx, maxy = src.bounds  # ouest, nord, est, sud

    # Create a bounding box with Shapely
    bbox = box(minx, miny, maxx, maxy)

    # Insert the bbox into a GeoDataFrame
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0])  # , crs=gpkg_crs['init'])

    # Re-project into the same coordinate system as the raster data
    # geo = geo.to_crs(crs=raster.crs.data)

    # Get the geometry coordinates by using the function.
    coords = getFeatures(geo)

    # clip the raster with the polygon
    out_tif = Path(raster.name).parent / f"{Path(raster.name).stem}_clipped{Path(raster.name).suffix}"
    if os.path.isfile(out_tif):
        return out_tif
    else:
        try:
            out_img, out_transform = mask(dataset=raster, shapes=coords, crop=True)
            out_meta = raster.meta.copy()
            out_meta.update({"driver": "GTiff",
                             "height": out_img.shape[1],
                             "width": out_img.shape[2],
                             "transform": out_transform})
            with rasterio.open(out_tif, "w", **out_meta) as dest:
                print(f"writing clipped raster to {out_tif}")
                dest.write(out_img)
            return out_tif
        except ValueError as e:  # if gpkg's extent outside raster: "ValueError: Input shapes do not overlap raster."
            # TODO: warning or exception? if warning, except must be set in images_to_samples
            warnings.warn(f"e\n {raster.name}\n{gpkg}")


def process_raster_img(rst_pth, gpkg_pth):
    with rasterio.open(rst_pth) as src:
        rst_pth = clip_raster_with_gpkg(src, gpkg_pth)
    return rst_pth, src


def reorder_bands(a: List[str], b: List[str]):
    read_band_order = []
    for band in a:
        if band in b:
            read_band_order.insert(a.index(band) + 1, b.index(band) + 1)
            # print(f'{a.index(band)},{band}, {b.index(band)}')
    return read_band_order


def gen_img_samples(rst_pth, tile_size, dist_samples, *band_order):
    with rasterio.open(rst_pth) as src:
        for row in range(0, src.height, dist_samples):
            for column in range(0, src.width, dist_samples):
                window = Window.from_slices(slice(row, row + tile_size),
                                            slice(column, column + tile_size))
                if band_order:
                    window_array = reshape_as_image(src.read(band_order[0], window=window))
                else:
                    window_array = reshape_as_image(src.read(window=window))

                if window_array.shape[0] < tile_size or window_array.shape[1] < tile_size:
                    padding = pad_diff(window_array.shape[0], window_array.shape[1], tile_size, tile_size)
                    window_array = pad(window_array, padding, fill=np.nan)

                yield window_array


def process_vector_label(rst_pth, gpkg_pth):
    if rst_pth is not None:
        with rasterio.open(rst_pth) as src:
            np_label = vector_to_raster(vector_file=gpkg_pth,
                                        input_image=src,
                                        out_shape=(src.height, src.width),
                                        attribute_name='properties/Quatreclasses',
                                        fill=0,
                                        target_ids=[1, '1', 2, '2', 3, '3', 4, '4'],
                                        merge_all=True,
                                        )
        return np_label


def gen_label_samples(np_label, dist_samples, tile_size):
    h, w = np_label.shape
    for row in range(0, h, dist_samples):
        for column in range(0, w, dist_samples):
            target = np_label[row:row + tile_size, column:column + tile_size]
            target_row = target.shape[0]
            target_col = target.shape[1]
            if target_row < tile_size or target_col < tile_size:
                padding = pad_diff(target_row, target_col, tile_size,
                                   tile_size)  # array, actual height, actual width, desired size
                target = pad(target, padding, fill=-1)
            indices = (row, column)
            yield target, indices


def minimum_annotated_percent(target_background_percent, min_annotated_percent):
    if not min_annotated_percent:
        return True
    elif float(target_background_percent) <= 100 - min_annotated_percent:
        return True

    return False


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


def sample_prep(src, data, target, indices, gpkg_classes, sample_size, sample_type, samples_count, samples_file,
                num_classes,
                val_percent,
                val_sample_file,
                min_annot_perc=None,
                class_prop=None,
                dontcare=-1
                ):
    added_samples = 0
    excl_samples = 0

    # print('gpkg_classes', gpkg_classes)
    pixel_classes = {key: 0 for key in gpkg_classes}
    background_val = 0
    pixel_classes[background_val] = 0
    class_prop = validate_class_prop_dict(pixel_classes, class_prop)
    pixel_classes[dontcare] = 0

    image_metadata = add_metadata_from_raster_to_sample(sat_img_arr=data,
                                                        raster_handle=src,
                                                        meta_map={},
                                                        raster_info={})
    # Save label's per class pixel count to image metadata
    image_metadata['source_label_bincount'] = {class_num: count for class_num, count in
                                               enumerate(np.bincount(target.clip(min=0).flatten()))
                                               if count > 0}  # TODO: add this to add_metadata_from[...] function?

    if sample_type == 'trn':
        idx_samples = samples_count['trn']
        append_to_dataset(val_sample_file["metadata"], repr(image_metadata))
    elif sample_type == 'tst':
        idx_samples = samples_count['tst']
    else:
        raise ValueError(f"Sample type must be trn or tst. Provided type is {sample_type}")

    idx_samples_v = samples_count['val']
    # Adds raster metadata to the dataset. All samples created by tiling below will point to that metadata by index
    metadata_idx = append_to_dataset(samples_file["metadata"], repr(image_metadata))
    u, count = np.unique(target, return_counts=True)
    # print('class:', u, 'count:', count)
    target_background_percent = round(count[0] / np.sum(count) * 100 if 0 in u else 0, 1)
    sample_metadata = {'sample_indices': indices}
    val = False
    if minimum_annotated_percent(target_background_percent, min_annot_perc) and \
            class_proportion(target, sample_size, class_prop):
        val = add_to_datasets(dataset=sample_type,
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

    sample_type_ = 'val' if val else sample_type
    # assert added_samples > 0, "No sample added for current raster. Problems may occur with use of metadata"

    if sample_type == 'tst':
        samples_count['tst'] = idx_samples
    else:
        samples_count['trn'] = idx_samples
        samples_count['val'] = idx_samples_v

    # return the appended samples count and number of classes.
    # print('pixel_classes', pixel_classes)
    # print(samples_count, num_classes)

    return samples_count, num_classes, pixel_classes


def class_pixel_ratio(pixel_classes: dict, source_data: str, file_path: str):
    with open(file_path, 'a+') as f:
        pixel_total = sum(pixel_classes.values())
        print(f'\n****{source_data}****\n', file=f)
        for i in pixel_classes:
            prop = round((pixel_classes[i] / pixel_total) * 100, 1) if pixel_total > 0 else 0
            print(f'{source_data}_class', i, ':', prop, '%', file=f)
        print(f'\n****{source_data}****\n', file=f)


def main(params):
    """
    Training and validation datasets preparation.
    :param params: (dict) Parameters found in the yaml config file.

    """
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    bucket_file_cache = []

    assert params['global']['task'] == 'segmentation', \
        f"images_to_samples.py isn't necessary when performing classification tasks"

    # SET BASIC VARIABLES AND PATHS. CREATE OUTPUT FOLDERS.
    data_path = Path(params['global']['data_path'])
    Path.mkdir(data_path, exist_ok=True, parents=True)
    val_percent = params['sample']['val_percent']
    samples_size = params["global"]["samples_size"]
    overlap = params["sample"]["overlap"]
    dist_samples = round(samples_size * (1 - (overlap / 100)))
    min_annot_perc = get_key_def('min_annotated_percent', params['sample']['sampling_method'], None, expected_type=int)
    ignore_index = get_key_def('ignore_index', params['training'], -1)

    list_params = params['read_img']
    source_pan = get_key_def('pan', list_params['source'], default=False, expected_type=bool)
    source_mul = get_key_def('mul', list_params['source'], default=False, expected_type=bool)
    mul_band_order = get_key_def('mulband', list_params['source'], default=[], expected_type=list)
    prep_band = get_key_def('band', list_params['prep'], default=[], expected_type=list)
    tst_set = get_key_def('benchmark', list_params, default=[], expected_type=list)
    in_pth = get_key_def('input_file', list_params, default='data_file.json', expected_type=str)
    gpkg_status = 'all'

    sample_path_name = f'samples{samples_size}_overlap{overlap}_min-annot{min_annot_perc}'
    samples_folder = data_path.joinpath(sample_path_name)
    if samples_folder.is_dir():
        warnings.warn(f'Data path exists: {samples_folder}. Suffix will be added to directory name.')
        samples_folder = Path(str(samples_folder) + '_' + now)
    else:
        tqdm.write(f'Writing samples to {samples_folder}')
    Path.mkdir(samples_folder, exist_ok=False)  # TODO: what if we want to append samples to existing hdf5?
    trn_hdf5, val_hdf5, tst_hdf5 = create_files_and_datasets(params, samples_folder)

    class_prop = get_key_def('class_proportion', params['sample']['sampling_method'], None, expected_type=dict)
    dontcare = get_key_def("ignore_index", params["training"], -1)
    number_samples = {'trn': 0, 'val': 0, 'tst': 0}
    number_classes = 0

    pixel_pan_counter = Counter()
    pixel_mul_counter = Counter()
    pixel_prep_counter = Counter()
    filename = samples_folder.joinpath('class_distribution.txt')

    with open(Path(in_pth), 'r') as fin:
        dict_images = json.load(fin)

        for i_dict in tqdm(dict_images['all_images'], desc=f'Writing samples to {samples_folder}'):

            if source_pan:
                if not len(i_dict['pan_img']) == 0 and i_dict['gpkg']:
                    if gpkg_status == 'all':
                        if 'corr' or 'prem' in i_dict['gpkg'].keys():
                            gpkg = list(i_dict['gpkg'].values())[0]
                            gpkg_classes = validate_num_classes(gpkg, params['global']['num_classes'],
                                                                'properties/Quatreclasses',
                                                                ignore_index)
                            for img_pan in i_dict['pan_img']:
                                assert_crs_match(img_pan, gpkg)
                                rst_pth, r_ = process_raster_img(img_pan, gpkg)
                                np_label = process_vector_label(rst_pth, gpkg)
                                if np_label is not None:
                                    if Path(gpkg).stem in tst_set:
                                        sample_type = 'tst'
                                        out_file = tst_hdf5
                                    else:
                                        sample_type = 'trn'
                                        out_file = trn_hdf5
                                    val_file = val_hdf5
                                    src = r_
                                    pan_label_gen = gen_label_samples(np_label, dist_samples, samples_size)
                                    pan_img_gen = gen_img_samples(rst_pth, samples_size, dist_samples)
                                else:
                                    continue
                for pan_img, pan_label in zip(pan_img_gen, pan_label_gen):
                    number_samples, number_classes, class_pixels_pan = sample_prep(src, pan_img, pan_label[0],
                                                                                   pan_label[1], gpkg_classes,
                                                                                   samples_size, sample_type,
                                                                                   number_samples, out_file,
                                                                                   number_classes,
                                                                                   val_percent, val_file,
                                                                                   min_annot_perc,
                                                                                   class_prop=class_prop,
                                                                                   dontcare=dontcare)
                    pixel_pan_counter.update(class_pixels_pan)

            if source_mul:
                if not len(i_dict['mul_img']) == 0 and i_dict['gpkg']:
                    band_order = reorder_bands(i_dict['mul_band'], mul_band_order)
                    if gpkg_status == 'all':
                        if 'corr' or 'prem' in i_dict['gpkg'].keys():
                            gpkg = list(i_dict['gpkg'].values())[0]
                            gpkg_classes = validate_num_classes(gpkg, params['global']['num_classes'],
                                                                'properties/Quatreclasses',
                                                                ignore_index)
                            for img_mul in i_dict['mul_img']:
                                assert_crs_match(img_mul, gpkg)
                                rst_pth, r_ = process_raster_img(img_mul, gpkg)
                                np_label = process_vector_label(rst_pth, gpkg)
                                if np_label is not None:
                                    if Path(gpkg).stem in tst_set:
                                        sample_type = 'tst'
                                        out_file = tst_hdf5
                                    else:
                                        sample_type = 'trn'
                                        out_file = trn_hdf5
                                    val_file = val_hdf5
                                    src = r_

                                    mul_label_gen = gen_label_samples(np_label, dist_samples, samples_size)
                                    mul_img_gen = gen_img_samples(rst_pth, samples_size, dist_samples, band_order)
                                else:
                                    continue
                for mul_img, mul_label in zip(mul_img_gen, mul_label_gen):
                    number_samples, number_classes, class_pixels_mul = sample_prep(src, mul_img, mul_label[0],
                                                                                   mul_label[1], gpkg_classes,
                                                                                   samples_size, sample_type,
                                                                                   number_samples, out_file,
                                                                                   number_classes,
                                                                                   val_percent, val_file,
                                                                                   min_annot_perc,
                                                                                   class_prop=class_prop,
                                                                                   dontcare=dontcare)
                    pixel_mul_counter.update(class_pixels_mul)

            if prep_band:
                bands_gen_list = []
                if set(prep_band).issubset({'R', 'G', 'B', 'N'}):
                    for ib in prep_band:
                        if i_dict[f'{ib}_band'] and i_dict['gpkg']:
                            if gpkg_status == 'all':
                                if 'corr' or 'prem' in i_dict['gpkg'].keys():
                                    gpkg = list(i_dict['gpkg'].values())[0]
                                    gpkg_classes = validate_num_classes(gpkg, params['global']['num_classes'],
                                                                        'properties/Quatreclasses',
                                                                        ignore_index)
                                    assert_crs_match(i_dict[f'{ib}_band'], gpkg)
                                    rst_pth, r_ = process_raster_img(i_dict[f'{ib}_band'], gpkg)
                                    np_label = process_vector_label(rst_pth, gpkg)
                                    prep_img_gen = gen_img_samples(rst_pth, samples_size, dist_samples)
                                    bands_gen_list.append(prep_img_gen)

                if np_label is not None:
                    if Path(gpkg).stem in tst_set:
                        sample_type = 'tst'
                        out_file = tst_hdf5
                    else:
                        sample_type = 'trn'
                        out_file = trn_hdf5
                    val_file = val_hdf5
                    src = r_
                    prep_label_gen = gen_label_samples(np_label, dist_samples, samples_size)
                    if len(prep_band) and len(bands_gen_list) == 1:
                        for b1, prep_label in zip(bands_gen_list[0], prep_label_gen):
                            prep_img = b1
                            number_samples, number_classes, class_pixels_prep = sample_prep(src, prep_img,
                                                                                            prep_label[0],
                                                                                            prep_label[1], gpkg_classes,
                                                                                            samples_size, sample_type,
                                                                                            number_samples, out_file,
                                                                                            number_classes,
                                                                                            val_percent, val_file,
                                                                                            min_annot_perc,
                                                                                            class_prop=class_prop,
                                                                                            dontcare=dontcare)
                            pixel_prep_counter.update(class_pixels_prep)

                    elif len(prep_band) and len(bands_gen_list) == 2:
                        for b1, b2, prep_label in zip(*bands_gen_list, prep_label_gen):
                            prep_img = np.dstack(np.array([b1, b2]))
                            number_samples, number_classes, class_pixels_prep = sample_prep(src, prep_img,
                                                                                            prep_label[0],
                                                                                            prep_label[1], gpkg_classes,
                                                                                            samples_size, sample_type,
                                                                                            number_samples, out_file,
                                                                                            number_classes,
                                                                                            val_percent, val_file,
                                                                                            min_annot_perc,
                                                                                            class_prop=class_prop,
                                                                                            dontcare=dontcare)
                            pixel_prep_counter.update(class_pixels_prep)

                    elif len(prep_band) and len(bands_gen_list) == 3:
                        for b1, b2, b3, prep_label in zip(*bands_gen_list, prep_label_gen):
                            prep_img = np.dstack(np.array([b1, b2, b3]))
                            number_samples, number_classes, class_pixels_prep = sample_prep(src, prep_img,
                                                                                            prep_label[0],
                                                                                            prep_label[1], gpkg_classes,
                                                                                            samples_size, sample_type,
                                                                                            number_samples, out_file,
                                                                                            number_classes,
                                                                                            val_percent, val_file,
                                                                                            min_annot_perc,
                                                                                            class_prop=class_prop,
                                                                                            dontcare=dontcare)
                            pixel_prep_counter.update(class_pixels_prep)

                    elif len(prep_band) and len(bands_gen_list) == 4:
                        for b1, b2, b3, b4, prep_label in zip(*bands_gen_list, prep_label_gen):
                            prep_img = np.dstack(np.array([b1, b2, b3, b4]))
                            number_samples, number_classes, class_pixels_prep = sample_prep(src, prep_img,
                                                                                            prep_label[0],
                                                                                            prep_label[1], gpkg_classes,
                                                                                            samples_size, sample_type,
                                                                                            number_samples, out_file,
                                                                                            number_classes,
                                                                                            val_percent, val_file,
                                                                                            min_annot_perc,
                                                                                            class_prop=class_prop,
                                                                                            dontcare=dontcare)
                            pixel_prep_counter.update(class_pixels_prep)
                    else:
                        continue
    trn_hdf5.close()
    val_hdf5.close()
    tst_hdf5.close()

    class_pixel_ratio(pixel_pan_counter, 'pan_source', filename)
    class_pixel_ratio(pixel_mul_counter, 'mul_source', filename)
    class_pixel_ratio(pixel_prep_counter, 'prep_source', filename)
    print("Number of samples created: ", number_samples, number_classes)


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

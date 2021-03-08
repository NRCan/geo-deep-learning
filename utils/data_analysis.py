import argparse
import os
from pathlib import Path
from typing import List

from utils.utils import get_key_def, read_csv
from utils.geoutils import vector_to_raster
from utils.readers import read_parameters, image_reader_as_array
from utils.verifications import validate_num_classes
import time
import rasterio
import csv
import numpy as np
from collections import OrderedDict
from numpy import genfromtxt
from tqdm import tqdm

import images_to_samples


def create_csv():
    """
    Creates samples from the input images for the pixel_inventory function

    """
    prep_csv_path = params['sample']['prep_csv_file']
    dist_samples = params['sample']['samples_dist']
    sample_size = params['global']['samples_size']
    data_path = params['global']['data_path']
    Path.mkdir(Path(data_path), exist_ok=True)
    num_classes = params['global']['num_classes']
    data_prep_csv = read_csv(prep_csv_path)

    csv_prop_data = params['global']['data_path'] + '/prop_data.csv'
    if os.path.isfile(csv_prop_data):
        os.remove(csv_prop_data)

    with tqdm(data_prep_csv) as _tqdm:
        for info in _tqdm:

            _tqdm.set_postfix(OrderedDict(file=f'{info["tif"]}', sample_size=params['global']['samples_size']))

            # Validate the number of class in the vector file
            validate_num_classes(info['gpkg'], num_classes, info['attribute_name'])

            assert os.path.isfile(info['tif']), f"could not open raster file at {info['tif']}"
            with rasterio.open(info['tif'], 'r') as raster:

                # Burn vector file in a raster file
                np_label_raster = vector_to_raster(vector_file=info['gpkg'],
                                                   input_image=raster,
                                                   attribute_name=info['attribute_name'],
                                                   fill=get_key_def('ignore_idx', get_key_def('training', params, {}),
                                                                    0))

                # Read the input raster image
                np_input_image = image_reader_as_array(input_image=raster,
                                                       aux_vector_file=get_key_def('aux_vector_file', params['global'],
                                                                                   None),
                                                       aux_vector_attrib=get_key_def('aux_vector_attrib',
                                                                                     params['global'], None),
                                                       aux_vector_ids=get_key_def('aux_vector_ids', params['global'],
                                                                                  None),
                                                       aux_vector_dist_maps=get_key_def('aux_vector_dist_maps',
                                                                                        params['global'], True),
                                                       aux_vector_dist_log=get_key_def('aux_vector_dist_log',
                                                                                       params['global'], True),
                                                       aux_vector_scale=get_key_def('aux_vector_scale',
                                                                                    params['global'], None))
                # Mask the zeros from input image into label raster.
                if params['sample']['mask_reference']:
                    np_label_raster = images_to_samples.mask_image(np_input_image, np_label_raster)

                np_label_raster = np.reshape(np_label_raster, (np_label_raster.shape[0], np_label_raster.shape[1], 1))

                h, w, num_bands = np_input_image.shape

                # half tile padding
                half_tile = int(sample_size / 2)
                pad_label_array = np.pad(np_label_raster, ((half_tile, half_tile), (half_tile, half_tile), (0, 0)),
                                         mode='constant')

                for row in range(0, h, dist_samples):
                    for column in range(0, w, dist_samples):
                        target = np.squeeze(pad_label_array[row:row + sample_size, column:column + sample_size, :], axis=2)

                        pixel_inventory(target, sample_size, params['global']['num_classes'] + 1,
                                        params['global']['data_path'], info['dataset'])


def pixel_inventory(sample, sample_size, num_classes, data_path, dataset):
    """
    Calculates the proportions of the classes contained in a sample
    :param sample: numpy array of the sample
    :param sample_size: (int) Size (in pixel) of the samples
    :param num_classes: (dict) Number of classes in reference data
    :param data_path: (str) path to the samples folder
    :param dataset: (str) Type of dataset of the sample. Can be 'trn' or 'val' or 'tst'

    """
    sample_total = sample_size ** 2
    samples_prop = []
    for i in range(0, num_classes):
        samples_prop.append(0.0)
        if i in np.unique(sample.flatten()):
            samples_prop[i] += (round((np.bincount(sample.flatten())[i] / sample_total) * 100, 1))
    samples_prop.append(dataset)

    f = open(data_path + '/prop_data.csv', 'a')

    with f:
        writer = csv.writer(f)
        writer.writerow(samples_prop)


def minimum_annotated_percent(target_background_percent, min_annotated_percent):
    """
    :param target_background_percent: background pixels in sample
    :param min_annotated_percent: (int) Minimum % of non background pixels in sample, in order to consider it part of
    the dataset
    :return: (Bool)
    """
    if float(target_background_percent) <= 100 - min_annotated_percent:
        return True

    return False


def minimum_annotated_percent_search(classes, annotated_p, sampling, sample_data):
    keys = ['map', 'std', 'trn_data']
    for i in classes:
        keys.append('prop' + str(i))
    stats_dict = {key: [] for key in keys}

    for i in tqdm(annotated_p):
        prop_classes = {}
        for j, (key, value) in enumerate(sampling.items()):
            if j >= 2:
                prop_classes.update({key: 0})

        numbers_sample = {'trn': 0, 'val': 0, 'tst': 0}

        for row in sample_data:
            if minimum_annotated_percent(row[0], i):
                compute_classes(classes, prop_classes, row, numbers_sample)

        if parameters_search_dict(stats_dict, prop_classes, numbers_sample, i) is False:
            break

    if len(sampling['method']) == 1:
        results(classes, stats_dict)
    else:
        return stats_dict


def class_proportion(row, classes, source):
    condition = []
    if params['data_analysis']['optimal_parameters_search']:
        for i in classes:
            if float(row[i]) >= source[i]:
                condition.append(1)
        if len(condition) == len(classes):
            return True
        else:
            return False

    else:
        for i in classes:
            if float(row[i]) >= source[str(i)]:
                condition.append(1)
        if len(condition) == len(classes):
            return True
        else:
            return False


def class_proportion_search(classes, sampling, sample_data):
    keys = ['combination', 'std', 'trn_data']
    for i in classes:
        keys.append('prop' + str(i))
    stats_dict = {key: [] for key in keys}

    prop_classes = {}
    threshold = [float(0.0) for i in classes]

    for i in classes:
        if i != 0:
            while threshold[i] < 100.0:

                for j, (key, value) in enumerate(sampling.items()):
                    if j >= 2:
                        prop_classes.update({key: 0})

                numbers_sample = {'trn': 0, 'val': 0, 'tst': 0}

                for row in tqdm(sample_data):
                    if class_proportion(row, classes, threshold):
                        compute_classes(classes, prop_classes, row, numbers_sample)

                if parameters_search_dict(stats_dict, prop_classes, numbers_sample, threshold) is False:
                    break
                else:
                    val = threshold[i]
                    threshold[i] = round(val + 0.1, 1)

    if len(sampling['method']) == 1:
        results(classes, stats_dict)
    else:
        return stats_dict


def compute_classes(classes, dict_classes, row, numbers_dict):
    for i in classes:
        dict_classes[str(i)] += int((float(row[i]) / 100) * params['global']['samples_size'] ** 2)
    if row[-1] == 'trn':
        numbers_dict['trn'] += 1
    elif row[-1] == 'val':
        numbers_dict['val'] += 1
    elif row[-1] == 'tst':
        numbers_dict['tst'] += 1

    return numbers_dict, dict_classes


def parameters_search_dict(stats_dict, prop_classes, numbers_sample, source):

    if all(value == 0 for value in prop_classes.values()):
        return False

    total_pixel = sum(prop_classes.values())
    prop = [round(i / total_pixel * 100, 1) for i in prop_classes.values()]
    std = round(np.std(prop), 3)

    if std <= stats_dict['std'] or stats_dict['std'] == []:

        if params['data_analysis']['sampling']['method'][0] == 'min_annotated_percent':
            # adds 'map' value to stats_dict
            stats_dict.update(map=source)

        elif params['data_analysis']['sampling']['method'][0] == 'class_proportion':
            # adds 'combination' value to stats_dict
            stats_dict.update(combination=source)

        # adds 'std' value to stats_dict
        stats_dict.update(std=std)

        # adds 'prop_class' value to stats_dict
        for i in prop_classes:
            stats_dict.update({'prop' + str(i): prop[int(i)]})

        # appends 'trn', 'val', 'tst' values to stats_dict
        stats_dict.update(trn_data=numbers_sample)

    else:
        return False


def results(classes, stats_dict):
    if len(params['data_analysis']['sampling']['method']) == 1:
        if params['data_analysis']['sampling']['method'][0] == 'min_annotated_percent':
            print(' ')
            print('optimal minimum annotated percent :', stats_dict['map'])

        elif params['data_analysis']['sampling']['method'][0] == 'class_proportion':
            print(' ')
            print('optimal class threshold combination :', stats_dict['combination'])

    elif len(params['data_analysis']['sampling']['method']) == 2:
        if params['data_analysis']['sampling']['method'][1] == 'min_annotated_percent':
            print('optimal minimum annotated percent :', stats_dict['combination'])

        elif params['data_analysis']['sampling']['method'][1] == 'class_proportion':
            print('optimal class threshold combination :', stats_dict['map'])

    for i in classes:
        print('Pixels from class ', i, ' : ', stats_dict['prop' + str(i)], '%')
    print(stats_dict['trn_data'])


def parameters_search(sampling, sample_data, classes):

    annotated_p = [i for i in range(0, 101)]

    if sampling['method'][0] == 'min_annotated_percent':
        stats = minimum_annotated_percent_search(classes, annotated_p, sampling, sample_data)
        if len(sampling['method']) == 2:
            sample = []
            for row in sample_data:
                if minimum_annotated_percent(row[0], stats['map']):
                    sample.append(row)
            res = class_proportion_search(classes, sampling, sample)
            print('optimal minimum annotated percent :', stats['map'])
            results(classes, res)

    elif sampling['method'][0] == 'class_proportion':
        stats = class_proportion_search(classes, sampling, sample_data)
        if len(sampling['method']) == 2:
            sample = []
            for row in sample_data:
                if class_proportion(row, classes, stats['combination']):
                    sample.append(row)
            res = minimum_annotated_percent_search(classes, annotated_p, sampling, sample)

            print('optimal class threshold combination :', stats['combination'])
            results(classes, res)


def main(params):  # TODO: test this.
    number_samples = {'trn': 0, 'val': 0, 'tst': 0}
    prop_csv = params['global']['data_path'] + '/prop_data.csv'
    sampling = params['data_analysis']['sampling_method']

    if params['data_analysis']['create_csv']:
        create_csv()

    sample_data = genfromtxt(prop_csv, delimiter=',', dtype='|U8')

    # creates pixel_classes dict and keys
    classes = []
    if "class_proportion" in sampling.keys():
        pixel_classes = {key: 0 for key in sampling["class_proportion"].keys()}
        classes = [int(key) for key in sampling["class_proportion"].keys()]

    if params['data_analysis']['optimal_parameters_search']:
        parameters_search(sampling, sample_data, classes)

    else:
        for row in sample_data:
            if len(list(sampling.keys())) == 1:
                if list(sampling.keys())[0] == 'min_annotated_percent':
                    if minimum_annotated_percent(row[0], sampling['map']):
                        # adds pixel count to pixel_classes dict for each class in the image
                        compute_classes(classes, pixel_classes, row, number_samples)

                elif list(sampling.keys())[0] == 'class_proportion':
                    if class_proportion(row, classes, sampling):
                        # adds pixel count to pixel_classes dict for each class in the image
                        compute_classes(classes, pixel_classes, row, number_samples)

            elif len(list(sampling.keys())) == 2:
                if list(sampling.keys())[0] == 'min_annotated_percent':
                    if minimum_annotated_percent(row[0], sampling['map']):
                        if list(sampling.keys())[1] == 'class_proportion':
                            if class_proportion(row, classes, sampling):
                                compute_classes(classes, pixel_classes, row, number_samples)

                elif list(sampling.keys())[0] == 'class_proportion':
                    if class_proportion(row, classes, sampling):
                        if list(sampling.keys())[1] == 'min_annotated_percent':
                            if minimum_annotated_percent(row[0], sampling['map']):
                                compute_classes(classes, pixel_classes, row, number_samples)

        total_pixel = 0
        for i in pixel_classes:
            total_pixel += pixel_classes[i]

        for i in pixel_classes:
            print('Pixels from class ', i, ' :', round(pixel_classes[i]/total_pixel * 100, 1), ' %')
        print(number_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample preparation')
    parser.add_argument('ParamFile', metavar='DIR',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    params = read_parameters(args.ParamFile)

    start_time = time.time()

    debug = True if params['global']['debug_mode'] else False

    main(params)

    print("Elapsed time:{}".format(time.time() - start_time))
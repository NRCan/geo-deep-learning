import torch
# import torch should be first. Unclear issue, mentionned here: https://github.com/pytorch/pytorch/issues/2083
import os
import numpy as np
import rasterio
import warnings
from ruamel_yaml import YAML
import fiona
import csv
try:
    import boto3
except ModuleNotFoundError:
    warnings.warn('The boto3 library counldn\'t be imported. Ignore if not using AWS s3 buckets', ImportWarning)
    pass


def create_or_empty_folder(folder):
    """Empty an existing folder or create it if it doesn't exist.
    Args:
        folder: full file path of the folder to be emptied/created
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)


def read_parameters(param_file):
    """Read and return parameters in .yaml file
    Args:
        param_file: Full file path of the parameters file
    Returns:
        YAML (Ruamel) CommentedMap dict-like object
    """
    yaml = YAML()
    with open(param_file) as yamlfile:
        params = yaml.load(yamlfile)
    return params


def assert_band_number(in_image, band_count_yaml):
    """Verify if provided image has the same number of bands as described in the .yaml
    Args:
        in_image: full file path of the image
        band_count_yaml: band count listed in the .yaml
    """
    try:
        in_array = image_reader_as_array(in_image)
    except Exception as e:
        print(e)

    msg = "The number of bands in the input image and the parameter 'number_of_bands' in the yaml file must be the same"
    if band_count_yaml == 1:
        assert len(in_array.shape) == 2, msg
    else:
        assert in_array.shape[2] == band_count_yaml, msg


def load_from_checkpoint(filename, model, optimizer=None):
    """Load weights from a previous checkpoint
    Args:
        filename: full file path of file containing checkpoint
        model: model to replace
        optimizer: optimiser to be used
    """
    if os.path.isfile(filename):
        print("=> loading model '{}'".format(filename))

        if torch.cuda.is_available():
            checkpoint = torch.load(filename)
        else:
            checkpoint = torch.load(filename, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

        print("=> loaded model '{}'".format(filename))
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            return model, optimizer
        elif optimizer is None:
            return model
    else:
        print("=> no model found at '{}'".format(filename))


def image_reader_as_array(file_name):
    """Read an image from a file and return a 3d array (h,w,c)
    Args:
        file_name: full file path of the image

    Return:
        numm_py_array of the image read
    """

    with rasterio.open(file_name, 'r') as src:
        np_array = np.empty([src.height, src.width, src.count], dtype=np.float32)
        for i in range(src.count):
            band = src.read(i+1)  # Bands starts at 1 in rasterio not 0
            np_array[:, :, i] = band

    return np_array


def validate_num_classes(vector_file, num_classes, value_field):
    """Validate that the number of classes in the vector file corresponds to the expected number
    Args:
        vector_file: full file path of the vector image
        num_classes: number of classes set in config.yaml
        value_field: name of the value field representing the required classes in the vector image file

    Return:
        None
    """

    distinct_att = set()
    with fiona.open(vector_file, 'r') as src:
        for feature in src:
            distinct_att.add(feature['properties'][value_field])  # Use property of set to store unique values

    if len(distinct_att)+1 != num_classes:
        raise ValueError('The number of classes in the yaml.config {} is different than the number of classes in '
                         'the file {} {}'.format (num_classes, vector_file, str(list(distinct_att))))


def list_s3_subfolders(bucket, data_path):
    list_classes = []

    client = boto3.client('s3')
    result = client.list_objects(Bucket=bucket, Prefix=data_path+'/', Delimiter='/')
    for p in result.get('CommonPrefixes'):
        if p['Prefix'].split('/')[-2] is not data_path:
            list_classes.append(p['Prefix'].split('/')[-2])
    return list_classes


def read_csv(csv_file_name, inference=False):
    """Open csv file and parse it, returning a list of dict.

    If inference == True, the dict contains this info:
    - tif full path
    Else, the returned list contains a dict with this info:
    - tif full path
    - gpkg full path
    - attribute_name
    - dataset (trn or val)
    """

    list_values = []
    with open(csv_file_name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if inference:
                list_values.append({'tif': row[0]})
            else:
                list_values.append({'tif': row[0], 'gpkg': row[1], 'attribute_name': row[2], 'dataset': row[3]})

    if inference:
        return list_values
    else:
        return sorted(list_values, key=lambda k: k['dataset'])

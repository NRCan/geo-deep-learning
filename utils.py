import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
from PIL import Image
from ruamel_yaml import YAML
from osgeo import gdal, ogr
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


def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns:
        dictionary of memory usage values in MB. Keys are device ids as integers.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def plot_some_results(data, target, img_suffix, work_file):
    """Plots data. Used for visualization during development.
    __author__ = 'Fabian Isensee' """
    d = data
    s = target
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(d.transpose(1, 2, 0))
    plt.title("input patch")
    plt.subplot(1, 3, 2)
    plt.imshow(s)
    plt.title("ground truth")
    plt.savefig(os.path.join(work_file, "result_%03.0f.png" % img_suffix))
    plt.close()


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


def create_new_raster_from_base(input_raster, output_raster, band_count, write_array=None):
    """Function to use info from input raster to create new one.
    Args:
        input_raster: input raster path and name
        output_raster: raster name and path to be created with info from input
        band_count: number of bands in the input raster
        write_array (optional): array to write into the new raster
    """
    input_image = gdal.Open(input_raster)
    src = input_image
    cols = src.RasterXSize
    rows = src.RasterYSize
    projection = src.GetProjection()
    geotransform = src.GetGeoTransform()

    new_raster = gdal.GetDriverByName('GTiff').Create(output_raster, cols, rows, band_count, gdal.GDT_Byte)
    new_raster.SetProjection(projection)
    new_raster.SetGeoTransform(geotransform)

    for band_num in range(0, band_count):
        band = new_raster.GetRasterBand(band_num + 1)
        band.SetNoDataValue(-9999)
        # Write array if provided. If not, the image is filled with NoDataValues
        if write_array is not None:
            band.WriteArray(write_array[:, :, band_num])
            band.FlushCache()
    return new_raster


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
    """
    raster = gdal.Open(file_name)
    band_num = raster.RasterCount
    band = raster.GetRasterBand(1)
    rows, columns = (band.XSize, band.YSize)

    np_array = np.empty([columns, rows, band_num], dtype=np.uint8)

    for i in range(0, band_num):
        band = raster.GetRasterBand(i + 1)
        arr = band.ReadAsArray()
        np_array[:, :, i] = arr

    return np_array


def validate_num_classes(vector_file, num_classes, value_field):
    """Validate that the number of classes in the .shp corresponds to the expected number
    Args:
        vector_file: full file path of the vector image
        num_classes: number of classes set in config.yaml
        value_field: name of the value field representing the required classes in the vector image file
    """
    source_ds = ogr.Open(vector_file)
    source_layer = source_ds.GetLayer()
    name_lyr = source_layer.GetLayerDefn().GetName()
    vector_classes = source_ds.ExecuteSQL("SELECT DISTINCT " + value_field + " FROM " + name_lyr).GetFeatureCount()
    if vector_classes + 1 != num_classes:
        raise ValueError('The number of classes in the yaml.config (%d) is different than the number of classes in '
                         'the file %s (%d)' % (num_classes, vector_file, vector_classes))


def list_s3_subfolders(bucket, data_path):
    list_classes = []

    client = boto3.client('s3')
    result = client.list_objects(Bucket=bucket, Prefix=data_path+'/', Delimiter='/')
    for p in result.get('CommonPrefixes'):
        if p['Prefix'].split('/')[-2] is not data_path:
            list_classes.append(p['Prefix'].split('/')[-2])
    return list_classes


def image_loader(path):
    """Image loader used for classification tasks in train_model.py and image_classification.py.
    The resized image is set to 76 so that small images can work be used with inception-v3.
    Args:
          path: full file path of the image
    Returns:
          img: PIL Image"""
    img = Image.open(path)
    width, height = img.size
    if height < 76:
        adjustment = 76/height
        img = img.resize((round(width * adjustment), round(height * adjustment)), resample=Image.BILINEAR)
    width, height = img.size
    if width < 76:
        adjustment = 76/width
        img = img.resize((round(width * adjustment), round(height * adjustment)), resample=Image.BILINEAR)
    return img

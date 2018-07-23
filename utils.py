import subprocess
from ruamel_yaml import YAML

import matplotlib.pyplot as plt
import os
import gdal
import numpy as np
from PIL import Image
import torch


def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
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

def plot_some_results(data, target, img_sufixe, dossierTravail):
    """__author__ = 'Fabian Isensee'
    https://github.com/Lasagne/Recips/blob/master/examples/UNet/massachusetts_road_segm.py
    """
    d = data
    s = target
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(d.transpose(1,2,0))
    plt.title("input patch")
    plt.subplot(1, 3, 2)
    plt.imshow(s)
    plt.title("ground truth")
    plt.savefig(os.path.join(dossierTravail, "result_%03.0f.png"%img_sufixe))
    plt.close()

def ReadParameters(ParamFile):
    """Read and return parameters in .yaml file
    Args:
        ParamFile: Full file path
    Returns:
        YAML (Ruamel) CommentedMap dict-like object
    """
    yaml = YAML()
    with open(ParamFile) as yamlfile:
        params = yaml.load(yamlfile)
    return params

def CreateNewRasterFromBase(input_raster, output_raster, band_count, write_array=None):
    """Function to use info from input raster to create new one.
    args:
    input_raster: input raster path and name
    output_raster: raster name and path to be created with info from input
    write_array (optional): array to write into the new raster
    """

    # Get info on input raster
    inputImage = gdal.Open(input_raster)
    src = inputImage
    cols = src.RasterXSize
    rows = src.RasterYSize
    projection = src.GetProjection()
    geotransform = src.GetGeoTransform()

    # Create new raster
    new_raster = gdal.GetDriverByName('GTiff').Create(output_raster, cols, rows, band_count, gdal.GDT_Byte)
    new_raster.SetProjection(projection)
    new_raster.SetGeoTransform(geotransform)

    for band_num in range(0, band_count):
        band = new_raster.GetRasterBand(band_num+1)
        band.SetNoDataValue(-9999)
        # Write array if provided. If not, the image is filled with NoDataValues
        if write_array is not None:
            if band_count > 1:
                band.WriteArray(write_array[:, :, band_num])
            else:
                band.WriteArray(write_array)
            band.FlushCache()

    inputImage = None
    return new_raster

def AssertBandNumber(in_image, band_count_yaml):
    in_array = np.array(Image.open(in_image))
    msg = "The number of band in the input image and the parameter 'number_of_bands' in the yaml file must be the same"
    if band_count_yaml == 1:
        assert len(in_array.shape) == 2, msg
    else:
        assert in_array.shape[2] == band_count_yaml, msg

def LoadFromCheckpoint(filename, model, optimizer=None):
    """function to load weights from a checkpoint"""
    if os.path.isfile(filename):
        print("=> loading model '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded model '{}'".format(filename))
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            return model, optimizer
        elif optimizer is None:
            return model
    else:
        print("=> no model found at '{}'".format(filename))

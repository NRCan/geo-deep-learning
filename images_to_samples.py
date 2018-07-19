# -*- coding: utf-8 -*-

import argparse
import csv
import os
import random

from PIL import Image
import gdal, osr, ogr
import h5py
from skimage import exposure

import numpy as np
from utils import ReadParameters

def createNewRasterFromBase(input_raster, output_raster, band_count, write_array=None):
    """Function to use info from input raster to create new one.
    args:
    input_raster: raster base
    output_raster: created raster with info from input
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

def CreateOrEmptyFolder(folder):
    """Create a folder if it does not exist and empty it if it exist"""

    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

def MaskImage(arrayA, arrayB):
    """Function to mask values of arrayB, based on 0 values from arrayA."""

    if len(arrayA.shape) == 3:
        mask = arrayA[:,:,0] != 0
    else:
        mask = arrayA != 0

    ma_array = np.zeros(arrayB.shape, dtype=np.uint8)
    if len(arrayB.shape) == 3:
        for i in range(0,3):
            ma_array[:, :, i] = mask*arrayB[:, :, i]

    else:
        ma_array = arrayB*mask
    return ma_array

def RandomSamples(num_samples, samples_file, tmp_samples_file):
    """Read prepared samples and rewrite them in random order."""

    RdmEchant = RdmList(num_samples)
    for elem in RdmEchant:

        data = tmp_samples_file['sat_img'][elem, ...]
        target = tmp_samples_file['map_img'][elem, ...]

        idx = RdmEchant.index(elem)

        samples_file["sat_img"][idx, ...] = data
        samples_file["map_img"][idx, ...] = target

def RdmList(len_list):
    """Create and return a list with random number in range len_list"""

    listRdm = []
    for i in range(len_list):
        listRdm.append(i)
    random.shuffle(listRdm)
    return listRdm

def ReadCSV(CSVFileName):
    """Open csv file and parse it."""

    list_values = []
    with open(CSVFileName, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            list_values.append({'tif': row[0], 'shp': row[1],'attribute_name': row[2], 'dataset': row[3]})

    return sorted(list_values, key=lambda k: k['dataset'])

def ResizeDatasets(hdf5_file):
    """Function to resize the datasets"""

    n = hdf5_file['sat_img'].shape[0]

    new_size = n + 1
    hdf5_file['sat_img'].resize(new_size, axis=0)
    hdf5_file['map_img'].resize(new_size, axis=0)

def ScaleIntensity(RGBArray):
    """Image enhancement. Rescale intensity to extend it to the range 0-255.
    based on: http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py"""

    v_min, v_max = np.percentile(RGBArray, (2, 98))
    scaledArray = np.nan_to_num(exposure.rescale_intensity(RGBArray, in_range=(v_min, v_max)))
    return scaledArray

def SamplesPreparation(sat_img, ref_img, sample_size, dist_samples, samples_count, num_classes, samples_file, dataset, backgroundSwitch):
    """Extract and write samples from RGB and reference images"""

    # read RGB and reference images as array
    RGBArray = np.array(Image.open(sat_img))
    scaled = ScaleIntensity(RGBArray)

    LabelArray = np.array(Image.open(ref_img))

    h, w, nbband = scaled.shape

    if dataset == 'trn':
        idx_samples = samples_count['trn']
    elif dataset == 'val':
        idx_samples = samples_count['val']

    # half tile padding
    half_tile = int(sample_size/2)
    pad_RGB_Array = np.pad(scaled, ((half_tile, half_tile),(half_tile, half_tile),(0,0)), mode='constant')
    pad_Label_Array = np.pad(LabelArray, ((half_tile, half_tile),(half_tile, half_tile)), mode='constant')

    for row in range(0, h, dist_samples):
        for column in range(0, w, dist_samples):
            data = (pad_RGB_Array[row:row+sample_size, column:column+sample_size,:])
            target = (pad_Label_Array[row:row+sample_size, column:column+sample_size])

            targetClassNum = max(target.ravel())

            if (backgroundSwitch and targetClassNum != 0) or (not backgroundSwitch):
                # Write if there are more than 2 classes in samples.
                ResizeDatasets(samples_file)
                samples_file["sat_img"][idx_samples, ...] = data
                samples_file["map_img"][idx_samples, ...] = target
                idx_samples+=1

            # update the number of classes in reference images
            if num_classes < targetClassNum:
                num_classes = targetClassNum

    if dataset == 'trn':
        samples_count['trn'] = idx_samples
    elif dataset == 'val':
        samples_count['val'] = idx_samples

    return samples_count, num_classes

def VectorToRaster(vector_file, attribute_name, new_raster):
    """Function to rasterize vector data."""

    source_ds = ogr.Open(vector_file)
    source_layer = source_ds.GetLayer()
    name_lyr = source_layer.GetLayerDefn().GetName()
    rev_lyr = source_ds.ExecuteSQL("SELECT * FROM " + name_lyr + " ORDER BY " + attribute_name + " ASC")

    # Rasterizer
    gdal.RasterizeLayer(new_raster, [1], rev_lyr, options=["ATTRIBUTE=%s" % attribute_name])

    source_ds = None
    source_layer = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample preparation')
    parser.add_argument('ParamFile', metavar='DIR',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    params = ReadParameters(args.ParamFile)
    csv_file = params['sample']['prep_csv_file']
    data_path = params['global']['data_path']
    samples_size = params['global']['samples_size']
    samples_dist = params['sample']['samples_dist']
    remove_background = params['sample']['remove_background']
    mask_input_image = params['sample']['mask_input_image']

    # Folder preparation and creation
    samples_folder = os.path.join(data_path, "samples")
    out_label_folder = os.path.join(data_path, "label")

    CreateOrEmptyFolder(samples_folder)
    CreateOrEmptyFolder(out_label_folder)

    list_data_prep = ReadCSV(csv_file)

    number_samples = {'trn': 0, 'val': 0}
    number_classes = 0

    # Create tmp file and datasets for samples storage (in images order).
    tmp_trn_hdf5 = h5py.File(os.path.join(samples_folder, "trn_tmp_samples.hdf5"), "w")
    tmp_val_hdf5 = h5py.File(os.path.join(samples_folder, "val_tmp_samples.hdf5"), "w")

    tmp_trn_hdf5.create_dataset("sat_img", (0, samples_size, samples_size, 3), np.uint8, maxshape=(None, samples_size, samples_size, 3))
    tmp_trn_hdf5.create_dataset("map_img", (0, samples_size, samples_size), np.uint8, maxshape=(None, samples_size, samples_size))
    tmp_val_hdf5.create_dataset("sat_img", (0, samples_size, samples_size, 3), np.uint8, maxshape=(None, samples_size, samples_size, 3))
    tmp_val_hdf5.create_dataset("map_img", (0, samples_size, samples_size), np.uint8, maxshape=(None, samples_size, samples_size))

    # loop in rows in csv
    for info in list_data_prep:
        img_name = os.path.basename(info['tif']).split('.')[0]
        tmp_label_name = os.path.join(out_label_folder, img_name + "_label_tmp.tif")
        label_name = os.path.join(out_label_folder, img_name + "_label.tif")

        print(img_name)

        # Create temp raster and rasterize values from shp in it.
        tmp_label_raster = createNewRasterFromBase(info['tif'], tmp_label_name, 1)
        VectorToRaster(info['shp'], info['attribute_name'], tmp_label_raster)
        tmp_label_raster = None

        # Mask zeros from input image into label raster.
        maskedArray = MaskImage(np.array(Image.open(info['tif'])), np.array(Image.open(tmp_label_name)))
        createNewRasterFromBase(info['tif'], label_name, 1, maskedArray)

        # For ECCC data
        # Mask zeros from label raster into input image.
        if mask_input_image:
            maskedImg = MaskImage(np.array(Image.open(label_name)), np.array(Image.open(info['tif'])))
            createNewRasterFromBase(label_name, info['tif'], 3, maskedImg)

        os.remove(tmp_label_name)

        if info['dataset'] == 'trn':
            out_file = tmp_trn_hdf5
        elif info['dataset'] == 'val':
            out_file = tmp_val_hdf5

        number_samples, number_classes = SamplesPreparation(info['tif'], label_name, samples_size, samples_dist, number_samples, number_classes, out_file, info['dataset'], remove_background)
        print(number_samples)
        out_file.flush()

    tmp_trn_hdf5.close()
    tmp_val_hdf5.close()

    # Create file and datasets for samples storage (in random order).
    for dset in (['trn', 'val']):
        hdf5_file = h5py.File(os.path.join(samples_folder, dset + "_samples.hdf5"), "w")
        hdf5_file.create_dataset("sat_img", (number_samples[dset], samples_size, samples_size, 3), np.uint8)
        hdf5_file.create_dataset("map_img", (number_samples[dset], samples_size, samples_size), np.uint8)

        tmp_hdf5 = h5py.File(os.path.join(samples_folder, dset + "_tmp_samples.hdf5"), "r")
        RandomSamples(number_samples[dset], hdf5_file, tmp_hdf5)

        tmp_hdf5.close()
        os.remove(os.path.join(samples_folder, dset + "_tmp_samples.hdf5"))

    print("Number of samples created: ", number_samples)
    print("End of process")

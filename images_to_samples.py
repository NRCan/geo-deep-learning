import numpy as np
import os
from PIL import Image
import fnmatch
import random
import argparse
from utils import ReadParameters

def WriteArray(file, npArray):
    """write numpy array in binary and append mode"""
    with open(file, 'ab') as out_file:
        npArray.tofile(out_file)

def rdmList(len_list):
    """Create and return a list with random number in range len_list"""
    listRdm = []
    for i in range(len_list):
        listRdm.append(i)
    random.shuffle(listRdm)
    return listRdm

def WriteInfo(output_folder, num_samples, num_classes):
    """Write txt file containing number of created samples and number of classes."""
    with open(os.path.join(output_folder, "info.txt"), 'a') as the_file:
        the_file.write(str(num_samples) + "\n")
        the_file.write(str(num_classes + 1) + "\n")
    the_file.close()


def SamplesPreparation(sat_img, ref_img, ImagesFolder, OutputFolder, sample_size, dist_samples):
    """Extract and write samples from a list of RGB and reference images
    During training, loading array is much more efficient than images."""
    assert(len(sat_img) == len(ref_img))

    num_samples = 0
    num_classes = 0

    for img in sat_img:

        # read RGB and reference images as array
        RGBArray = np.array(Image.open(os.path.join(ImagesFolder, "RGB", img)))
        LabelArray = np.array(Image.open(os.path.join(ImagesFolder, "label", ref_img[sat_img.index(img)])))
        h, w, nbband = RGBArray.shape
        print(img, ' ', RGBArray.shape)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        transp = np.transpose(RGBArray, (2, 0, 1))

        # half tile padding
        half_tile = int(sample_size/2)
        pad_RGB_Array = np.pad(transp, ((0,0),(half_tile, half_tile),(half_tile, half_tile)), mode='constant')
        pad_Label_Array = np.pad(LabelArray, ((half_tile, half_tile),(half_tile, half_tile)), mode='constant')

        for row in range(0, h, dist_samples):
            for column in range(0, w, dist_samples):
                data = (pad_RGB_Array[:,row:row+sample_size, column:column+sample_size])
                target = (pad_Label_Array[row:row+sample_size, column:column+sample_size])

                num_samples+=1

                WriteArray(os.path.join(OutputFolder, "tmp_samples_RGB.dat"), data)
                WriteArray(os.path.join(OutputFolder, "tmp_samples_Label.dat"), target)

                if num_classes < max(target.ravel()):
                    num_classes = max(target.ravel())
    return num_samples, num_classes

def RandomSamples(OutputFolder, sample_size, num_samples):
    """Read prepared samples and rewrite them in random order."""
    RdmEchant = rdmList(num_samples)

    for idx in RdmEchant:
        data_file = open(os.path.join(OutputFolder, "tmp_samples_RGB.dat"), "rb")
        ref_file = open(os.path.join(OutputFolder, "tmp_samples_Label.dat"), "rb")

        data_file.seek(idx*sample_size*sample_size*3)
        ref_file.seek(idx*sample_size*sample_size)

        data = np.fromfile(data_file, dtype=np.uint8, count=3*sample_size*sample_size)
        target = np.fromfile(ref_file, dtype=np.uint8, count=sample_size*sample_size)

        WriteArray(os.path.join(OutputFolder, "samples_RGB.dat"), data)
        WriteArray(os.path.join(OutputFolder, "samples_Label.dat"), target)

    # Delete temp files.
    data_file.close()
    ref_file.close()
    os.remove(os.path.join(OutputFolder, "tmp_samples_RGB.dat"))
    os.remove(os.path.join(OutputFolder, "tmp_samples_Label.dat"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample preparation')
    parser.add_argument('ParamFile', metavar='DIR',
                        help='path to parameters txt')
    args = parser.parse_args()
    params = ReadParameters(args.ParamFile)
    images_folder =  params['sample']['images_folder']
    samples_folder = params['sample']['samples_folder']
    samples_size = params['sample']['samples_size']
    samples_dist = params['sample']['samples_dist']

    # List RGB and reference images in both folders.
    images_RGB = [img for img in os.listdir(os.path.join(images_folder, "RGB")) if fnmatch.fnmatch(img, "*.tif*")]
    images_Label = [img for img in os.listdir(os.path.join(images_folder, "label")) if fnmatch.fnmatch(img, "*.tif*")]
    images_RGB.sort()
    images_Label.sort()

    nbrsamples, nbrclasses = SamplesPreparation(images_RGB, images_Label, images_folder, samples_folder, samples_size, samples_dist)
    RandomSamples(samples_folder, samples_size, nbrsamples)
    WriteInfo(samples_folder, nbrsamples, nbrclasses)
    print("End of process")

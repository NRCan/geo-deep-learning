import numpy as np
import os
from PIL import Image
import fnmatch
import random
import argparse
from utils import ReadParameters
import h5py
from skimage import exposure

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
    
def ScaleIntensity(RGBArray):
    """Image enhancement. Rescale intensity to extend it to the range 0-255.
    based on: http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py"""
    v_min, v_max = np.percentile(RGBArray, (2, 98))
    scaledArray = np.nan_to_num(exposure.rescale_intensity(RGBArray, in_range=(v_min, v_max)))
    return scaledArray 

def CreateOrEmptyFolder(folder):
    """Create a folder if it does not exist and empty it if it exist"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

def SamplesPreparation(sat_img, ref_img, ImagesFolder, dataset, OutputFolder, sample_size, dist_samples, backgroundSwitch):
    """Extract and write samples from a list of RGB and reference images
    During training, loading array is much more efficient than images.
    This code is mainly based on: https://medium.com/the-downlinq/broad-area-satellite-imagery-semantic-segmentation-basiss-4a7ea2c8466f"""
    
    assert(len(sat_img) == len(ref_img))

    num_samples = 0
    num_classes = 0

    for img in sat_img:

        # read RGB and reference images as array
        RGBArray = np.array(Image.open(os.path.join(ImagesFolder, dataset, "RGB", img)))
        scaled = ScaleIntensity(RGBArray)
        
        LabelArray = np.array(Image.open(os.path.join(ImagesFolder, dataset, "label", ref_img[sat_img.index(img)])))
        
        h, w, nbband = scaled.shape
        print(img, ' ', scaled.shape)

        # half tile padding
        half_tile = int(sample_size/2)
        pad_RGB_Array = np.pad(scaled, ((half_tile, half_tile),(half_tile, half_tile),(0,0)), mode='constant')
        pad_Label_Array = np.pad(LabelArray, ((half_tile, half_tile),(half_tile, half_tile)), mode='constant')

        for row in range(0, h, dist_samples):
            for column in range(0, w, dist_samples):
                data = (pad_RGB_Array[row:row+sample_size, column:column+sample_size,:])
                target = (pad_Label_Array[row:row+sample_size, column:column+sample_size])

                targetClassNum = max(target.ravel())

                if backgroundSwitch and targetClassNum != 0:
                    # Write if there are more than 2 classes in samples.
                    WriteArray(os.path.join(OutputFolder, "tmp_samples_RGB.sam"), data)
                    WriteArray(os.path.join(OutputFolder, "tmp_samples_Label.sam"), target)
                    num_samples+=1

                elif not backgroundSwitch:
                    WriteArray(os.path.join(OutputFolder, "tmp_samples_RGB.sam"), data)
                    WriteArray(os.path.join(OutputFolder, "tmp_samples_Label.sam"), target)
                    num_samples+=1

                # update the number of classes in reference images
                if num_classes < targetClassNum:
                    num_classes = targetClassNum

    return num_samples, num_classes

def RandomSamples(OutputFolder, sample_size, num_samples, dataset):
    """Read prepared samples and rewrite them in random order."""
    
    RdmEchant = rdmList(num_samples)
    
    sat_img_shape = (num_samples, sample_size, sample_size, 3)
    ref_img_shape = (num_samples, sample_size, sample_size)
        
    hdf5_file = h5py.File(os.path.join(OutputFolder, dataset + "_samples.hdf5"), "w")

    hdf5_file.create_dataset("sat_img", sat_img_shape, np.uint8)
    hdf5_file.create_dataset("map_img", ref_img_shape, np.uint8)

    data_file = open(os.path.join(OutputFolder, "tmp_samples_RGB.sam"), "rb")
    ref_file = open(os.path.join(OutputFolder, "tmp_samples_Label.sam"), "rb")

    for elem in RdmEchant:

        data_file.seek(elem*sample_size*sample_size*3)
        ref_file.seek(elem*sample_size*sample_size)

        data = np.reshape(np.fromfile(data_file, dtype=np.uint8, count=3*sample_size*sample_size), [sample_size, sample_size, 3])
        target = np.reshape(np.fromfile(ref_file, dtype=np.uint8, count=sample_size*sample_size), [sample_size, sample_size])
        
        idx = RdmEchant.index(elem)
        hdf5_file["sat_img"][idx, ...] = data
        hdf5_file["map_img"][idx, ...] = target

    # Delete temp files
    data_file.close()
    ref_file.close()
    
    os.remove(os.path.join(OutputFolder, "tmp_samples_RGB.sam"))
    os.remove(os.path.join(OutputFolder, "tmp_samples_Label.sam"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample preparation')
    parser.add_argument('ParamFile', metavar='DIR',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    params = ReadParameters(args.ParamFile)
    data_path = params['global']['data_path']
    samples_size = params['global']['samples_size']
    samples_dist = params['sample']['samples_dist']
    remove_background = params['sample']['remove_background']
    
    for dataset in ['trn', 'val']:
        
        samples_folder = os.path.join(data_path, dataset, 'samples')
        CreateOrEmptyFolder(samples_folder)
        
        # List RGB and reference images in both folders.
        images_RGB = [img for img in os.listdir(os.path.join(data_path, dataset, "RGB")) if fnmatch.fnmatch(img, "*.tif*")]
        images_Label = [img for img in os.listdir(os.path.join(data_path, dataset, "label")) if fnmatch.fnmatch(img, "*.tif*")]
        images_RGB.sort()
        images_Label.sort()
    
        nbrsamples, nbrclasses = SamplesPreparation(images_RGB, images_Label, data_path, dataset, samples_folder, samples_size, samples_dist, remove_background)
        RandomSamples(samples_folder, samples_size, nbrsamples , dataset)
        WriteInfo(samples_folder, nbrsamples, nbrclasses)
        print(dataset, " samples created")
    print("End of process")

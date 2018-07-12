import numpy as np
import os
from torch.autograd import Variable
import torch
import unet_pytorch
import time
import argparse
from PIL import Image
import fnmatch
from utils import ReadParameters
# from torchvision import transforms
from skimage import exposure

from osgeo import gdal

def NormArray(HWCArray):
    
    transpData = np.float32(np.transpose(HWCArray, (2, 0, 1)))
    return torch.from_numpy(transpData)

def main(working_folder, img_list, num_classes, Weights_File_Name):
    """
    Args:
        working_folder:
        listImg:
        NbClasses:
        Weights_File_Name:
    """
    # get model
    model = unet_pytorch.UNetSmall(num_classes)

    if torch.cuda.is_available():
        model = model.cuda()

    # load weights
    if os.path.isfile(Weights_File_Name):
        print("=> loading model '{}'".format(Weights_File_Name))
        checkpoint = torch.load(Weights_File_Name)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded weights '{}'".format(Weights_File_Name))
    else:
        print("=> no checkpoint found at '{}'".format(Weights_File_Name))

    since = time.time()

    for img in img_list:
        Classification(working_folder, model, img)
        print('Image ', img, ' classified')

    time_elapsed = time.time() - since
    print('Classification complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def Classification(folderImages, model, image):
    """
    Args:
        model:
        image:
    Returns:
    """
    # Chunk size. Should not be modified often. We want the biggest chunk to be process at a time but,
    # a too large image chunk bust the GPU memory when processing.
    chunk_size = 1024

    # switch to evaluate mode
    model.eval()

    RGBArray = np.array(Image.open(os.path.join(folderImages, image)))
    h, w, nb = RGBArray.shape
    
    paddedArray = np.pad(RGBArray, ((0, int(chunk_size/2)),(0, int(chunk_size/2)),(0,0)), mode='constant')
    
    outputNP = np.empty([h,w], dtype=np.uint8)
    
    with torch.no_grad():
        for row in range(0, h, chunk_size):
            for col in range(0, w, chunk_size):
                
                partRGB = paddedArray[row:row+chunk_size, col:col+chunk_size, :]
                
                TorchData = NormArray(partRGB)
                TorchData.unsqueeze_(0)
                
                # get the inputs and wrap in Variable
                if torch.cuda.is_available():
                    inputs = Variable(TorchData.cuda())
                else:
                    inputs = Variable(TorchData)
                # forward
                outputs = model(inputs)

                a, pred = torch.max(outputs, dim=1)
                segmentation = torch.squeeze(pred)
                
                reslon, reslarg = outputNP[row:row+chunk_size, col:col+chunk_size].shape
                outputNP[row:row+chunk_size, col:col+chunk_size] = segmentation[:reslon, :reslarg]
                
        CreateNewRasterFromBase(os.path.join(folderImages, image), os.path.join(folderImages, image.split('.')[0] + '_classif.tif'), outputNP)

def CreateNewRasterFromBase(InputRasterPath, OutputRasterFn, array):
    """Read RGB image geospatial information and write them in the out raster"""
    # Read info
    inputImage = gdal.Open(InputRasterPath)
    src = inputImage
    cols = src.RasterXSize
    rows = src.RasterYSize
    projection = src.GetProjection()
    geotransform = src.GetGeoTransform()

    # write info
    new_raster = gdal.GetDriverByName('GTiff').Create(OutputRasterFn, cols, rows, 1, gdal.GDT_Byte)
    new_raster.SetProjection(projection)
    new_raster.SetGeoTransform(geotransform)
    band = new_raster.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    band.WriteArray(array)

if __name__ == '__main__':
    print('Start: ')
    parser = argparse.ArgumentParser(description='Image classification using trained model')
    parser.add_argument('param_file', metavar='file',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()

    params = ReadParameters(args.param_file)
    working_folder = params['classification']['working_folder']
    model_name = params['classification']['model_name']
    num_classes = params['global']['num_classes']


    listImg = [img for img in os.listdir(working_folder) if fnmatch.fnmatch(img, "*.tif*")]
    main(working_folder, listImg, num_classes, model_name)

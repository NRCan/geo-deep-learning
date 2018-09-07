import numpy as np
import os
from torch.autograd import Variable
import torch
import unet_pytorch
import time
import argparse
import fnmatch
from utils import ReadParameters, CreateNewRasterFromBase, AssertBandNumber, LoadFromCheckpoint, ImageReaderAsArray


def main(working_folder, img_list, num_classes, Weights_File_Name, in_image_band_count):
    """
    Args:
        working_folder:
        listImg:
        NbClasses:
        Weights_File_Name:
    """
    # get model
    model = unet_pytorch.UNetSmall(num_classes, number_of_bands)

    if torch.cuda.is_available():
        model = model.cuda()

    # load weights
    model = LoadFromCheckpoint(Weights_File_Name, model)

    since = time.time()

    for img in img_list:
        # assert that img band and the parameter in yaml have the same value
        AssertBandNumber(os.path.join(working_folder, img), number_of_bands)
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
    chunk_size = 512

    # switch to evaluate mode
    model.eval()

    input_image = ImageReaderAsArray(os.path.join(folderImages, image))
    if len(input_image.shape) == 3:
        h, w, nb = input_image.shape
        paddedArray = np.pad(input_image, ((0, int(chunk_size/2)),(0, int(chunk_size/2)),(0,0)), mode='constant')
    elif len(input_image.shape) == 2:
        h, w = input_image.shape
        paddedArray = np.expand_dims(np.pad(input_image, ((0, int(chunk_size/2)),(0, int(chunk_size/2))), mode='constant'), axis=0)

    outputNP = np.empty([h,w,1], dtype=np.uint8)

    with torch.no_grad():
        for row in range(0, h, chunk_size):
            for col in range(0, w, chunk_size):

                chunk_input = paddedArray[row:row+chunk_size, col:col+chunk_size, :]
                TorchData = torch.from_numpy(np.float32(np.transpose(chunk_input, (2, 0, 1))))

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

                reslon, reslarg, b = outputNP[row:row+chunk_size, col:col+chunk_size].shape
                outputNP[row:row+chunk_size, col:col+chunk_size, 0] = segmentation[:reslon, :reslarg]

        CreateNewRasterFromBase(os.path.join(folderImages, image), os.path.join(folderImages, image.split('.')[0] + '_classif.tif'), 1, outputNP)

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
    number_of_bands = params['global']['number_of_bands']


    listImg = [img for img in os.listdir(working_folder) if fnmatch.fnmatch(img, "*.tif*")]
    main(working_folder, listImg, num_classes, model_name, number_of_bands)

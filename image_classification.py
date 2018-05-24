import numpy as np
import os
from torch.autograd import Variable
import torch
import unet_pytorch_clas
import time
import argparse
from PIL import Image
import fnmatch
from utils import ReadParameters

def main(working_folder, img_list, num_classes, Weights_File_Name):
    """
    Args:
        working_folder:
        listImg:
        NbClasses:
        Weights_File_Name:
    """
    # get model
    model = unet_pytorch_clas.UNetSmall(num_classes)

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
    
    RGBArray = np.float32(np.array(Image.open(os.path.join(folderImages, image))))
    # transpose 
    # H x W x C to 
    # C x H x W  
    transp = np.transpose(RGBArray, (2, 0, 1))
    nb, h, w = transp.shape
    del RGBArray
    outputNP = np.empty([h,w], dtype=np.uint8)
    with torch.no_grad():
        for row in range(0, h, chunk_size):
            for col in range(0, w, chunk_size):
                partRGB = transp[:, row:row+chunk_size, col:col+chunk_size]
                TorchData = torch.from_numpy(partRGB)
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
                
                outputNP[row:row+chunk_size, col:col+chunk_size] = segmentation
                
        print(outputNP.shape)
        pilImage = Image.fromarray(np.uint8(outputNP))
        pilImage.save(os.path.join(folderImages, image.split('.')[0] + '_classif.tif'))
            
if __name__ == '__main__':
    """ 
    To be modified with yaml
    """
    
    parser = argparse.ArgumentParser(description='Image classification using trained model')
    parser.add_argument('param_file', metavar='file',
                        help='txt file containing parameters')
    args = parser.parse_args()

    print('Start: ')
    params = ReadParameters(args.param_file)
    working_folder = params[0]
    model_name = params[1]
    num_classes = int(params[2])
    
    listImg = [img for img in os.listdir(working_folder) if fnmatch.fnmatch(img, "*.tif*")]
    main(working_folder, listImg, num_classes, model_name)

        
import numpy as np
import os

from torch.autograd import Variable
import torch
import unet

import time
import argparse

from PIL import Image
import torchvision.transforms as transforms
import fnmatch

def main(TravailFolder, listImg, NbClasses, Weights_File_Name):
    """
    Args:
        TravailFolder:
        listImg:
        NbClasses:
        Weights_File_Name:
    """
    # get model
    model = unet.Unet(NbClasses)

    if torch.cuda.is_available():
        model = model.cuda()
    
    # load weights
    if os.path.isfile(os.path.join(TravailFolder, Weights_File_Name + '.pth')):
        print("=> loading model '{}'".format(Weights_File_Name))
        checkpoint = torch.load(os.path.join(TravailFolder, Weights_File_Name + '.pth'))

        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(Weights_File_Name))
    else:
        print("=> no checkpoint found at '{}'".format(os.path.join(TravailFolder, Weights_File_Name + '.pth')))
    
    since = time.time()
    
    
    for img in listImg:
        torch.cuda.empty_cache()

        Classification(model, img)
        
        cur_elapsed = time.time() - since
        print('Current elapsed time {:.0f}m {:.0f}s'.format(cur_elapsed // 60, cur_elapsed % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def Classification(model, image):
    """
    Args:
        model:
        image:
    Returns:
    """

    # switch to evaluate mode
    model.eval()
    # Iterate over data.
    torch.cuda.empty_cache()

    RGBArray = np.array(Image.open(image))
    
    transp = np.transpose(RGBArray, (2, 0, 1))
    # pad_RGB_Array = np.pad(transp, ((0,0),(int(tailleTuile / 2), int(tailleTuile / 2)),(int(tailleTuile / 2), int(tailleTuile / 2))), mode='constant')
    
    TorchData = torch.from_numpy(transp)
    with torch.no_grad():
        # get the inputs and wrap in Variable
        if torch.cuda.is_available():
            inputs = Variable(TorchData.cuda())
        else:
            inputs = Variable(TorchData)

        # forward
        outputs = model(inputs)    
        a, pred = torch.max(outputs, dim=1)
        segmentation = torch.squeeze(pred)
        pilImage = transforms.ToPILImage()(segmentation)
        imgName = image.split('.')[0]
        pilImage.save(os.path.join(TravailFolder, imgName + '_classif.tif'))
            


if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Road and Building Extraction')
#     parser.add_argument('data', metavar='DIR',
#                         help='path to dataset csv')
#     parser.add_argument('--epochs', default=75, type=int, metavar='N',
#                         help='number of total epochs to run')
#     parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                         help='epoch to start from (used with resume flag')
#     parser.add_argument('-b', '--batch-size', default=16, type=int,
#                         metavar='N', help='mini-batch size (default: 16)')
# 
#     args = parser.parse_args()

    # main(args.data, batch_size=args.batch_size, num_epochs=args.epochs, start_epoch=args.start_epoch)


    #### parametres ###
    print('Debut:')
    # TravailFolder = "D:\Processus\image_to_echantillons\img_1"
    TravailFolder = "/space/hall0/work/nrcan/geobase/extraction/Deep_learning/pytorch/"
    listImg = [img for img in os.listdir(TravailFolder) if fnmatch.fnmatch(img, "*.tif*")]
    ModelName = "unet.pth"
    nbrClasses = 4
    main(TravailFolder, listImg, nbrClasses, ModelName)
    print('Fin')

        
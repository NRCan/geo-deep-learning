# import matplotlib.pyplot as plt
import numpy as np
import os

from torch.autograd import Variable
from torch.utils.data import DataLoader
# from torchvision import transforms
from torch import nn
import metrics
import unet_pytorch
# import unet
import CreateDataset
# from utils import get_gpu_memory_map
from metrics import AverageMeter
# from sklearn.metrics import confusion_matrix

import torch
import torch.optim as optim
import time
import argparse
import shutil
from utils import ReadParameters

def flatten_labels(annotations):
    flatten = annotations.view(-1)
    return flatten

def flatten_outputs(predictions, number_of_classes):
    """Flattens the predictions batch except for the predictions dimension"""
    logits_permuted = predictions.permute(0, 2, 3, 1)
    logits_permuted_cont = logits_permuted.contiguous()
    outputs_flatten = logits_permuted_cont.view(-1, number_of_classes)
    return outputs_flatten


def save_checkpoint(state, is_best, filename='./checkpoint.pth.tar'):
    """
    Create a function to save the model state
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def main(working_folder, batch_size, num_epochs, learning_rate, sample_size, num_classes, num_samples_trn, num_samples_val):
    """
    Function to train and validate a model for semantic segmentation. 
    Args:
        working_folder
        batch_size:
        num_epochs:
        learning_rate:
        sample_size:
        num_classes:
        num_samples_trn:
        num_samples_val:
    Returns:
        File 'model_best.pth.tar' containing trained weights
    """
    since = time.time()

    # get model
    model = unet_pytorch.UNetSmall(num_classes)
    # model = unet.Unet(num_classes)

    if torch.cuda.is_available():
        model = model.cuda()
        
    # set up cross entropy
    poids = torch.tensor([1.,2.,2.,2.])
    criterion = nn.CrossEntropyLoss(weight=poids).cuda()

    # optimizer
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # loss initialisation
    best_loss = 999

    # get data
    trn_dataset = CreateDataset.SegmentationDataset(os.path.join(working_folder, "trn/samples"), num_samples_trn, sample_size)
    val_dataset = CreateDataset.SegmentationDataset(os.path.join(working_folder, "val/samples"), num_samples_val, sample_size)
    # creating loaders
    train_dataloader = DataLoader(trn_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    
    for epoch in range(0, num_epochs):
        print()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)
    
        torch.cuda.empty_cache()
        # run training and validation
        train_metrics = train(train_dataloader, model, criterion, optimizer, epoch, num_classes, batch_size)

        torch.cuda.empty_cache()
        
        # print("GPU momory: ", get_gpu_memory_map())
        
        Val_loss = validation(val_dataloader, model, criterion, epoch, num_classes, batch_size)
        
        torch.cuda.empty_cache()
        
        if Val_loss < best_loss:
            print("save checkpoint")
            save_checkpoint({'epoch': epoch, 'arch': 'UNetSmall', 'state_dict': model.state_dict(), 'best_loss': best_loss, 'optimizer': optimizer.state_dict()}, is_best=False)
            best_loss = Val_loss
            
        cur_elapsed = time.time() - since
        print('Current elapsed time {:.0f}m {:.0f}s'.format(cur_elapsed // 60, cur_elapsed % 60))
    save_checkpoint({'epoch': epoch, 'arch': 'UNetSmall', 'state_dict': model.state_dict(), 'best_loss': best_loss, 'optimizer': optimizer.state_dict()}, is_best=True)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def train(train_loader, model, criterion, optimizer, epoch_num, num_classes, batch_size):
    """
    Args:
        train_loader:
        model:
        criterion:
        optimizer:
        epoch:
    Returns:
    """
    model.train()
    
    # logging accuracy and loss
    train_acc = AverageMeter()
    train_loss = AverageMeter()

    # iterate over data
    for idx, data in enumerate(train_loader):
        # We need to flatten annotations and logits to apply index of valid annotations.
        # https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/master/pytorch_segmentation_detection/recipes/pascal_voc/segmentation/resnet_18_8s_train.ipynb
        # get the inputs and wrap in Variable
        if torch.cuda.is_available():
            inputs = Variable(data['sat_img'].cuda())
            labels = Variable(flatten_labels(data['map_img']).cuda())
        else:
            inputs = Variable(data['sat_img'])
            labels = Variable(flatten_labels(data['map_img']))
        torch.cuda.empty_cache()
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward
        outputs = model(inputs)
        outputs_flatten = flatten_outputs(outputs, num_classes)

        del outputs
        torch.cuda.empty_cache()
        
        loss = criterion(outputs_flatten, labels)
        
        train_loss.update(loss.item(), inputs.size(0))
        
        del inputs
        
        # backward
        loss.backward()
        optimizer.step()
        
        if idx == len(train_loader) - 1:
            a, segmentation = torch.max(outputs_flatten, dim=1)
            acc = metrics.accuracy(segmentation, labels)
            train_acc.update(acc, batch_size)
            
    print('Training Loss: {:.4f} Acc: {:.4f}'.format(train_loss.avg, train_acc.avg))
    # print('Training Loss: {:.4f}'.format(train_loss.avg))

def validation(valid_loader, model, criterion, epoch_num, nbreClasses, batch_size):
    """
    Args:
        train_loader:
        model:
        criterion:
        optimizer:
        epoch:
    Returns:
    """
    # logging accuracy and loss
    valid_acc = AverageMeter()
    valid_loss = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # Iterate over data.
    for idx, data in enumerate(valid_loader):
        torch.cuda.empty_cache()
        with torch.no_grad():
            # flatten label
            labels_flatten = flatten_labels(data['map_img'])           
            # get the inputs and wrap in Variable
            if torch.cuda.is_available():
                inputs = Variable(data['sat_img'].cuda())
                labels = Variable(labels_flatten.cuda())
                # index = Variable(index.cuda())
            else:
                inputs = Variable(data['sat_img'])
                labels = Variable(labels_flatten)
                # index = Variable(index)
            del labels_flatten
            torch.cuda.empty_cache()
            # forward
            outputs = model(inputs)
            # print(outputs.shape)
            outputs_flatten = flatten_outputs(outputs, nbreClasses)
            
            loss = criterion(outputs_flatten, labels)
            valid_loss.update(loss.item(), inputs.size(0))
            if idx == len(valid_loader) - 1:
                a, segmentation = torch.max(outputs_flatten, dim=1)
                acc = metrics.accuracy(segmentation, labels)
                valid_acc.update(acc, batch_size)
                
    print('Validation Loss: {:.4f} Acc: {:.4f}'.format(valid_loss.avg, valid_acc.avg))
    # print('Validation Loss: {:.4f}'.format(valid_loss.avg))
    return valid_loss.avg
    
#     return {'valid_loss': valid_loss.avg, 'valid_acc': valid_acc.avg}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training execution')
    parser.add_argument('ParamFile', metavar='DIR',
                        help='path to parameters txt')
    args = parser.parse_args()
    parampampam = ReadParameters(args.ParamFile)

    working_folder = parampampam[0]
    batch_size = int(parampampam[1])
    num_epoch = int(parampampam[2])
    lr = float(parampampam[3])
    tailleTuile = int(parampampam[4])
    nbrClasses = int(parampampam[5])
    nbrEchantTrn = int(parampampam[6])
    nbrEchantVal = int(parampampam[7])
    print(parampampam)


    #### parametres ###
    print('Debut:')
    # working_folder = "D:\Processus\image_to_echantillons\img_1"
#     working_folder = "/space/hall0/work/nrcan/geobase/extraction/Deep_learning/pytorch/data_training"
#     batch_size = 96
#     num_epoch = 150
#     start_epoch = 0
#     lr = 0.0005
#     tailleTuile = 128
#     nbrClasses = 4
#     nbrEchantTrn = 38400
#     nbrEchantVal = 19200
    main(working_folder, batch_size, num_epoch, lr, tailleTuile, nbrClasses, nbrEchantTrn, nbrEchantVal)
    print('Fin')

        
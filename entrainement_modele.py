import matplotlib.pyplot as plt
import numpy as np
import os

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import metrics
import unet_pytorch
import CreateDataset
from utils import get_gpu_memory_map
from utils import AverageMeter
from sklearn.metrics import confusion_matrix

import torch
import torch.optim as optim
import time
import argparse
import shutil


def plot_some_results(data, target, img_sufixe, folder):
    """__author__ = 'Fabian Isensee'
    https://github.com/Lasagne/Recips/blob/master/examples/UNet/massachusetts_road_segm.py"""
    d = data
    s = target
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(d.transpose(1,2,0))
    plt.title("input patch")
    plt.subplot(1, 3, 2)
    plt.imshow(s)
    plt.title("ground truth")
    plt.savefig(os.path.join(folder, "result_%03.0f.png"%img_sufixe))
    plt.close()
        
def flatten_labels(annotations):
    return annotations.view(-1)

def flatten_outputs(logits, number_of_classes):
    """Flattens the logits batch except for the logits dimension"""
    
    logits_permuted = logits.permute(0, 2, 3, 1)
    logits_permuted_cont = logits_permuted.contiguous()
    logits_flatten = logits_permuted_cont.view(-1, number_of_classes)
    return logits_flatten

# Create a function to save the model state
def save_checkpoint(state, is_best, filename='./checkpoint.pth.tar'):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def main(TravailFolder, batch_size, num_epochs, start_epoch, learning_rate, TailleTuile, NbClasses, NbEchantillonsTrn, NbEchantillonsVal):
    """
    Args:
        data_path:
        batch_size:
        num_epochs:
    Returns:
    """
    since = time.time()

    # get model
    model = unet_pytorch.UNetSmall(NbClasses)

    if torch.cuda.is_available():
        model = model.cuda()
        

    # set up binary cross entropy
    criterion = nn.CrossEntropyLoss()

    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # starting params
    best_loss = 999

    # get data
    trn_dataset = CreateDataset.SegmentationDataset(os.path.join(TravailFolder, "echantillons_entrainement"), NbEchantillonsTrn, TailleTuile)
    val_dataset = CreateDataset.SegmentationDataset(os.path.join(TravailFolder, "echantillons_validation"), NbEchantillonsVal, TailleTuile)
    # creating loaders
    train_dataloader = DataLoader(trn_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, num_workers=2, shuffle=True)
    
    for epoch in range(start_epoch, num_epochs):
        print()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
    
        torch.cuda.empty_cache()
        # run training and validation
        train_metrics = train(train_dataloader, model, criterion, optimizer, epoch, nbrClasses)

        torch.cuda.empty_cache()
        
        # print("GPU memoire: ", get_gpu_memory_map())
        
        Val_loss = validation(val_dataloader, model, criterion, epoch, nbrClasses)
        
        # print(valid_metricNew)
        
        torch.cuda.empty_cache()
        
        if Val_loss < best_loss:
            print("Val_loss est plus petite que best_loss")
            save_checkpoint({'epoch': epoch, 'arch': 'UNetSmall', 'state_dict': model.state_dict(), 'best_loss': best_loss, 'optimizer': optimizer.state_dict()}, is_best=False)
            # torch.save(model.state_dict(), 'unet_best.pth')
            best_loss = Val_loss

        cur_elapsed = time.time() - since
        print('Current elapsed time {:.0f}m {:.0f}s'.format(cur_elapsed // 60, cur_elapsed % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def train(train_loader, model, criterion, optimizer, epoch_num, nbreClasses):
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
    losses = AverageMeter()
    
    # logging accuracy and loss
    # train_acc = metrics.MetricTracker()
    # train_loss = metrics.MetricTracker()

    # iterate over data
    for idx, data in enumerate(train_loader):
        # print(model.parameters())
        # We need to flatten annotations and logits to apply index of valid annotations.
        # https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/master/pytorch_segmentation_detection/recipes/pascal_voc/segmentation/resnet_18_8s_train.ipynb
        labels_flatten = flatten_labels(data['map_img'])
        # print(labels_flatten.shape)
        
        # get the inputs and wrap in Variable
        if torch.cuda.is_available():
            inputs = Variable(data['sat_img'].cuda())
            labels = Variable(labels_flatten.cuda())
        else:
            inputs = Variable(data['sat_img'])
            labels = Variable(labels_flatten)
        del labels_flatten
        torch.cuda.empty_cache()
        
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        # preds = model(inputs)
        outputs = model(inputs)
        m = nn.LogSoftmax(1)
        outputs = m(outputs)
        # outputs = torch.nn.functional.sigmoid(preds)
        outputs_flatten = flatten_outputs(outputs, nbreClasses)
        
        
        del outputs
        torch.cuda.empty_cache()
        
        loss = criterion(outputs_flatten, labels)
        losses.update(loss.item(), inputs.size(0))
        # perte += loss.data[0]
        del inputs
        # backward
        loss.backward()
        optimizer.step()
    print("perte trn: ", losses.avg)

def validation(valid_loader, model, criterion, epoch_num, nbreClasses):
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
    # valid_acc = metrics.MetricTracker()
    # valid_loss = metrics.MetricTracker()
    valid_losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    overall_confusion_matrix = None
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
            outputs = torch.nn.functional.sigmoid(outputs)
            m = nn.LogSoftmax(1)
            outputs = m(outputs)
            outputs_flatten = flatten_outputs(outputs, nbreClasses)
            loss = criterion(outputs_flatten, labels)
            valid_losses.update(loss.item(), outputs.size(0))

    print('Validation Loss: ', valid_losses.avg)
    # print('Validation Loss: ', valid_loss.avg, ' Acc: ', valid_acc.avg)
    return valid_losses.avg
    
#     return {'valid_loss': valid_loss.avg, 'valid_acc': valid_acc.avg}

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
#     parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
#                         metavar='LR', help='initial learning rate')
#     parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                         help='momentum')
#     parser.add_argument('--print-freq', default=4, type=int, metavar='N',
#                         help='number of time to log per epoch')
#     parser.add_argument('--run', default=0, type=int, metavar='N',
#                         help='number of run (for tensorboard logging)')
#     parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                         help='path to latest checkpoint (default: none)')
#     parser.add_argument('--data-set', default='mass_roads_crop', type=str,
#                         help='mass_roads or mass_buildings or mass_roads_crop')
# 
#     args = parser.parse_args()

    # main(args.data, batch_size=args.batch_size, num_epochs=args.epochs, start_epoch=args.start_epoch, learning_rate=args.lr, momentum=args.momentum, print_freq=args.print_freq, run=args.run, resume=args.resume, data_set=args.data_set)


    #### parametres ###
    print('Debut:')
    # TravailFolder = "D:\Processus\image_to_echantillons\img_1"
    TravailFolder = "/space/hall0/work/nrcan/geobase/extraction/Deep_learning/pytorch/"
    batch_size = 8
    num_epoch = 50
    start_epoch = 0
    lr = 0.001
    tailleTuile = 512
    nbrClasses = 4
    nbrEchantTrn = 1000
    nbrEchantVal = 200
    main(TravailFolder, batch_size, num_epoch, start_epoch, lr, tailleTuile, nbrClasses, nbrEchantTrn, nbrEchantVal)
    print('Fin')

        
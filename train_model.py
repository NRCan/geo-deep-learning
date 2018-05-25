# import matplotlib.pyplot as plt
# import numpy as np
import argparse
import os
import shutil
import time

from ruamel_yaml import YAML
from torch import nn
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import CreateDataset
from metrics import AverageMeter
import metrics
import torch.optim as optim
import unet_pytorch

# from torchvision import transforms
# import unet
# from utils import get_gpu_memory_map
# from sklearn.metrics import confusion_matrix
def flatten_labels(annotations):
    flatten = annotations.view(-1)
    return flatten

def flatten_outputs(predictions, number_of_classes):
    """Flattens the predictions batch except for the predictions dimension"""
    logits_permuted = predictions.permute(0, 2, 3, 1)
    logits_permuted_cont = logits_permuted.contiguous()
    outputs_flatten = logits_permuted_cont.view(-1, number_of_classes)
    return outputs_flatten


def save_checkpoint(state, filename):
    """
    Function to save the model state
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    :param state:
    :param filename:
    """
    torch.save(state, filename)

def load_from_checkpoint(filename, model, optimizer):
    """function to load weights from a checkpoint"""
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}'".format(filename))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

def main(data_path, output_path, sample_size, num_trn_samples, num_val_samples, pretrained, batch_size, num_epochs, learning_rate, weight_decay, step_size, gamma, num_classes, weight_classes):
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
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight_classes)).cuda()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # lr decay
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # loss initialisation
    best_loss = 999

    # optionally resume from a checkpoint
    if pretrained:
        load_from_checkpoint(pretrained, model, optimizer)

    # get data
    trn_dataset = CreateDataset.SegmentationDataset(os.path.join(data_path, "trn/samples"), num_trn_samples, sample_size)
    val_dataset = CreateDataset.SegmentationDataset(os.path.join(data_path, "val/samples"), num_val_samples, sample_size)
    # creating loaders
    train_dataloader = DataLoader(trn_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    for epoch in range(0, num_epochs):
        print()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        # run training and validation
        train_loss = train(train_dataloader, model, criterion, optimizer, lr_scheduler, epoch, num_classes, batch_size)
        Val_loss = validation(val_dataloader, model, criterion, epoch, num_classes, batch_size)

        if Val_loss < best_loss:
            print("save checkpoint")
            filename=os.path.join(output_path, 'checkpoint.pth.tar')
            save_checkpoint({'epoch': epoch, 'arch': 'UNetSmall', 'state_dict': model.state_dict(), 'best_loss': best_loss, 'optimizer': optimizer.state_dict()}, filename)
            best_loss = Val_loss

        cur_elapsed = time.time() - since
        print('Current elapsed time {:.0f}m {:.0f}s'.format(cur_elapsed // 60, cur_elapsed % 60))

    filename=os.path.join(output_path, 'last_epoch.pth.tar')
    save_checkpoint({'epoch': epoch, 'arch': 'UNetSmall', 'state_dict': model.state_dict(), 'best_loss': best_loss, 'optimizer': optimizer.state_dict()}, filename)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def train(train_loader, model, criterion, optimizer, scheduler, epoch_num, num_classes, batch_size):
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

    scheduler.step()

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

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        outputs_flatten = flatten_outputs(outputs, num_classes)

        del outputs

        loss = criterion(outputs_flatten, labels)

        train_loss.update(loss.item(), inputs.size(0))

        del inputs

        # backward
        loss.backward()
        optimizer.step()

        # Compute accuracy only on last batch (time consuming)
        if idx == len(train_loader) - 1:
            a, segmentation = torch.max(outputs_flatten, dim=1)
            acc = metrics.accuracy(segmentation, labels)
            train_acc.update(acc, batch_size)

    print('Training Loss: {:.4f} Acc: {:.4f}'.format(train_loss.avg, train_acc.avg))

def validation(valid_loader, model, criterion, epoch_num, num_classes, batch_size):
    """
    Args:
        validation_loader:
        model:
        criterion:
        epoch:
        num_classes:
        batch_size:
    Returns:
    """
    # logging accuracy and loss
    valid_acc = AverageMeter()
    valid_loss = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # Iterate over data.
    for idx, data in enumerate(valid_loader):
        with torch.no_grad():
            # get the inputs and wrap in Variable
            if torch.cuda.is_available():
                inputs = Variable(data['sat_img'].cuda())
                labels = Variable(flatten_labels(data['map_img']).cuda())

            else:
                inputs = Variable(data['sat_img'])
                labels = Variable(flatten_labels(data['map_img']))

            # forward
            outputs = model(inputs)
            outputs_flatten = flatten_outputs(outputs, num_classes)

            loss = criterion(outputs_flatten, labels)
            valid_loss.update(loss.item(), inputs.size(0))

            # Compute accuracy only on last batch (time consuming)
            if idx == len(valid_loader) - 1:
                a, segmentation = torch.max(outputs_flatten, dim=1)
                acc = metrics.accuracy(segmentation, labels)
                valid_acc.update(acc, batch_size)

    print('Validation Loss: {:.4f} Acc: {:.4f}'.format(valid_loss.avg, valid_acc.avg))
    return valid_loss.avg

if __name__ == '__main__':
    print('Start:')
    parser = argparse.ArgumentParser(description='Training execution')
    parser.add_argument('ParamFile', metavar='DIR',
                        help='path to parameters yaml')
    args = parser.parse_args()
    yaml = YAML()
    with open(args.ParamFile, 'r') as yamlfile:
        cfg = yaml.load(yamlfile)

    main(cfg['data_path'], cfg['output_path'], cfg['samples_size'], cfg['num_trn_samples'], cfg['num_val_samples'], cfg['pretrained'], cfg['batch_size'], cfg['num_epochs'], cfg['learning_rate'], cfg['weight_decay'], cfg['step_size'], cfg['gamma'], cfg['num_classes'], cfg['classes_weight'])
    print('End of training')

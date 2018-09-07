import argparse
import os
import time

from torch import nn
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import CreateDataset
import augmentation as aug
from logger import InformationLogger
from metrics import ClassificationReport, iou, CreateMetricsdict
import torch.optim as optim
import unet_pytorch
from utils import ReadParameters, LoadFromCheckpoint
import numpy as np

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
    Function to save the models state
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    :param state:
    :param filename:
    """
    torch.save(state, filename)

def net(net_params):

    if net_params['name'].lower() == 'unetsmall':
        model = unet_pytorch.UNetSmall(net_params['num_classes'], net_params['number_of_bands'], net_params['dropout'], net_params['probability'])

    elif net_params['name'].lower() == 'unet':
        model = unet_pytorch.UNet(net_params['num_classes'], net_params['number_of_bands'], net_params['dropout'], net_params['probability'])

    return model

def main(data_path, output_path, num_trn_samples, num_val_samples, pretrained, batch_size, num_epochs, learning_rate, weight_decay, step_size, gamma, num_classes, weight_classes, number_of_bands, nn_parameters):
    """
    Function to train and validate a models for semantic segmentation.
    Args:
        data_path:
        output_path:
        num_trn_samples:
        pretrained:
        batch_size:
        num_epochs:
        learning_rate:
        weight_decay:
        step_size:
        gamma:
        num_classes:
        weight_classes:
        number_of_bands:
    Returns:
        Files 'checkpoint.pth.tar' and 'last_epoch.pth.tar' containing trained weight
    """
    since = time.time()

    # init loggers
    TrnLog = InformationLogger(output_path, 'trn')
    ValLog = InformationLogger(output_path, 'val')

    # get model
    model = net(nn_parameters)

    if torch.cuda.is_available():
        model = model.cuda()

    # set up cross entropy
    if weight_classes is not False:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight_classes)).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # lr decay
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # loss initialisation
    best_loss = 999

    # optionally resume from a checkpoint
    if pretrained:
        model, optimizer = LoadFromCheckpoint(pretrained, model, optimizer)

    # get data
    trn_dataset = CreateDataset.SegmentationDataset(os.path.join(data_path, "samples"), num_trn_samples, "trn", transform=transforms.Compose([aug.RandomRotationTarget(), aug.HorizontalFlip(), aug.ToTensorTarget()]))
    val_dataset = CreateDataset.SegmentationDataset(os.path.join(data_path, "samples"), num_val_samples, "val", transform=transforms.Compose([aug.ToTensorTarget()]))
    # creating loaders
    trn_dataloader = DataLoader(trn_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    for epoch in range(0, num_epochs):
        print()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        # run training and validation
        Trn_report = train(trn_dataloader, model, criterion, optimizer, lr_scheduler, epoch, num_classes, batch_size)
        TrnLog.AddValues(Trn_report, epoch)

        Val_report = validation(val_dataloader, model, criterion, epoch, num_classes, batch_size)
        val_loss = Val_report['loss'].avg
        ValLog.AddValues(Val_report, epoch)

        if val_loss < best_loss:
            print("save checkpoint")
            filename=os.path.join(output_path, 'checkpoint.pth.tar')
            best_loss = val_loss
            save_checkpoint({'epoch': epoch, 'arch': 'UNetSmall', 'state_dict': model.state_dict(), 'best_loss': best_loss, 'optimizer': optimizer.state_dict()}, filename)

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
        models:
        criterion:
        optimizer:
        epoch:
    Returns:
    """
    model.train()

    scheduler.step()

    train_metrics = CreateMetricsdict(num_classes)

    # iterate over data
    for idx, data in enumerate(train_loader):
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
        del inputs

        loss = criterion(outputs_flatten, labels)
        train_metrics['loss'].update(loss.item(), batch_size)

        # backward
        loss.backward()
        optimizer.step()

        # Compute accuracy and iou every 2 batches and average values. time consuming.
        if idx % 2 == 0:
            a, segmentation = torch.max(outputs_flatten, dim=1)

            train_metrics = ClassificationReport(segmentation, labels, num_classes, batch_size, train_metrics)
            train_metrics = iou(segmentation, labels, batch_size, train_metrics)

    print('Training Loss: {:.4f}'.format(train_metrics['loss'].avg))
    print('Training iou: {:.4f}'.format(train_metrics['iou'].avg))
    print('Training precision: {:.4f}'.format(train_metrics['precision'].avg))
    print('Training recall: {:.4f}'.format(train_metrics['recall'].avg))
    print('Training f1-score: {:.4f}'.format(train_metrics['fscore'].avg))

    return train_metrics

def validation(valid_loader, model, criterion, epoch_num, num_classes, batch_size):
    """
    Args:
        validation_loader:
        models:
        criterion:
        epoch:
        num_classes:
        batch_size:
    Returns:
    """

    valid_metrics = CreateMetricsdict(num_classes)

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
            valid_metrics['loss'].update(loss.item(), batch_size)

            # Compute metrics every 2 batches. Time consuming.
            if idx % 2 == 0:
                a, segmentation = torch.max(outputs_flatten, dim=1)

                valid_metrics = ClassificationReport(segmentation, labels, num_classes , batch_size, valid_metrics)
                valid_metrics = iou(segmentation, labels, batch_size, valid_metrics)

    print('Validation Loss: {:.4f}'.format(valid_metrics['loss'].avg))
    print('Validation iou: {:.4f}'.format(valid_metrics['iou'].avg))
    print('Validation precision: {:.4f}'.format(valid_metrics['precision'].avg))
    print('Validation recall: {:.4f}'.format(valid_metrics['recall'].avg))
    print('Validation f1-score: {:.4f}'.format(valid_metrics['fscore'].avg))

    return valid_metrics

if __name__ == '__main__':
    print('Start:')
    parser = argparse.ArgumentParser(description='Training execution')
    parser.add_argument('param_file', metavar='DIR',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    params = ReadParameters(args.param_file)
    nn_parameters = {'name': 'unetsmall',
                'num_classes': params['global']['num_classes'],
                'number_of_bands': params['global']['number_of_bands'],
                'dropout': params['training']['dropout'],
                'probability': params['training']['probability']}

    main(params['global']['data_path'],
         params['training']['output_path'],
         params['training']['num_trn_samples'],
         params['training']['num_val_samples'],
         params['training']['pretrained'],
         params['training']['batch_size'],
         params['training']['num_epochs'],
         params['training']['learning_rate'],
         params['training']['weight_decay'],
         params['training']['step_size'],
         params['training']['gamma'],
         params['global']['num_classes'],
         params['training']['class_weights'],
         params['global']['number_of_bands'],
         nn_parameters)
    print('End of training')

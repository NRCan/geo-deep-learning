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
from metrics import AverageMeter, ClassificationReport
import torch.optim as optim
import unet_pytorch
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

def main(data_path, output_path, num_trn_samples, num_val_samples, pretrained, batch_size, num_epochs, learning_rate, weight_decay, step_size, gamma, num_classes, weight_classes):
    """
    Function to train and validate a model for semantic segmentation.
    Args:
        working_folder
        batch_size:
        num_epochs:
        learning_rate:
        num_classes:
        num_samples_trn:
        num_samples_val:
    Returns:
        File 'model_best.pth.tar' containing trained weights
    """
    since = time.time()

    # init loggers
    TrnLog = InformationLogger(output_path, 'trn')
    ValLog = InformationLogger(output_path, 'val')

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
    trn_dataset = CreateDataset.SegmentationDataset(os.path.join(data_path, "trn/samples"), num_trn_samples, "trn", transform=transforms.Compose([aug.RandomRotationTarget(), aug.HorizontalFlip(), aug.ToTensorTarget()]))
    val_dataset = CreateDataset.SegmentationDataset(os.path.join(data_path, "val/samples"), num_val_samples, "val", transform=transforms.Compose([aug.ToTensorTarget()]))
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
        ValLog.AddValues(Val_report, epoch)

        if Val_report['loss'] < best_loss:
            print("save checkpoint")
            filename=os.path.join(output_path, 'checkpoint.pth.tar')
            best_loss = Val_report['loss']
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
        model:
        criterion:
        optimizer:
        epoch:
    Returns:
    """
    model.train()

    scheduler.step()

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
        # TODO: might be worth it to calculate it every n batches and then average values...
        if idx == len(train_loader) - 1:
            a, segmentation = torch.max(outputs_flatten, dim=1)
            trn_report = ClassificationReport(segmentation, labels, nbClasses=num_classes)
            print("Training Precision", trn_report['prfAvg'][0], "Recall", trn_report['prfAvg'][1], "f-score", trn_report['prfAvg'][2])

    print('Training Loss: {:.4f}'.format(train_loss.avg))
    trn_report['loss'] = train_loss.avg
    return trn_report

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
            # TODO: might be worth it to calculate it every n batches and then average values...
            if idx == len(valid_loader) - 1:
                a, segmentation = torch.max(outputs_flatten, dim=1)
                val_report = ClassificationReport(segmentation, labels, nbClasses=num_classes)
                print("Validation Precision", val_report['prfAvg'][0], "Recall", val_report['prfAvg'][1], "f-score", val_report['prfAvg'][2])

    print('Validation Loss: {:.4f}'.format(valid_loss.avg))
    val_report['loss'] = valid_loss.avg
    return val_report

if __name__ == '__main__':
    print('Start:')
    parser = argparse.ArgumentParser(description='Training execution')
    parser.add_argument('param_file', metavar='DIR',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    params = ReadParameters(args.param_file)
    
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
         params['training']['class_weights'])
    print('End of training')

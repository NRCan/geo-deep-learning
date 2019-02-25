import torch
# import torch should be first. Unclear issue, mentioned here: https://github.com/pytorch/pytorch/issues/2083
import argparse
import os
import csv
import time
import h5py
import datetime
import warnings
import torchvision
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

import CreateDataset
import augmentation as aug
from logger import InformationLogger, save_logs_to_bucket
from metrics import report_classification, iou, create_metrics_dict
from models.model_choice import net
from utils import read_parameters, load_from_checkpoint, list_s3_subfolders

try:
    import boto3
except ModuleNotFoundError:
    warnings.warn('The boto3 library counldn\'t be imported. Ignore if not using AWS s3 buckets', ImportWarning)
    pass


def verify_weights(num_classes, weights):
    """Verifies that the number of weights equals the number of classes if any are given
    Args:
        num_classes: number of classes defined in the configuration file
        weights: weights defined in the configuration file
    """
    if weights:
        if num_classes != len(weights):
            raise ValueError('The number of class weights in the configuration file is different than the number of '
                             'classes')


def verify_sample_count(num_trn_samples, num_val_samples, hdf5_folder, bucket=None):
    """Verifies that the number of training and validation samples defined in the configuration file is less or equal to
     the number of training and validation samples in the hdf5 file
     Args:
         num_trn_samples: number of training samples defined in the configuration file
         num_val_samples: number of validation samples defined in the configuration file
         hdf5_folder: data path in which the samples are saved
         bucket: aws s3 bucket where data is stored. '' if not on AWS
     """
    if not bucket:
        with h5py.File(os.path.join(hdf5_folder + '/samples', "trn_samples.hdf5"), 'r') as f:
            train_samples = len(f['map_img'])
        if num_trn_samples > train_samples:
            raise IndexError('The number of training samples in the configuration file exceeds the number of '
                             'samples in the hdf5 training dataset.')
        with h5py.File(os.path.join(hdf5_folder + '/samples', "val_samples.hdf5"), 'r') as f:
            valid_samples = len(f['map_img'])
        if num_val_samples > valid_samples:
            raise IndexError('The number of validation samples in the configuration file exceeds the number of '
                             'samples in the hdf5 validation dataset.')
    else:
        with h5py.File('samples/trn_samples.hdf5', 'r') as f:
            train_samples = len(f['map_img'])
        if num_trn_samples > train_samples:
            raise IndexError('The number of training samples in the configuration file exceeds the number of '
                             'samples in the hdf5 training dataset.')
        with h5py.File('samples/val_samples.hdf5', 'r') as f:
            valid_samples = len(f['map_img'])
        if num_val_samples > valid_samples:
            raise IndexError('The number of training samples in the configuration file exceeds the number of '
                             'samples in the hdf5 validation dataset.')


def flatten_labels(annotations):
    """Flatten labels"""
    flatten = annotations.view(-1)
    return flatten


def flatten_outputs(predictions, number_of_classes):
    """Flatten the prediction batch except the prediction dimensions"""
    logits_permuted = predictions.permute(0, 2, 3, 1)
    logits_permuted_cont = logits_permuted.contiguous()
    outputs_flatten = logits_permuted_cont.view(-1, number_of_classes)
    return outputs_flatten


def save_checkpoint(state, filename):
    """Save the model's state"""
    torch.save(state, filename)


def loader(path):
    img = Image.open(path)
    return img


def get_s3_classification_images(dataset, bucket, bucket_name, data_path, output_path, num_classes):
    classes = list_s3_subfolders(bucket_name, os.path.join(data_path, dataset))
    classes.sort()
    assert num_classes == len(classes), "The configuration file specified %d classes, but only %d class folders were " \
                                        "found in %s." % (num_classes, len(classes), os.path.join(data_path, dataset))
    with open(os.path.join(output_path, 'classes.csv'), 'wt') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(classes)

    path = os.path.join('Images', dataset)
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    for c in classes:
        classpath = os.path.join(path, c)
        try:
            os.mkdir(classpath)
        except FileExistsError:
            pass
        for f in bucket.objects.filter(Prefix=os.path.join(data_path, dataset, c)):
            if f.key != data_path + '/':
                bucket.download_file(f.key, os.path.join(classpath, f.key.split('/')[-1]))


def get_local_classes(num_classes, data_path, output_path):
    # Get classes locally and write to csv in output_path
    classes = next(os.walk(os.path.join(data_path, 'trn')))[1]
    classes.sort()
    assert num_classes == len(classes), "The configuration file specified %d classes, but only %d class folders were " \
                                        "found in %s." % (num_classes, len(classes), os.path.join(data_path, 'trn'))
    with open(os.path.join(output_path, 'classes.csv'), 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(classes)


def main(bucket_name, data_path, output_path, num_trn_samples, num_val_samples, pretrained, batch_size, num_epochs,
         learning_rate, weight_decay, step_size, gamma, num_classes, class_weights, model, classifier, model_name):
    """Function to train and validate a models for semantic segmentation.
    Args:
        bucket_name: bucket in which data is stored if using AWS S3
        data_path: full file path of the folder containing h5py files
        output_path: full file path in which the model will be saved
        num_trn_samples: number of training samples
        num_val_samples: number of validation samples
        pretrained: booleam indicating if the model is pretrained
        batch_size: number of samples to process simultaneously
        num_epochs: number of epochs
        learning_rate: learning rate
        weight_decay: weight decay
        step_size: step size
        gamma: multiplicative factor of learning rate decay
        num_classes: number of classes
        class_weights: weights to apply to each class. A value > 1.0 will apply more weights to the learning of the class
        model: CNN model (tensor)
        classifier: True if doing image classification, False if doing semantic segmentation.
        model_name: name of the model used for training.
    Returns:
        Files 'checkpoint.pth.tar' and 'last_epoch.pth.tar' containing trained weight
    """
    if bucket_name:
        if output_path is None:
            bucket_output_path = None
        else:
            bucket_output_path = output_path
        output_path = 'output_path'
        try:
            os.mkdir(output_path)
        except FileExistsError:
            pass
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        if classifier:
            for i in ['trn', 'val']:
                get_s3_classification_images(i, bucket, bucket_name, data_path, output_path, num_classes)
                class_file = os.path.join(output_path, 'classes.csv')
                if bucket_output_path:
                    bucket.upload_file(class_file, os.path.join(bucket_output_path, 'classes.csv'))
                else:
                    bucket.upload_file(class_file, 'classes.csv')
            data_path = 'Images'
        else:
            if data_path:
                bucket.download_file(os.path.join(data_path, 'samples/trn_samples.hdf5'),
                                     'samples/trn_samples.hdf5')
                bucket.download_file(os.path.join(data_path, 'samples/val_samples.hdf5'),
                                     'samples/val_samples.hdf5')
            else:
                bucket.download_file('samples/trn_samples.hdf5', 'samples/trn_samples.hdf5')
                bucket.download_file('samples/val_samples.hdf5', 'samples/val_samples.hdf5')
            verify_sample_count(num_trn_samples, num_val_samples, data_path, bucket_name)
    elif classifier:
        get_local_classes(num_classes, data_path, output_path)
    else:
        verify_sample_count(num_trn_samples, num_val_samples, data_path, bucket_name)
    verify_weights(num_classes, class_weights)

    since = time.time()
    best_loss = 999

    trn_log = InformationLogger(output_path, 'trn')
    val_log = InformationLogger(output_path, 'val')

    if torch.cuda.is_available():
        model = model.cuda()
        if class_weights:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights)).cuda()
        else:
            criterion = nn.CrossEntropyLoss().cuda()
    else:
        if class_weights:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
        else:
            criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # learning rate decay

    if pretrained != '':
        model, optimizer = load_from_checkpoint(pretrained, model, optimizer)

    if classifier:
        trn_dataset = torchvision.datasets.ImageFolder(os.path.join(data_path, "trn"),
                                                       transform=transforms.Compose(
                                                           [transforms.RandomRotation((0, 275)),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.Resize(299), transforms.ToTensor()]),
                                                       loader=loader)
        val_dataset = torchvision.datasets.ImageFolder(os.path.join(data_path, "val"),
                                                       transform=transforms.Compose(
                                                           [transforms.Resize(299), transforms.ToTensor()]),
                                                       loader=loader)
    else:
        if not bucket_name:
            trn_dataset = CreateDataset.SegmentationDataset(os.path.join(data_path, "samples"), num_trn_samples, "trn",
                                                            transform=transforms.Compose([aug.RandomRotationTarget(),
                                                                                          aug.HorizontalFlip(),
                                                                                          aug.ToTensorTarget()]))
            val_dataset = CreateDataset.SegmentationDataset(os.path.join(data_path, "samples"), num_val_samples, "val",
                                                            transform=transforms.Compose([aug.ToTensorTarget()]))
        else:
            trn_dataset = CreateDataset.SegmentationDataset('samples', num_trn_samples, "trn",
                                                            transform=transforms.Compose([aug.RandomRotationTarget(),
                                                                                          aug.HorizontalFlip(),
                                                                                          aug.ToTensorTarget()]))
            val_dataset = CreateDataset.SegmentationDataset("samples", num_val_samples, "val",
                                                            transform=transforms.Compose([aug.ToTensorTarget()]))

    # Shuffle must be set to True.
    trn_dataloader = DataLoader(trn_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    now = datetime.datetime.now().strftime("%Y-%m-%d %I:%M ")
    for epoch in range(0, num_epochs):
        print()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        trn_report = train(trn_dataloader, model, criterion, optimizer, lr_scheduler, num_classes, batch_size,
                           classifier)
        trn_log.add_values(trn_report, epoch)

        val_report = validation(val_dataloader, model, criterion, num_classes, batch_size, classifier)
        val_loss = val_report['loss'].avg
        val_log.add_values(val_report, epoch)

        if val_loss < best_loss:
            print("save checkpoint")
            filename = os.path.join(output_path, 'checkpoint.pth.tar')
            best_loss = val_loss
            save_checkpoint({'epoch': epoch, 'arch': model_name, 'model': model.state_dict(), 'best_loss':
                best_loss, 'optimizer': optimizer.state_dict()}, filename)

            if bucket_name:
                if bucket_output_path:
                    bucket_filename = os.path.join(bucket_output_path, 'checkpoint.pth.tar')
                else:
                    bucket_filename = 'checkpoint.pth.tar'
                bucket.upload_file(filename, bucket_filename)

        if bucket_name:
            save_logs_to_bucket(bucket, bucket_output_path, output_path, now)

        cur_elapsed = time.time() - since
        print('Current elapsed time {:.0f}m {:.0f}s'.format(cur_elapsed // 60, cur_elapsed % 60))

    filename = os.path.join(output_path, 'last_epoch.pth.tar')
    save_checkpoint({'epoch': epoch, 'arch': model_name, 'model': model.state_dict(), 'best_loss': best_loss,
                     'optimizer': optimizer.state_dict()}, filename)

    if bucket_name:
        if bucket_output_path:
            bucket_filename = os.path.join(bucket_output_path, 'last_epoch.pth.tar')
            bucket.upload_file("output.txt", os.path.join(bucket_output_path, f"Logs/{now}_output.txt"))
        else:
            bucket_filename = 'last_epoch.pth.tar'
            bucket.upload_file("output.txt", f"Logs/{now}_output.txt")
        bucket.upload_file(filename, bucket_filename)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def train(train_loader, model, criterion, optimizer, scheduler, num_classes, batch_size, classifier):
    """ Train the model and return the metrics of the training phase.
    Args:
        train_loader: training data loader
        model: model to train
        criterion: loss criterion
        optimizer: optimizer to use
        scheduler: learning rate scheduler
        num_classes: number of classes
        batch_size: number of samples to process simultaneously
        classifier: True if doing a classification task, False if doing semantic segmentation
    """
    model.train()
    scheduler.step()
    train_metrics = create_metrics_dict(num_classes)

    for index, data in enumerate(train_loader):
        if classifier:
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs_flatten = outputs
        else:
            if torch.cuda.is_available():
                inputs = data['sat_img'].cuda()
                labels = flatten_labels(data['map_img']).cuda()
            else:
                inputs = data['sat_img']
                labels = flatten_labels(data['map_img'])
            # forward
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs_flatten = flatten_outputs(outputs, num_classes)

        del outputs
        del inputs
        loss = criterion(outputs_flatten, labels)
        train_metrics['loss'].update(loss.item(), batch_size)

        loss.backward()
        optimizer.step()

        # Compute accuracy and iou every 2 batches and average values. Time consuming.
        if index % 2 == 0:
            a, segmentation = torch.max(outputs_flatten, dim=1)

            train_metrics = report_classification(segmentation, labels, batch_size, train_metrics)
            train_metrics = iou(segmentation, labels, batch_size, train_metrics)

    print('Training Loss: {:.4f}'.format(train_metrics['loss'].avg))
    print('Training iou: {:.4f}'.format(train_metrics['iou'].avg))
    print('Training precision: {:.4f}'.format(train_metrics['precision'].avg))
    print('Training recall: {:.4f}'.format(train_metrics['recall'].avg))
    print('Training f1-score: {:.4f}'.format(train_metrics['fscore'].avg))

    return train_metrics


def validation(valid_loader, model, criterion, num_classes, batch_size, classifier):
    """Args:
        valid_loader: validation data loader
        model: model to validate
        criterion: loss criterion
        num_classes: number of classes
        batch_size: number of samples to process simultaneously
        classifier: True if doing a classification task, False if doing semantic segmentation
    """

    valid_metrics = create_metrics_dict(num_classes)
    model.eval()

    for index, data in enumerate(valid_loader):
        with torch.no_grad():
            if classifier:
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = model(inputs)
                outputs_flatten = outputs
            else:
                if torch.cuda.is_available():
                    inputs = data['sat_img'].cuda()
                    labels = flatten_labels(data['map_img']).cuda()
                else:
                    inputs = data['sat_img']
                    labels = flatten_labels(data['map_img'])

                outputs = model(inputs)
                outputs_flatten = flatten_outputs(outputs, num_classes)

            loss = criterion(outputs_flatten, labels)
            valid_metrics['loss'].update(loss.item(), batch_size)

            # Compute metrics every 2 batches. Time consuming.
            if index % 2 == 0:
                a, segmentation = torch.max(outputs_flatten, dim=1)
                valid_metrics = report_classification(segmentation, labels, batch_size, valid_metrics)
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
    params = read_parameters(args.param_file)
    cnn_model, state_dict_path, model_name = net(params)

    main(params['global']['bucket_name'],
         params['global']['data_path'],
         params['training']['output_path'],
         params['training']['num_trn_samples'],
         params['training']['num_val_samples'],
         state_dict_path,
         params['training']['batch_size'],
         params['training']['num_epochs'],
         params['training']['learning_rate'],
         params['training']['weight_decay'],
         params['training']['step_size'],
         params['training']['gamma'],
         params['global']['num_classes'],
         params['training']['class_weights'],
         cnn_model,
         params['global']['classify'],
         model_name)
    print('End of training')

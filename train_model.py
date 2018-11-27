import argparse
import os
import time
import h5py
import torch
import datetime
import warnings
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import CreateDataset
import augmentation as aug
from logger import InformationLogger
from metrics import report_classification, iou, create_metrics_dict
from models.model_choice import net
from utils import read_parameters, load_from_checkpoint

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


def main(bucket_name, data_path, output_path, num_trn_samples, num_val_samples, pretrained, batch_size, num_epochs,
         learning_rate,
         weight_decay, step_size, gamma, num_classes, class_weights, model):
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
    Returns:
        Files 'checkpoint.pth.tar' and 'last_epoch.pth.tar' containing trained weight
    """

    if bucket_name:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        if data_path:
            bucket.download_file(os.path.join(data_path, 'samples/trn_samples.hdf5'),
                                 'samples/trn_samples.hdf5')
            bucket.download_file(os.path.join(data_path, 'samples/val_samples.hdf5'),
                                 'samples/val_samples.hdf5')
        else:
            bucket.download_file('samples/trn_samples.hdf5', 'samples/trn_samples.hdf5')
            bucket.download_file('samples/val_samples.hdf5', 'samples/val_samples.hdf5')
    verify_weights(num_classes, class_weights)
    verify_sample_count(num_trn_samples, num_val_samples, data_path, bucket_name)

    since = time.time()
    best_loss = 999

    if bucket_name:
        os.mkdir(output_path)
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

    trn_dataloader = DataLoader(trn_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    for epoch in range(0, num_epochs):
        print()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        trn_report = train(trn_dataloader, model, criterion, optimizer, lr_scheduler, num_classes, batch_size)
        trn_log.add_values(trn_report, epoch)

        val_report = validation(val_dataloader, model, criterion, num_classes, batch_size)
        val_loss = val_report['loss'].avg
        val_log.add_values(val_report, epoch)

        if val_loss < best_loss:
            print("save checkpoint")
            filename = os.path.join(output_path, 'checkpoint.pth.tar')
            best_loss = val_loss
            save_checkpoint({'epoch': epoch, 'arch': 'UNetSmall', 'model': model.state_dict(), 'best_loss':
                best_loss, 'optimizer': optimizer.state_dict()}, filename)
            check = open(filename, 'rb')
            if bucket_name:
                bucket.put_object(Key=filename, Body=check)
                os.remove(os.path.join(output_path, 'checkpoint.pth.tar'))

        cur_elapsed = time.time() - since
        print('Current elapsed time {:.0f}m {:.0f}s'.format(cur_elapsed // 60, cur_elapsed % 60))

    filename = os.path.join(output_path, 'last_epoch.pth.tar')
    save_checkpoint({'epoch': epoch, 'arch': 'UNetSmall', 'model': model.state_dict(), 'best_loss': best_loss,
                     'optimizer': optimizer.state_dict()}, filename)
    if bucket_name:
        trained_model = open(filename, 'rb')
        bucket.put_object(Key=filename, Body=trained_model)
        os.remove(filename)
        now = datetime.datetime.now().strftime("%Y-%m-%d %I:%M ")
        if output_path:
            try:
                bucket.put_object(Key=output_path + '/', Body='')
                bucket.put_object(Key=os.path.join(output_path, "Logs/"), Body='')
            except ClientError:
                pass
            logs = open(os.path.join(output_path, "trn_classes_score.log"), 'rb')
            bucket.put_object(Key=os.path.join(output_path, "Logs/" + now + "trn_classes_score.log"), Body=logs)
            logs = open(os.path.join(output_path, "val_classes_score.log"), 'rb')
            bucket.put_object(Key=os.path.join(output_path, "Logs/" + now + "val_classes_score.log"), Body=logs)
            logs = open(os.path.join(output_path, "trn_averaged_score.log"), 'rb')
            bucket.put_object(Key=os.path.join(output_path, "Logs/" + now + "trn_averaged_score.log"), Body=logs)
            logs = open(os.path.join(output_path, "val_averaged_score.log"), 'rb')
            bucket.put_object(Key=os.path.join(output_path, "Logs/" + now + "val_averaged_score.log"), Body=logs)
            logs = open(os.path.join(output_path, "trn_losses_values.log"), 'rb')
            bucket.put_object(Key=os.path.join(output_path, "Logs/" + now + "trn_losses_values.log"), Body=logs)
            logs = open(os.path.join(output_path, "val_losses_values.log"), 'rb')
            bucket.put_object(Key=os.path.join(output_path, "Logs/" + now + "val_losses_values.log"), Body=logs)
            logs = open(os.path.join(output_path, "trn_iou.log"), 'rb')
            bucket.put_object(Key=os.path.join(output_path, "Logs/" + now + "trn_iou.log"), Body=logs)
            logs = open(os.path.join(output_path, "val_iou.log"), 'rb')
            bucket.put_object(Key=os.path.join(output_path, "Logs/" + now + "val_iou.log"), Body=logs)
            os.remove(os.path.join(output_path, "trn_iou.log"))
            os.remove(os.path.join(output_path, "val_iou.log"))
            os.remove(os.path.join(output_path, "val_losses_values.log"))
            os.remove(os.path.join(output_path, "trn_losses_values.log"))
            os.remove(os.path.join(output_path, "val_averaged_score.log"))
            os.remove(os.path.join(output_path, "trn_averaged_score.log"))
            os.remove(os.path.join(output_path, "val_classes_score.log"))
            os.remove(os.path.join(output_path, "trn_classes_score.log"))
        else:
            try:
                bucket.put_object(Key="Logs/", Body='')
            except ClientError:
                pass
            logs = open("trn_classes_score.log", 'rb')
            bucket.put_object(Key="Logs/" + now + "trn_classes_score.log", Body=logs)
            logs = open("val_classes_score.log", 'rb')
            bucket.put_object(Key="Logs/" + now + "val_classes_score.log", Body=logs)
            logs = open("trn_averaged_score.log", 'rb')
            bucket.put_object(Key="Logs/" + now + "trn_averaged_score.log", Body=logs)
            logs = open("val_averaged_score.log", 'rb')
            bucket.put_object(Key="Logs/" + now + "val_averaged_score.log", Body=logs)
            logs = open("trn_losses_values.log", 'rb')
            bucket.put_object(Key="Logs/" + now + "trn_losses_values.log", Body=logs)
            logs = open("val_losses_values.log", 'rb')
            bucket.put_object(Key="Logs/" + now + "val_losses_values.log", Body=logs)
            logs = open("trn_iou.log", 'rb')
            bucket.put_object(Key="Logs/" + now + "trn_iou.log", Body=logs)
            os.remove("trn_iou.log")
            os.remove("val_losses_values.log")
            os.remove("trn_losses_values.log")
            os.remove("val_averaged_score.log")
            os.remove("trn_averaged_score.log")
            os.remove("val_classes_score.log")
            os.remove("trn_classes_score.log")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def train(train_loader, model, criterion, optimizer, scheduler, num_classes, batch_size):
    """ Train the model and return the metrics of the training phase.
    Args:
        train_loader: training data loader
        model: model to train
        criterion: loss criterion
        optimizer: optimizer to use
        scheduler: learning rate scheduler
        num_classes: number of classes
        batch_size: number of samples to process simultaneously
    """
    model.train()
    scheduler.step()
    train_metrics = create_metrics_dict(num_classes)

    for index, data in enumerate(train_loader):

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


def validation(valid_loader, model, criterion, num_classes, batch_size):
    """Args:
        valid_loader: validation data loader
        model: model to validate
        criterion: loss criterion
        num_classes: number of classes
        batch_size: number of samples to process simultaneously
    Returns:
    """

    valid_metrics = create_metrics_dict(num_classes)
    model.eval()

    for index, data in enumerate(valid_loader):
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = data['sat_img'].cuda()
                labels = flatten_labels(data['map_img']).cuda()
            else:
                inputs = data['sat_img']
                labels = flatten_labels(data['map_img'])

            # forward
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
    cnn_model, state_dict_path = net(params)

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
         cnn_model)
    print('End of training')

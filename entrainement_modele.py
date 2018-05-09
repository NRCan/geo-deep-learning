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
    plt.imshow(s[0])
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

def get_valid_annotations_index(flatten_annotations, mask_out_value=255):
    return torch.squeeze( torch.nonzero((flatten_annotations != mask_out_value )), 1)

def main(TravailFolder, batch_size, num_epochs, start_epoch, learning_rate, momentum, resume, TailleTuile, NbClasses, NbEchantillonsTrn, NbEchantillonsVal):
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
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # decay LR
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # starting params
    # best_loss = 999
    best_validation_score = 0

    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)

            if checkpoint['epoch'] > start_epoch:
                start_epoch = checkpoint['epoch']

            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # get data
    trn_dataset = CreateDataset.SegmentationDataset(os.path.join(TravailFolder, "echantillons_entrainement"), NbEchantillonsTrn, TailleTuile)
    val_dataset = CreateDataset.SegmentationDataset(os.path.join(TravailFolder, "echantillons_validation"), NbEchantillonsVal, TailleTuile)

    # creating loaders
    train_dataloader = DataLoader(trn_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, num_workers=1, shuffle=False)

    # loggers
    # train_logger = logger.Logger('../logs/run_{}/training'.format(str(run)), print_freq)
    # val_logger = logger.Logger('../logs/run_{}/validation'.format(str(run)), print_freq)
    train_logger = "toto"
    val_logger = "boubou"
    
    for epoch in range(start_epoch, num_epochs):
        print()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # step the learning rate scheduler
        # lr_scheduler.step()
        lr_scheduler = "boubou"
        

        torch.cuda.empty_cache()
        # run training and validation
        train_metrics = train(train_dataloader, model, criterion, optimizer, lr_scheduler, train_logger, epoch, nbrClasses)

        torch.cuda.empty_cache()
        
        # print("GPU memoire: ", get_gpu_memory_map())
        
        current_validation_score = validation(val_dataloader, model, criterion, val_logger, epoch, nbrClasses)
        
        # print(valid_metricNew)
        
        torch.cuda.empty_cache()


#         # store best loss and save a model checkpoint
#         is_best = valid_metrics['valid_loss'] < best_loss
#         best_loss = min(valid_metrics['valid_loss'], best_loss)
#         save_checkpoint({
#             'epoch': epoch,
#             'arch': 'UNetSmall',
#             'state_dict': model.state_dict(),
#             'best_loss': best_loss,
#             'optimizer': optimizer.state_dict()
#         }, is_best, 'checkpt.pth.tar')
        # Save the model if it has a better MIoU score.
        
        if current_validation_score > best_validation_score:
    
            torch.save(model.state_dict(), 'unet_best.pth')
            best_validation_score = current_validation_score

        cur_elapsed = time.time() - since
        print('Current elapsed time {:.0f}m {:.0f}s'.format(cur_elapsed // 60, cur_elapsed % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def train(train_loader, model, criterion, optimizer, scheduler, logger, epoch_num, nbreClasses):
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
    train_acc = metrics.MetricTracker()
    train_loss = metrics.MetricTracker()

#    log_iter = len(train_loader)//logger.print_freq

    # scheduler.step()

    # iterate over data
    # for idx, data in enumerate(tqdm(train_loader, desc="training")):
    for idx, data in enumerate(train_loader):
        
        
        # We need to flatten annotations and logits to apply index of valid
        # annotations. All of this is because pytorch doesn't have tf.gather_nd()
        # https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/master/pytorch_segmentation_detection/recipes/pascal_voc/segmentation/resnet_18_8s_train.ipynb
        
        # flatten label
        labels_flatten = flatten_labels(data['map_img'])
        # index = get_valid_annotations_index(labels_flatten, mask_out_value=255)
        # print(index)
        # labels_flatten_valid = torch.index_select(labels_flatten, 0, index)
        
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
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # prob_map = model(inputs) # last activation was a sigmoid
        # outputs = (prob_map > 0.3).float()
        preds = model(inputs)
        outputs = torch.nn.functional.sigmoid(preds)
        
        
        del inputs
        torch.cuda.empty_cache()
        
        
        # flatten outputs
        outputs_flatten = flatten_outputs(outputs, nbreClasses)
        
        del outputs
        torch.cuda.empty_cache()
        
        # outputs_flatten_valid = torch.index_select(outputs_flatten, 0, index)

        
        # loss = criterion(outputs, labels)
        loss = criterion(outputs_flatten, labels)

        # backward
        loss.backward()
        optimizer.step()
        
        # print(loss.item(), outputs.size(0))
        # print(metrics.dice_coeff(outputs_flatten_valid, labels), outputs.size(0))
        
        # train_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
        # train_loss.update(loss.item(), outputs.size(0))
        

            
            
        # print(target.dtype, target.shape)
    
        # compute coeff_dice
        with torch.no_grad():
            num_in_target = outputs_flatten.size(0)
            float_target = labels.float()
            del labels
            torch.cuda.empty_cache()
            smooth = 1.
            pred = outputs_flatten.view(num_in_target, -1)
            truth = float_target.view(num_in_target, -1)
            intersection = (pred * truth).sum(1)
        
            dice_coefficient = (2. * intersection + smooth) /(pred.sum(1) + truth.sum(1) + smooth)
           
            
            train_acc.update(dice_coefficient, outputs_flatten.size(0))
            train_loss.update(loss.item(), outputs_flatten.size(0))
            # print("trn_acc: ", dice_coefficient)
            # print("trn_loss: ", loss.item())
            del outputs_flatten
            del float_target
            torch.cuda.empty_cache()
        # tensorboard logging
#         if idx % log_iter == 0:
# 
#             step = (epoch_num*logger.print_freq)+(idx/log_iter)
# 
#             # log accuracy and loss
#             info = {
#                 'loss': train_loss.avg,
#                 'accuracy': train_acc.avg
#             }
# 
#             for tag, value in info.items():
#                 logger.scalar_summary(tag, value, step)
# 
#             # log weights, biases, and gradients
#             for tag, value in model.named_parameters():
#                 tag = tag.replace('.', '/')
#                 logger.histo_summary(tag, value.data.cpu().numpy(), step)
#                 logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step)
# 
#             # log the sample images
#             log_img = [data_utils.show_tensorboard_image(data['sat_img'], data['map_img'], outputs, as_numpy=True),]
#             logger.image_summary('train_images', log_img, step)

    print('Training Loss: ', train_loss.avg, ' Acc: ', train_acc.avg)
    # print()

    # liberer de la memoire
    # del inputs
    # del labels
    return {'train_loss': train_loss.avg, 'train_acc': train_acc.avg}

def validation(valid_loader, model, criterion, logger, epoch_num, nbreClasses):
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
    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()

    # log_iter = len(valid_loader)//logger.print_freq

    # switch to evaluate mode
    model.eval()
    overall_confusion_matrix = None
    # Iterate over data.
    for idx, data in enumerate(valid_loader):
        torch.cuda.empty_cache()
        with torch.no_grad():
            # flatten label
            labels_flatten = flatten_labels(data['map_img'])
            # index = get_valid_annotations_index(labels_flatten, mask_out_value=255)
            # print(index)
            # labels_flatten_valid = torch.index_select(labels_flatten, 0, index)
            
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
    #         # forward
            # prob_map = model(inputs) # last activation was a sigmoid
            # outputs = (prob_map > 0.3).float()
            outputs = model(inputs)
            # print(outputs.shape)
            outputs = torch.nn.functional.sigmoid(outputs)
            # print(outputs.shape)
            # outputs = torch.nn.LogSoftmax(outputs)
            # flatten outputs
            outputs_flatten = flatten_outputs(outputs, nbreClasses)
            # outputs_flatten_valid = torch.index_select(outputs_flatten, 0, index)
    # 
    #         
    #         # loss = criterion(outputs, labels)
            loss = criterion(outputs_flatten, labels)
    
            num_in_target = outputs_flatten.size(0)
            float_target = labels.float()
            smooth = 1.
            pred = outputs_flatten.view(num_in_target, -1)
            truth = float_target.view(num_in_target, -1)
            intersection = (pred * truth).sum(1)
        
            dice_coefficient = (2. * intersection + smooth) /(pred.sum(1) + truth.sum(1) + smooth)
            valid_acc.update(dice_coefficient, outputs.size(0))
            valid_loss.update(loss.item(), outputs.size(0))
        # print("valid_acc: ", dice_coefficient)
        # print("valid_loss: ", loss.item())
#     
#        valid_acc.update(metrics.dice_coeff(outputs_flatten_valid, labels), outputs.size(0))
#        valid_loss.update(loss.item(), outputs.size(0))


#         logits = model(inputs)
#                 
# 
#         # First we do argmax on gpu and then transfer it to cpu
#         logits = logits.data
#         _, prediction = logits.max(1)
#         prediction = prediction.squeeze(1)
# 
#         prediction_np = prediction.cpu().numpy().flatten()
#         annotation_np = labels.numpy().flatten()
# 
#         # Mask-out value is ignored by default in the sklearn
#         # read sources to see how that was handled
# 
#         current_confusion_matrix = confusion_matrix(y_true=annotation_np, y_pred=prediction_np)
# 
#         if overall_confusion_matrix is None:
#             overall_confusion_matrix = current_confusion_matrix
#         else:
#             overall_confusion_matrix += current_confusion_matrix
#     
#     intersection = np.diag(overall_confusion_matrix)
#     ground_truth_set = overall_confusion_matrix.sum(axis=1)
#     predicted_set = overall_confusion_matrix.sum(axis=0)
#     union =  ground_truth_set + predicted_set - intersection
# 
#     intersection_over_union = intersection / union.astype(np.float32)
#     mean_intersection_over_union = np.mean(intersection_over_union)
#     
#     model.train()
# 
#     
#     print("MIoU: " , mean_intersection_over_union)
#     print()
#     return mean_intersection_over_union


        # tensorboard logging
#         if idx % log_iter == 0:
# 
#             step = (epoch_num*logger.print_freq)+(idx/log_iter)
# 
#             # log accuracy and loss
#             info = {
#                 'loss': valid_loss.avg,
#                 'accuracy': valid_acc.avg
#             }
# 
#             for tag, value in info.items():
#                 logger.scalar_summary(tag, value, step)
# 
#             # log the sample images
#             log_img = [data_utils.show_tensorboard_image(data['sat_img'], data['map_img'], outputs, as_numpy=True),]
#             logger.image_summary('valid_images', log_img, step)
    print('Validation Loss: ', valid_loss.avg, ' Acc: ', valid_acc.avg)
    return valid_loss.avg
#     print()
#     
#     # liberer de la memoire
#     del labels
#     
#     return {'valid_loss': valid_loss.avg, 'valid_acc': valid_acc.avg}


# create a function to save the model state (https://github.com/pytorch/examples/blob/master/imagenet/main.py)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
    num_epoch = 10
    start_epoch = 0
    lr = 0.0005
    momentum = 0.9
    resume = ''
    tailleTuile = 512
    nbrClasses = 4
    nbrEchantTrn = 1000
    nbrEchantVal = 560
    main(TravailFolder, batch_size, num_epoch, start_epoch, lr, momentum, resume,tailleTuile, nbrClasses, nbrEchantTrn, nbrEchantVal)
    print('Fin')

        
# region imports
import logging
from typing import List, Sequence

import torch
# import torch should be first. Unclear issue, mentioned here: https://github.com/pytorch/pytorch/issues/2083
import argparse
from pathlib import Path
import time
import h5py
from datetime import datetime
import warnings
import functools

from tqdm import tqdm
from collections import OrderedDict
import shutil
import numpy as np

try:
    from pynvml import *
except ModuleNotFoundError:
    warnings.warn(f"The python Nvidia management library could not be imported. Ignore if running on CPU only.")

from torch.utils.data import DataLoader
from PIL import Image
from sklearn.utils import compute_sample_weight
from utils import augmentation as aug, create_dataset
from utils.logger import InformationLogger, save_logs_to_bucket, tsv_line
from utils.metrics import report_classification, create_metrics_dict, iou
from models.model_choice import net, load_checkpoint, verify_weights
from utils.utils import load_from_checkpoint, get_device_ids, gpu_stats, get_key_def, get_git_hash
from utils.visualization import vis_from_batch
from utils.readers import read_parameters

from mlflow import log_params, set_tracking_uri, set_experiment, log_artifact, start_run

from utils.tracker_basic import Tracking_Pane
# endregion


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
    # outputs_flatten = torch.tensor(predictions


def loader(path):
    img = Image.open(path)
    return img


def create_dataloader(samples_folder: Path,
                      batch_size: int,
                      eval_batch_size: int,
                      gpu_devices_dict: dict,
                      sample_size: int,
                      dontcare_val: int,
                      crop_size: int,
                      meta_map,
                      num_bands: int,
                      BGR_to_RGB: bool,
                      scale: Sequence,
                      params: dict,
                      dontcare2backgr: bool = False,
                      calc_eval_bs: bool = False,
                      debug: bool = False):
    """
    Function to create dataloader objects for training, validation and test datasets.
    :param samples_folder: path to folder containting .hdf5 files if task is segmentation
    :param batch_size: (int) batch size
    :param gpu_devices_dict: (dict) dictionary where each key contains an available GPU with its ram info stored as value
    :param sample_size: (int) size of hdf5 samples (used to evaluate eval batch-size)
    :param dontcare_val: (int) value in label to be ignored during loss calculation
    :param meta_map: metadata mapping object
    :param num_bands: (int) number of bands in imagery
    :param BGR_to_RGB: (bool) if True, BGR channels will be flipped to RGB
    :param scale: (List) imagery data will be scaled to this min and max value (ex.: 0 to 1)
    :param params: (dict) Parameters found in the yaml config file.
    :param dontcare2backgr: (bool) if True, all dontcare values in label will be replaced with 0 (background value) before training
    :return: trn_dataloader, val_dataloader, tst_dataloader
    """
    if not samples_folder.is_dir():
        raise FileNotFoundError(f'Could not locate: {samples_folder}')
    if not len([f for f in samples_folder.glob('**/*.hdf5')]) >= 1:
        raise FileNotFoundError(f"Couldn't locate .hdf5 files in {samples_folder}")
    num_samples, samples_weight = get_num_samples(samples_path=samples_folder, params=params, dontcare=dontcare_val)
    if not num_samples['trn'] >= batch_size and num_samples['val'] >= batch_size:
        raise ValueError(f"Number of samples in .hdf5 files is less than batch size")
    logging.info(f"Number of samples : {num_samples}\n")
    if not meta_map:
        dataset_constr = create_dataset.SegmentationDataset
    else:
        dataset_constr = functools.partial(create_dataset.MetaSegmentationDataset, meta_map=meta_map)
    datasets = []

    for subset in ["trn", "val", "tst"]:

        datasets.append(dataset_constr(samples_folder, subset, num_bands,
                                       max_sample_count=num_samples[subset],
                                       dontcare=dontcare_val,
                                       # radiom_transform=aug.compose_transforms(params=params,
                                       #                                         dataset=subset,
                                       #                                         aug_type='radiometric'),
                                       # geom_transform=aug.compose_transforms(params=params,
                                       #                                       dataset=subset,
                                       #                                       aug_type='geometric',
                                       #                                       dontcare=dontcare_val,
                                       #                                       crop_size=crop_size),
                                       totensor_transform=aug.compose_transforms(params=params,
                                                                                 dataset=subset,
                                                                                 input_space=BGR_to_RGB,
                                                                                 scale=scale,
                                                                                 dontcare2backgr=dontcare2backgr,
                                                                                 dontcare=dontcare_val,
                                                                                 aug_type='totensor'),
                                       params=params,
                                       debug=debug))
    trn_dataset, val_dataset, tst_dataset = datasets

    # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
    num_workers = len(gpu_devices_dict.keys()) * 4 if len(gpu_devices_dict.keys()) > 1 else 4

    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    if gpu_devices_dict and calc_eval_bs:
        max_pix_per_mb_gpu = 280  # TODO: this value may need to be finetuned
        eval_batch_size = calc_eval_batchsize(gpu_devices_dict, batch_size, sample_size, max_pix_per_mb_gpu)

    trn_dataloader = DataLoader(trn_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, num_workers=num_workers, shuffle=False, drop_last=True)
    tst_dataloader = DataLoader(tst_dataset, batch_size=eval_batch_size, num_workers=num_workers, shuffle=False, drop_last=True) if num_samples['tst'] > 0 else None

    return trn_dataloader, val_dataloader, tst_dataloader


def calc_eval_batchsize(gpu_devices_dict: dict, batch_size: int, sample_size: int, max_pix_per_mb_gpu: int = 280):
    """
    Calculate maximum batch size that could fit on GPU during evaluation based on thumb rule with harcoded
    "pixels per MB of GPU RAM" as threshold. The batch size often needs to be smaller if crop is applied during training
    @param gpu_devices_dict: dictionary containing info on GPU devices as returned by lst_device_ids (utils.py)
    @param batch_size: batch size for training
    @param sample_size: size of hdf5 samples
    @return: returns a downgraded evaluation batch size if the original batch size is considered too high compared to
    the GPU's memory
    """
    eval_batch_size_rd = batch_size
    # get max ram for smallest gpu
    smallest_gpu_ram = min(gpu_info['max_ram'] for _, gpu_info in gpu_devices_dict.items())
    # rule of thumb to determine eval batch size based on approximate max pixels a gpu can handle during evaluation
    pix_per_mb_gpu = (batch_size / len(gpu_devices_dict.keys()) * sample_size ** 2) / smallest_gpu_ram
    if pix_per_mb_gpu >= max_pix_per_mb_gpu:
        eval_batch_size = smallest_gpu_ram * max_pix_per_mb_gpu / sample_size ** 2
        eval_batch_size_rd = int(eval_batch_size - eval_batch_size % len(gpu_devices_dict.keys()))
        eval_batch_size_rd = 1 if eval_batch_size_rd < 1 else eval_batch_size_rd
        logging.warning(f'Validation and test batch size downgraded from {batch_size} to {eval_batch_size} '
                        f'based on max ram of smallest GPU available')
    return eval_batch_size_rd


def get_num_samples(samples_path, params, dontcare):
    """
    Function to retrieve number of samples, either from config file or directly from hdf5 file.
    :param samples_path: (str) Path to samples folder
    :param params: (dict) Parameters found in the yaml config file.
    :param dontcare:
    :return: (dict) number of samples for trn, val and tst.
    """
    num_samples = {'trn': 0, 'val': 0, 'tst': 0}
    weights = []
    samples_weight = None
    for i in ['trn', 'val', 'tst']:
        if get_key_def(f"num_{i}_samples", params['training'], None) is not None:
            num_samples[i] = params['training'][f"num_{i}_samples"]

            with h5py.File(samples_path.joinpath(f"{i}_samples.hdf5"), 'r') as hdf5_file:
                file_num_samples = len(hdf5_file['map_img'])
            if num_samples[i] > file_num_samples:
                raise IndexError(f"The number of training samples in the configuration file ({num_samples[i]}) "
                                 f"exceeds the number of samples in the hdf5 training dataset ({file_num_samples}).")
        else:
            with h5py.File(samples_path.joinpath(f"{i}_samples.hdf5"), "r") as hdf5_file:
                num_samples[i] = len(hdf5_file['map_img'])
                if i == 'trn':
                    for x in range(num_samples[i]):
                        label = hdf5_file['map_img'][x]
                        label = np.where(label == 255, 0, label)
                        unique_labels = np.unique(label)
                        weights.append(''.join([str(int(i)) for i in unique_labels]))
                        samples_weight = compute_sample_weight('balanced', weights)

        # # TODO: why does i == trn have to me down there?
        # with h5py.File(samples_path.joinpath(f"{i}_samples.hdf5"), "r") as hdf5_file:
        #     if i == 'trn':
        #         for x in range(num_samples[i]):
        #             label = hdf5_file['map_img'][x]
        #             label = np.where(label == dontcare, 0, label)
        #             unique_labels = np.unique(label)
        #             weights.append(''.join([str(int(i)) for i in unique_labels]))
        #             samples_weight = compute_sample_weight('balanced', weights)

    return num_samples, samples_weight


def vis_from_dataloader(tracker, params, eval_loader, model, ep_num, output_path, dataset='', scale=None, device=None, vis_batch_range=None):
    """
    Use a model and dataloader to provide outputs that can then be sent to vis_from_batch function to visualize performances of model, for example.
    :param params: (dict) Parameters found in the yaml config file.
    :param eval_loader: data loader
    :param model: model to evaluate
    :param ep_num: epoch index (for file naming purposes)
    :param dataset: (str) 'val or 'tst'
    :param device: device used by pytorch (cpu ou cuda)
    :param vis_batch_range: (int) max number of samples to perform visualization on

    :return:
    """
    vis_path = output_path.joinpath(f'visualization')
    logging.info(f'Visualization figures will be saved to {vis_path}\n')
    min_vis_batch, max_vis_batch, increment = vis_batch_range

    model.eval()
    # with tqdm(eval_loader, dynamic_ncols=True) as _tqdm:
    for batch_index, data in enumerate(tracker.track(eval_loader, 'val vis')):
        if vis_batch_range is not None and batch_index in range(min_vis_batch, max_vis_batch, increment):
            with torch.no_grad():
                try:  # For HPC when device 0 not available. Error: RuntimeError: CUDA error: invalid device ordinal
                    inputs = data['sat_img'].to(device)
                    labels = data['map_img'].to(device)
                except RuntimeError:
                    logging.exception(f'Unable to use device {device}. Trying "cuda:0"')
                    device = torch.device('cuda:0')
                    inputs = data['sat_img'].to(device)
                    labels = data['map_img'].to(device)

                outputs = model(inputs)
                if isinstance(outputs, OrderedDict):
                    outputs = outputs['out']

                vis_from_batch(params, inputs, outputs,
                               batch_index=batch_index,
                               vis_path=vis_path,
                               labels=labels,
                               dataset=dataset,
                               ep_num=ep_num,
                               scale=scale)
    logging.info(f'Saved visualization figures.\n')


def train(tracker, train_loader, model, criterion, optimizer, scheduler, num_classes, batch_size, ep_idx, progress_log, device, scale, vis_params, debug=False):
    """
    Train the model and return the metrics of the training epoch
    :param train_loader: training data loader
    :param model: model to train
    :param criterion: loss criterion
    :param optimizer: optimizer to use
    :param scheduler: learning rate scheduler
    :param num_classes: number of classes
    :param batch_size: number of samples to process simultaneously
    :param ep_idx: epoch index (for hypertrainer log)
    :param progress_log: progress log file (for hypertrainer log)
    :param device: device used by pytorch (cpu ou cuda)
    :param scale: Scale to which values in sat img have been redefined. Useful during visualization
    :param vis_params: (Dict) Parameters useful during visualization
    :param debug: (bool) Debug mode
    :return: Updated training loss
    """
    model.train()
    train_metrics = create_metrics_dict(num_classes)

    # for batch_index, data in enumerate(tqdm(train_loader, desc=f'Iterating train batches with {device.type}')):
    for batch_index, data in enumerate(tracker.track(train_loader, 'trn batch')):
        # loggingdebug()o(f'{len(data["index"].tolist())}   Images = {data["index"].tolist()}')
        tracker.add_stat('samples', data["index"].tolist(), task='batch')
        progress_log.open('a', buffering=1).write(tsv_line(ep_idx, 'trn', batch_index, len(train_loader), time.time()))

        try:  # For HPC when device 0 not available. Error: RuntimeError: CUDA error: invalid device ordinal
            inputs = data['sat_img'].to(device)
            labels = data['map_img'].to(device)
        except RuntimeError:
            logging.exception(f'Unable to use device {device}. Trying "cuda:0"')
            device = torch.device('cuda:0')
            inputs = data['sat_img'].to(device)
            labels = data['map_img'].to(device)

        if num_classes < len(torch.unique(labels)):
            labels = labels[torch.where(labels < num_classes)]
        # forward
        optimizer.zero_grad()
        outputs = model(inputs)
        # added for torchvision models that output an OrderedDict with outputs in 'out' key.
        # More info: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
        if isinstance(outputs, OrderedDict):
            outputs = outputs['out']
        elif isinstance(outputs, tuple):
            outputs = outputs[0]

        # vis_batch_range: range of batches to perform visualization on. see README.md for more info.
        # vis_at_eval: (bool) if True, will perform visualization at eval time, as long as vis_batch_range is valid
        if vis_params['vis_batch_range'] and vis_params['vis_at_train']:
            min_vis_batch, max_vis_batch, increment = vis_params['vis_batch_range']
            if batch_index in range(min_vis_batch, max_vis_batch, increment):
                vis_path = progress_log.parent.joinpath('visualization')
                if ep_idx == 0:
                    logging.info(f'Visualizing on train outputs for batches in range {vis_params["vis_batch_range"]}. All images will be saved to {vis_path}\n')
                vis_from_batch(vis_params, inputs, outputs,
                               batch_index=batch_index,
                               vis_path=vis_path,
                               labels=labels,
                               dataset='trn',
                               ep_num=ep_idx+1,
                               scale=scale)

        loss = criterion(outputs, labels)
        train_metrics['loss'].update(loss.item(), batch_size)

        tracker.add_stats({'dataset' : 'trn',
                           'loss'    : f'{train_metrics["loss"].avg:.4f}',
                           'lr'      :optimizer.param_groups[0]['lr']},
                           task='batch')

        if device.type == 'cuda' and debug:
            res, mem = gpu_stats(device=device.index)
            logging.debug(OrderedDict(trn_loss=f'{train_metrics["loss"].val:.2f}',
                                          gpu_perc=f'{res.gpu} %',
                                          gpu_RAM=f'{mem.used / (1024 ** 2):.0f}/{mem.total / (1024 ** 2):.0f} MiB',
                                          lr=optimizer.param_groups[0]['lr'],
                                          img=data['sat_img'].numpy().shape,
                                          smpl=data['map_img'].numpy().shape,
                                          bs=batch_size,
                                          out_vals=np.unique(outputs[0].argmax(dim=0).detach().cpu().numpy()),     # TODO: change to all np.unique to torch.unique.cpu()
                                          gt_vals=np.unique(labels[0].detach().cpu().numpy())))
            tracker.add_stats({'device'  : device,
                               'gpu_perc':f'{res.gpu} %',
                               'gpu_RAM' :f'{mem.used/(1024**2):.0f}/{mem.total/(1024**2):.0f}MB'},
                                task='batch')
        loss.backward()
        optimizer.step()

    scheduler.step()
    if train_metrics["loss"].avg is not None:
        logging.info(f'Training Loss: {train_metrics["loss"].avg:.4f}')
        tracker.add_stat('trn loss', train_metrics["loss"].avg, task='epoch')
    tracker.notify_end('trn batch')
    return train_metrics


def evaluation(tracker, eval_loader, model, criterion, num_classes, batch_size, ep_idx, progress_log, scale, vis_params, batch_metrics=None, dataset='val', device=None, debug=False):
    """
    Evaluate the model and return the updated metrics
    :param eval_loader: data loader
    :param model: model to evaluate
    :param criterion: loss criterion
    :param num_classes: number of classes
    :param batch_size: number of samples to process simultaneously
    :param ep_idx: epoch index (for hypertrainer log)
    :param progress_log: progress log file (for hypertrainer log)
    :param scale: Scale to which values in sat img have been redefined. Useful during visualization
    :param vis_params: (Dict) Parameters useful during visualization
    :param batch_metrics: (int) Metrics computed every (int) batches. If left blank, will not perform metrics.
    :param dataset: (str) 'val or 'tst'
    :param device: device used by pytorch (cpu ou cuda)
    :param debug: if True, debug functions will be performed
    :return: (dict) eval_metrics

    """
    eval_metrics = create_metrics_dict(num_classes)
    model.eval()

    # for batch_index, data in enumerate(tqdm(eval_loader, dynamic_ncols=True, desc=f'Iterating {dataset} ')):
    for batch_index, data in enumerate(tracker.track(eval_loader, 'val batch')):
        logging.debug(f'{len(data["index"].tolist())}   Images = {data["index"].tolist()}')
        tracker.add_stat('samples', data["index"].tolist(), task='batch')
        progress_log.open('a', buffering=1).write(tsv_line(ep_idx, dataset, batch_index, len(eval_loader), time.time()))

        with torch.no_grad():
            try:  # For HPC when device 0 not available. Error: RuntimeError: CUDA error: invalid device ordinal
                inputs = data['sat_img'].to(device)
                labels = data['map_img'].to(device)
            except RuntimeError:
                logging.exception(f'Unable to use device {device}. Trying "cuda:0"')
                device = torch.device('cuda:0')
                inputs = data['sat_img'].to(device)
                labels = data['map_img'].to(device)

            labels_flatten = flatten_labels(labels)

            outputs = model(inputs)
            if isinstance(outputs, OrderedDict):
                outputs = outputs['out']
            elif isinstance(outputs, tuple):
                outputs = outputs[0]

            # vis_batch_range: range of batches to perform visualization on. see README.md for more info.
            # vis_at_eval: (bool) if True, will perform visualization at eval time, as long as vis_batch_range is valid
            if vis_params['vis_batch_range'] and vis_params['vis_at_eval']:
                min_vis_batch, max_vis_batch, increment = vis_params['vis_batch_range']
                if batch_index in range(min_vis_batch, max_vis_batch, increment):
                    vis_path = progress_log.parent.joinpath('visualization')
                    if ep_idx == 0 and batch_index == min_vis_batch:
                        tqdm.write(f'Visualizing on {dataset} outputs for batches in range {vis_params["vis_batch_range"]}. All '
                                   f'images will be saved to {vis_path}\n')
                    vis_from_batch(vis_params, inputs, outputs,
                                   batch_index=batch_index,
                                   vis_path=vis_path,
                                   labels=labels,
                                   dataset=dataset,
                                   ep_num=ep_idx+1,
                                   scale=scale)

            outputs_flatten = flatten_outputs(outputs, num_classes)
            # for j in range(outputs.shape[0]):
            #     for i in range(outputs.shape[1]):
            #         Image.fromarray(outputs.detach().cpu().numpy()[j, i, ...]).convert('RGB').save(f'D:/NRCan_data/MECnet_implementation/runs/mecnet_batch{j}_2class_class{i}.png')
            #         print('saved!', f'D:/NRCan_data/MECnet_implementation/runs/mecnet_batch{j}_2class_class{i}.png')

            loss = criterion(outputs, labels)
            # print((labels & outputs).shape)
            eval_metrics['loss'].update(loss.item(), batch_size)

            if (dataset == 'val') and (batch_metrics is not None):
                # Compute metrics every n batches. Time consuming.
                if not batch_metrics <= len(eval_loader):
                    logging.error(f"Batch_metrics ({batch_metrics}) is smaller than batch size "
                                  f"{len(eval_loader)}. Metrics in validation loop won't be computed")
                if (batch_index+1) % batch_metrics == 0:   # +1 to skip val loop at very beginning
                    a, segmentation = torch.max(outputs_flatten, dim=1)
                    eval_metrics = iou(segmentation, labels_flatten, batch_size, num_classes, eval_metrics)
                    eval_metrics = report_classification(segmentation, labels_flatten, batch_size, eval_metrics,
                                                         ignore_index=eval_loader.dataset.dontcare)
            elif dataset == 'tst':
                a, segmentation = torch.max(outputs_flatten, dim=1)
                eval_metrics = iou(segmentation, labels_flatten, batch_size, num_classes, eval_metrics)
                eval_metrics = report_classification(segmentation, labels_flatten, batch_size, eval_metrics,
                                                     ignore_index=eval_loader.dataset.dontcare)

            logging.debug(OrderedDict(dataset=dataset, loss=f'{eval_metrics["loss"].avg:.4f}'))
            tracker.add_stats({'dataset' : dataset, 'loss' : f'{eval_metrics["loss"].avg:.4f}'}, task='batch')

            if debug and device.type == 'cuda':
                res, mem = gpu_stats(device=device.index)
                logging.debug(OrderedDict(device=device, gpu_perc=f'{res.gpu} %',
                                              gpu_RAM=f'{mem.used/(1024**2):.0f}/{mem.total/(1024**2):.0f} MiB'))
                tracker.add_stats({'device'  : device,
                                   'gpu_perc':f'{res.gpu} %',
                                   'gpu_RAM' :f'{mem.used/(1024**2):.0f}/{mem.total/(1024**2):.0f}MB'},
                                   task='batch')

    logging.info(f"{dataset} Loss: {eval_metrics['loss'].avg}")
    tracker.add_stat('val loss', eval_metrics['loss'].avg, task='epoch')
    if batch_metrics is not None:
        logging.info(f"{dataset} precision: {eval_metrics['precision'].avg}")
        logging.info(f"{dataset} recall: {eval_metrics['recall'].avg}")
        logging.info(f"{dataset} fscore: {eval_metrics['fscore'].avg}")
        logging.info(f"{dataset} iou: {eval_metrics['iou'].avg}")

        tracker.add_stat('precision', eval_metrics['precision'].avg, task='epoch')
        tracker.add_stat('recall', eval_metrics['recall'].avg, task='epoch')
        tracker.add_stat('fscore', eval_metrics['fscore'].avg, task='epoch')
        tracker.add_stat('iou', eval_metrics['iou'].avg, task='epoch')
    tracker.notify_end('val batch')

    return eval_metrics


def main(params, config_path):
    """
    Function to train and validate a model for semantic segmentation.

    Process
    -------
    1. Model is instantiated and checkpoint is loaded from path, if provided in
       `your_config.yaml`.
    2. GPUs are requested according to desired amount of `num_gpus` and
       available GPUs.
    3. If more than 1 GPU is requested, model is cast to DataParallel model
    4. Dataloaders are created with `create_dataloader()`
    5. Loss criterion, optimizer and learning rate are set with
       `set_hyperparameters()` as requested in `config.yaml`.
    5. Using these hyperparameters, the application will try to minimize the
       loss on the training data and evaluate every epoch on the validation
       data.
    6. For every epoch, the application shows and logs the loss on "trn" and
       "val" datasets.
    7. For every epoch (if `batch_metrics: 1`), the application shows and logs
       the accuracy, recall and f-score on "val" dataset. Those metrics are
       also computed on each class.
    8. At the end of the training process, the application shows and logs the
       accuracy, recall and f-score on "tst" dataset. Those metrics are also
       computed on each class.

    -------
    :param params: (dict) Parameters found in the yaml config file.
    :param config_path: (str) Path to the yaml config file.
    """
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # region MANDATORY PARAMETERS
    num_classes = get_key_def('num_classes', params['global'], expected_type=int)
    num_bands = get_key_def('number_of_bands', params['global'], expected_type=int)
    batch_size = get_key_def('batch_size', params['training'], expected_type=int)
    eval_batch_size = get_key_def('eval_batch_size', params['training'], expected_type=int, default=batch_size)
    num_epochs = get_key_def('num_epochs', params['training'], expected_type=int)
    model_name = get_key_def('model_name', params['global'], expected_type=str).lower()
    BGR_to_RGB = get_key_def('BGR_to_RGB', params['global'], expected_type=bool)
    # endregion

    # region OPTIONAL PARAMETERS
    # basics
    debug = get_key_def('debug_mode', params['global'], default=False, expected_type=bool)
    task = get_key_def('task', params['global'], default='segmentation', expected_type=str)
    if not task == 'segmentation':
        raise ValueError(f"The task should be segmentation. The provided value is {task}")
    dontcare_val = get_key_def("ignore_index", params["training"], default=-1, expected_type=int)
    crop_size = get_key_def('target_size', params['training'], default=None, expected_type=int)
    batch_metrics = get_key_def('batch_metrics', params['training'], default=None, expected_type=int)
    meta_map = get_key_def("meta_map", params["global"], default=None)
    if meta_map and not Path(meta_map).is_file():
        raise FileNotFoundError(f'Couldn\'t locate {meta_map}')
    bucket_name = get_key_def('bucket_name', params['global'])  # AWS
    scale = get_key_def('scale_data', params['global'], default=[0, 1], expected_type=List)
    # endregion

    # region model params
    loss_fn = get_key_def('loss_fn', params['training'], default='CrossEntropy', expected_type=str)
    class_weights = get_key_def('class_weights', params['training'], default=None, expected_type=Sequence)
    if class_weights:
        verify_weights(num_classes, class_weights)
    optimizer = get_key_def('optimizer', params['training'], default='adam', expected_type=str)
    pretrained = get_key_def('pretrained', params['training'], default=True, expected_type=bool)
    train_state_dict_path = get_key_def('state_dict_path', params['training'], default=None, expected_type=str)
    if train_state_dict_path and not Path(train_state_dict_path).is_file():
        raise FileNotFoundError(f'Could not locate pretrained checkpoint for training: {train_state_dict_path}')
    dropout_prob = get_key_def('dropout_prob', params['training'], default=None, expected_type=float)
    # Read the concatenation point
    # TODO: find a way to maybe implement it in classification one day
    conc_point = get_key_def('concatenate_depth', params['global'], None)
    # endregion

    # region gpu parameters
    num_devices = get_key_def('num_gpus', params['global'], default=0, expected_type=int)
    if num_devices and not num_devices >= 0:
        raise ValueError("missing mandatory num gpus parameter")
    default_max_used_ram = 15
    max_used_ram = get_key_def('max_used_ram', params['global'], default=default_max_used_ram, expected_type=int)
    max_used_perc = get_key_def('max_used_perc', params['global'], default=15, expected_type=int)
    # endregion

    # region mlflow logging
    mlflow_uri = get_key_def('mlflow_uri', params['global'], default="./mlruns")
    Path(mlflow_uri).mkdir(exist_ok=True)
    experiment_name = get_key_def('mlflow_experiment_name', params['global'], default='gdl-training', expected_type=str)
    run_name = get_key_def('mlflow_run_name', params['global'], default='gdl', expected_type=str)
    # endregion

    # region parameters to find hdf5 samples
    data_path = Path(get_key_def('data_path', params['global'], './data', expected_type=str))
    samples_size = get_key_def("samples_size", params["global"], default=1024, expected_type=int)
    overlap = get_key_def("overlap", params["sample"], default=5, expected_type=int)
    min_annot_perc = get_key_def('min_annotated_percent', params['sample']['sampling_method'], default=0,
                                 expected_type=int)
    if not data_path.is_dir():
        raise FileNotFoundError(f'Could not locate data path {data_path}')
    samples_folder_name = (f'samples{samples_size}_overlap{overlap}_min-annot{min_annot_perc}_{num_bands}bands_{experiment_name}')
    samples_folder = data_path.joinpath(samples_folder_name)
    # endregion

    # region visualization parameters
    vis_at_train = get_key_def('vis_at_train', params['visualization'], default=False)
    vis_at_eval = get_key_def('vis_at_evaluation', params['visualization'], default=False)
    vis_batch_range = get_key_def('vis_batch_range', params['visualization'], default=None)
    vis_at_checkpoint = get_key_def('vis_at_checkpoint', params['visualization'], default=False)
    ep_vis_min_thresh = get_key_def('vis_at_ckpt_min_ep_diff', params['visualization'], default=1, expected_type=int)
    vis_at_ckpt_dataset = get_key_def('vis_at_ckpt_dataset', params['visualization'], 'val')
    colormap_file = get_key_def('colormap_file', params['visualization'], None)
    heatmaps = get_key_def('heatmaps', params['visualization'], False)
    heatmaps_inf = get_key_def('heatmaps', params['inference'], False)
    grid = get_key_def('grid', params['visualization'], False)
    mean = get_key_def('mean', params['training']['normalization'])
    std = get_key_def('std', params['training']['normalization'])
    vis_params = {'colormap_file': colormap_file, 'heatmaps': heatmaps, 'heatmaps_inf': heatmaps_inf, 'grid': grid,
                  'mean': mean, 'std': std, 'vis_batch_range': vis_batch_range, 'vis_at_train': vis_at_train,
                  'vis_at_eval': vis_at_eval, 'ignore_index': dontcare_val, 'inference_input_path': None}
    # endregion

    # region coordconv parameters
    coordconv_params = {}
    for param, val in params['global'].items():
        if 'coordconv' in param:
            coordconv_params[param] = val
    # endregion

    # region git
    # add git hash from current commit to parameters if available. Parameters will be saved to model's .pth.tar
    params['global']['git_hash'] = get_git_hash()
    # endregion

    # automatic model naming with unique id for each training
    model_id = config_path.stem
    output_path = samples_folder.joinpath('model') / model_id
    if output_path.is_dir():
        last_mod_time_suffix = datetime.fromtimestamp(output_path.stat().st_mtime).strftime('%Y%m%d-%H%M%S')
        archive_output_path = samples_folder.joinpath('model') / f"{model_id}_{last_mod_time_suffix}"
        shutil.move(output_path, archive_output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    shutil.copy(str(config_path), str(output_path))  # copy yaml to output path where model will be saved

    import logging.config  # See: https://docs.python.org/2.4/lib/logging-config-fileformat.html
    log_config_path = Path('utils/logging.conf').absolute()
    logfile = f'{output_path}/{model_id}.log'
    logfile_debug = f'{output_path}/{model_id}_debug.log'
    console_level_logging = 'INFO' if not debug else 'DEBUG'
    if params['global']['my_comp']:
        logfile = f'{"D:/NRCan_data/MECnet_implementation/runs/"}/{model_id}.log'
        logfile_debug = f'{"D:/NRCan_data/MECnet_implementation/runs/"}/{model_id}_debug.log'
        logging.config.fileConfig(log_config_path, defaults={'logfilename': logfile,
                                                             'logfilename_debug': logfile_debug,
                                                             'console_level': console_level_logging})
    else:
        logging.config.fileConfig(log_config_path, defaults={'logfilename': logfile,
                                                             'logfilename_debug': logfile_debug,
                                                             'console_level': console_level_logging})

    # now that we know where logs will be saved, we can start logging!
    if not (0 <= max_used_ram <= 100):
        logging.warning(f'Max used ram parameter should be a percentage. Got {max_used_ram}. '
                        f'Will set default value of {default_max_used_ram} %')
        max_used_ram = default_max_used_ram

    logging.info(f'Model and log files will be saved to: {output_path}\n\n')
    if debug:
        logging.warning(f'Debug mode activated. Some debug features may mobilize extra disk space and '
                        f'cause delays in execution.')
    if dontcare_val < 0 and vis_batch_range:
        logging.warning(f'Visualization: expected positive value for ignore_index, got {dontcare_val}.'
                        f'Will be overridden to 255 during visualization only. Problems may occur.')

    # list of GPU devices that are available and unused. If no GPUs, returns empty list
    gpu_devices_dict = get_device_ids(num_devices,
                                      max_used_ram_perc=max_used_ram,
                                      max_used_perc=max_used_perc)
    logging.info(f'GPUs devices available: {gpu_devices_dict}')
    num_devices = len(gpu_devices_dict.keys())
    device = torch.device(f'cuda:{list(gpu_devices_dict.keys())[0]}' if gpu_devices_dict else 'cpu')

    logging.info(f'Creating dataloaders from data in {samples_folder}...\n')

    # overwrite dontcare values in label if loss is not lovasz or crossentropy. FIXME: hacky fix.
    dontcare2backgr = False
    if loss_fn not in ['Lovasz', 'CrossEntropy', 'OhemCrossEntropy']:
        dontcare2backgr = True
        logging.warning(f'Dontcare is not implemented for loss function "{loss_fn}". '
                        f'Dontcare values ({dontcare_val}) in label will be replaced with background value (0)')

    # Will check if batch size needs to be a lower value only if cropping samples during training
    calc_eval_bs = True if crop_size else False

    trn_dataloader, val_dataloader, tst_dataloader = create_dataloader(samples_folder=samples_folder,
                                                                       batch_size=batch_size,
                                                                       eval_batch_size=eval_batch_size,
                                                                       gpu_devices_dict=gpu_devices_dict,
                                                                       sample_size=samples_size,
                                                                       dontcare_val=dontcare_val,
                                                                       crop_size=crop_size,
                                                                       meta_map=meta_map,
                                                                       num_bands=num_bands,
                                                                       BGR_to_RGB=BGR_to_RGB,
                                                                       scale=scale,
                                                                       params=params,
                                                                       dontcare2backgr=dontcare2backgr,
                                                                       calc_eval_bs=calc_eval_bs,
                                                                       debug=debug)
    # region INSTANTIATE MODEL AND LOAD CHECKPOINT FROM PATH
    num_classes_corrected = num_classes + 1  # + 1 for background # FIXME temporary patch for num_classes problem.
    model, model_name, criterion, optimizer, lr_scheduler = net(model_name=model_name,
                                                                num_bands=num_bands,
                                                                num_channels=num_classes_corrected,
                                                                dontcare_val=dontcare_val,
                                                                num_devices=num_devices,
                                                                train_state_dict_path=train_state_dict_path,
                                                                pretrained=pretrained,
                                                                dropout_prob=dropout_prob,
                                                                loss_fn=loss_fn,
                                                                class_weights=class_weights,
                                                                optimizer=optimizer,
                                                                net_params=params,
                                                                conc_point=conc_point,
                                                                coordconv_params=coordconv_params)

    logging.info(f'Instantiated {model_name} model with {num_classes_corrected} output channels.\n'
                 f'lr_scheduler:{type(lr_scheduler)} criterion:{type(criterion)} optimizer:{type(optimizer)}')
    # endregion
    # region mlflow tracking path + parameters logging
    set_tracking_uri(mlflow_uri)
    set_experiment(experiment_name)
    start_run(run_name=run_name)
    log_params(params['training'])
    log_params(params['global'])
    log_params(params['sample'])

    if bucket_name:
        from utils.aws import download_s3_files
        bucket, bucket_output_path, output_path, data_path = download_s3_files(bucket_name=bucket_name,
                                                                               data_path=data_path,
                                                                               output_path=output_path)
    since = time.time()
    best_loss = 999
    last_vis_epoch = 0
    # endregion

    # region init loggers
    progress_log = output_path / 'progress.log'
    if not progress_log.exists():
        progress_log.open('w', buffering=1).write(tsv_line('ep_idx', 'phase', 'iter', 'i_p_ep', 'time'))  # Add header

    trn_log = InformationLogger('trn')
    val_log = InformationLogger('val')
    tst_log = InformationLogger('tst')
    filename = output_path.joinpath('checkpoint.pth.tar')
    # endregion

    tracker = Tracking_Pane(output_path,
                            mode='trn_seg',
                            stats_to_track = {'epoch' : ['save_check',
                                                         'iou',
                                                         'val loss',
                                                         'trn loss',
                                                         'precision',
                                                         'recall',
                                                         'fscore'],
                                              'batch' : ['loss',
                                                         'lr',
                                                         'dataset',
                                                         'device',
                                                         'gpu_perc',
                                                         'gpu_RAM',
                                                         ['samples', 20]]})

    # region VISUALIZATION: generate pngs of inputs, labels and outputs
    vis_batch_range = get_key_def('vis_batch_range', params['visualization'], None)
    if vis_batch_range is not None:
        # Make sure user-provided range is a tuple with 3 integers (start, finish, increment).
        # Check once for all visualization tasks.
        if not isinstance(vis_batch_range, list) and len(vis_batch_range) == 3 and all(isinstance(x, int)
                                                                                       for x in vis_batch_range):
            raise ValueError(f'Vis_batch_range expects three integers in a list: start batch, end batch, increment.'
                             f'Got {vis_batch_range}')
        vis_at_init_dataset = get_key_def('vis_at_init_dataset', params['visualization'], 'val')

        # Visualization at initialization. Visualize batch range before first eopch.
        if get_key_def('vis_at_init', params['visualization'], False):
            logging.info(f'Visualizing initialized model on batch range {vis_batch_range} '
                         f'from {vis_at_init_dataset} dataset...\n')
            vis_from_dataloader(tracker,
                                vis_params=params,
                                eval_loader=val_dataloader if vis_at_init_dataset == 'val' else tst_dataloader,
                                model=model,
                                ep_num=0,
                                output_path=output_path,
                                dataset=vis_at_init_dataset,
                                scale=scale,
                                device=device,
                                vis_batch_range=vis_batch_range)
    # endregion


    for epoch in tracker.track(range(0, num_epochs), 'epoch'):
        logging.info(f'\nEpoch {epoch}/{num_epochs - 1}\n{"-" * 20}')

        trn_report = train(tracker,
                           train_loader=trn_dataloader,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           scheduler=lr_scheduler,
                           num_classes=num_classes_corrected,
                           batch_size=batch_size,
                           ep_idx=epoch,
                           progress_log=progress_log,
                           device=device,
                           scale=scale,
                           vis_params=vis_params,
                           debug=debug)
        trn_log.add_values(trn_report, epoch, ignore=['precision', 'recall', 'fscore', 'iou'])

        val_report = evaluation(tracker,
                                eval_loader=val_dataloader,
                                model=model,
                                criterion=criterion,
                                num_classes=num_classes_corrected,
                                batch_size=batch_size,
                                ep_idx=epoch,
                                progress_log=progress_log,
                                batch_metrics=batch_metrics,
                                dataset='val',
                                device=device,
                                scale=scale,
                                vis_params=vis_params,
                                debug=debug)
        val_loss = val_report['loss'].avg

        if batch_metrics is not None:
            val_log.add_values(val_report, epoch)
        else:
            val_log.add_values(val_report, epoch, ignore=['precision', 'recall', 'fscore', 'iou'])

        if val_loss < best_loss:
            logging.info("save checkpoint\n")
            tracker.add_stat('save_check', 'true!', task='epoch')
            best_loss = val_loss
            # More info: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-torch-nn-dataparallel-models
            state_dict = model.module.state_dict() if num_devices > 1 else model.state_dict()
            torch.save({'epoch': epoch,
                        'params': params,
                        'model': state_dict,
                        'best_loss': best_loss,
                        'optimizer': optimizer.state_dict()}, filename)
            if epoch == 0:
                log_artifact(filename)
            if bucket_name:
                bucket_filename = bucket_output_path.joinpath('checkpoint.pth.tar')
                bucket.upload_file(filename, bucket_filename)

            # VISUALIZATION: generate pngs of img samples, labels and outputs as alternative to follow training
            if vis_batch_range is not None and vis_at_checkpoint and epoch - last_vis_epoch >= ep_vis_min_thresh:
                if last_vis_epoch == 0:
                    logging.info(f'Visualizing with {vis_at_ckpt_dataset} dataset samples on checkpointed model for'
                                 f'batches in range {vis_batch_range}')
                vis_from_dataloader(tracker,
                                    vis_params=vis_params,
                                    eval_loader=val_dataloader if vis_at_ckpt_dataset == 'val' else tst_dataloader,
                                    model=model,
                                    ep_num=epoch+1,
                                    output_path=output_path,
                                    dataset=vis_at_ckpt_dataset,
                                    scale=scale,
                                    device=device,
                                    vis_batch_range=vis_batch_range)
                last_vis_epoch = epoch

        if bucket_name:
            save_logs_to_bucket(bucket, bucket_output_path, output_path, now, params['training']['batch_metrics'])

        cur_elapsed = time.time() - since
        logging.info(f'Current elapsed time {cur_elapsed // 60:.0f}m {cur_elapsed % 60:.0f}s')
    tracker.notify_end('epoch')

    # load checkpoint model and evaluate it on test dataset.
    if num_epochs > 0:   # if num_epochs is set to 0, model is loaded to evaluate on test set
        checkpoint = load_checkpoint(filename)
        model, _ = load_from_checkpoint(checkpoint, model)

    if tst_dataloader:
        tst_report = evaluation(eval_loader=tst_dataloader,
                                model=model,
                                criterion=criterion,
                                num_classes=num_classes_corrected,
                                batch_size=batch_size,
                                ep_idx=num_epochs,
                                progress_log=progress_log,
                                batch_metrics=batch_metrics,
                                dataset='tst',
                                scale=scale,
                                vis_params=vis_params,
                                device=device)
        tst_log.add_values(tst_report, num_epochs)

        if bucket_name:
            bucket_filename = bucket_output_path.joinpath('last_epoch.pth.tar')
            bucket.upload_file("output.txt", bucket_output_path.joinpath(f"Logs/{now}_output.txt"))
            bucket.upload_file(filename, bucket_filename)

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    log_artifact(logfile)
    log_artifact(logfile_debug)


if __name__ == '__main__':
    print(f'Start\n')
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Training execution')
    parser.add_argument('param_file', help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    config_path = Path(args.param_file)
    params = read_parameters(args.param_file)

    # Limit of the NIR implementation TODO: Update after each version
    modalities = None if 'modalities' not in params['global'] else params['global']['modalities']
    if 'deeplabv3' not in params['global']['model_name'] and modalities == 'RGBN':
        print(
            '\n The NIR modality will only be concatenate at the begining,' /
            ' the implementation of the concatenation point is only available' /
            ' for the deeplabv3 model for now. \n More will follow on demande.\n'
             )

    main(params, config_path)
    print('End of training')

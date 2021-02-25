import torch
# import torch should be first. Unclear issue, mentioned here: https://github.com/pytorch/pytorch/issues/2083
import argparse
from pathlib import Path
import time
import h5py
import datetime
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

import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image

from utils import augmentation as aug, CreateDataset
from utils.optimizer import create_optimizer
from utils.logger import InformationLogger, save_logs_to_bucket, tsv_line
from utils.metrics import report_classification, create_metrics_dict, iou
from models.model_choice import net, load_checkpoint
from losses import MultiClassCriterion
from utils.utils import load_from_checkpoint, get_device_ids, gpu_stats, get_key_def
from utils.visualization import vis_from_batch
from utils.readers import read_parameters
from mlflow import log_params, set_tracking_uri, set_experiment, log_artifact, start_run


def verify_weights(num_classes, weights):
    """Verifies that the number of weights equals the number of classes if any are given
    Args:
        num_classes: number of classes defined in the configuration file
        weights: weights defined in the configuration file
    """
    if num_classes == 1 and len(weights) == 2:
        warnings.warn("got two class weights for single class defined in configuration file; will assume index 0 = background")
    elif num_classes != len(weights):
        raise ValueError('The number of class weights in the configuration file is different than the number of classes')


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


def loader(path):
    img = Image.open(path)
    return img


def create_dataloader(samples_folder, batch_size, num_devices, params):
    """
    Function to create dataloader objects for training, validation and test datasets.
    :param samples_folder: path to the folder containting .hdf5 files if task is segmentation
    :param batch_size: (int) batch size
    :param num_devices: (int) number of GPUs used
    :param params: (dict) Parameters found in the yaml config file.
    :return: trn_dataloader, val_dataloader, tst_dataloader
    """
    debug = get_key_def('debug_mode', params['global'], False)
    dontcare_val = get_key_def("ignore_index", params["training"], -1)

    assert samples_folder.is_dir(), f'Could not locate: {samples_folder}'
    assert len([f for f in samples_folder.glob('**/*.hdf5')]) >= 1, f"Couldn't locate .hdf5 files in {samples_folder}"
    num_samples = get_num_samples(samples_path=samples_folder, params=params)
    assert num_samples['trn'] >= batch_size and num_samples['val'] >= batch_size, f"Number of samples in .hdf5 files is less than batch size"
    print(f"Number of samples : {num_samples}\n")
    meta_map = get_key_def("meta_map", params["global"], {})
    num_bands = get_key_def("number_of_bands", params["global"], {})
    if not meta_map:
        dataset_constr = CreateDataset.SegmentationDataset
    else:
        dataset_constr = functools.partial(CreateDataset.MetaSegmentationDataset, meta_map=meta_map)
    datasets = []

    for subset in ["trn", "val", "tst"]:
        # TODO: transform the aug.compose_transforms in a class with radiom, geom, totensor in def
        datasets.append(dataset_constr(samples_folder, subset, num_bands,
                                       max_sample_count=num_samples[subset],
                                       dontcare=dontcare_val,
                                       radiom_transform=aug.compose_transforms(params, subset, type='radiometric'),
                                       geom_transform=aug.compose_transforms(params, subset, type='geometric',
                                                                             ignore_index=dontcare_val),
                                       totensor_transform=aug.compose_transforms(params, subset, type='totensor'),
                                       debug=debug))
    trn_dataset, val_dataset, tst_dataset = datasets

    # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
    num_workers = num_devices * 4 if num_devices > 1 else 4

    # Shuffle must be set to True.
    trn_dataloader = DataLoader(trn_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                drop_last=True)
    # Using batch_metrics with shuffle=False on val dataset will always mesure metrics on the same portion of the val samples.
    # Shuffle should be set to True.
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                drop_last=True)
    tst_dataloader = DataLoader(tst_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                drop_last=True) if num_samples['tst'] > 0 else None

    return trn_dataloader, val_dataloader, tst_dataloader


def get_num_samples(samples_path, params):
    """
    Function to retrieve number of samples, either from config file or directly from hdf5 file.
    :param samples_path: (str) Path to samples folder
    :param params: (dict) Parameters found in the yaml config file.
    :return: (dict) number of samples for trn, val and tst.
    """
    num_samples = {'trn': 0, 'val': 0, 'tst': 0}

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

    return num_samples


def set_hyperparameters(params, num_classes, model, checkpoint, dontcare_val):
    """
    Function to set hyperparameters based on values provided in yaml config file.
    Will also set model to GPU, if available.
    If none provided, default functions values may be used.
    :param params: (dict) Parameters found in the yaml config file
    :param num_classes: (int) number of classes for current task
    :param model: Model loaded from model_choice.py
    :param checkpoint: (dict) state dict as loaded by model_choice.py
    :return: model, criterion, optimizer, lr_scheduler, num_gpus
    """
    # set mandatory hyperparameters values with those in config file if they exist
    lr = get_key_def('learning_rate', params['training'], None, "missing mandatory learning rate parameter")
    weight_decay = get_key_def('weight_decay', params['training'], None, "missing mandatory weight decay parameter")
    step_size = get_key_def('step_size', params['training'], None, "missing mandatory step size parameter")
    gamma = get_key_def('gamma', params['training'], None, "missing mandatory gamma parameter")

    # optional hyperparameters. Set to None if not in config file
    class_weights = torch.tensor(params['training']['class_weights']) if params['training']['class_weights'] else None
    if params['training']['class_weights']:
        verify_weights(num_classes, class_weights)

    # Loss function
    criterion = MultiClassCriterion(loss_type=params['training']['loss_fn'],
                                    ignore_index=dontcare_val,
                                    weight=class_weights)

    # Optimizer
    opt_fn = params['training']['optimizer']
    optimizer = create_optimizer(params=model.parameters(), mode=opt_fn, base_lr=lr, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

    if checkpoint:
        tqdm.write(f'Loading checkpoint...')
        model, optimizer = load_from_checkpoint(checkpoint, model, optimizer=optimizer)

    return model, criterion, optimizer, lr_scheduler


def vis_from_dataloader(params, eval_loader, model, ep_num, output_path, dataset='', device=None, vis_batch_range=None):
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
    tqdm.write(f'Visualization figures will be saved to {vis_path}\n')
    min_vis_batch, max_vis_batch, increment = vis_batch_range

    model.eval()
    with tqdm(eval_loader, dynamic_ncols=True) as _tqdm:
        for batch_index, data in enumerate(_tqdm):
            if vis_batch_range is not None and batch_index in range(min_vis_batch, max_vis_batch, increment):
                with torch.no_grad():
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
                                   ep_num=ep_num)
    tqdm.write(f'Saved visualization figures.\n')


def train(train_loader,
          model,
          criterion,
          optimizer,
          scheduler,
          num_classes,
          batch_size,
          ep_idx,
          progress_log,
          vis_params,
          device,
          debug=False
          ):
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
    :param vis_params: (dict) Parameters found in the yaml config file. Named vis_params because they are only used for
                        visualization functions.
    :param device: device used by pytorch (cpu ou cuda)
    :param debug: (bool) Debug mode
    :return: Updated training loss
    """
    model.train()
    train_metrics = create_metrics_dict(num_classes)
    vis_at_train = get_key_def('vis_at_train', vis_params['visualization'], False)
    vis_batch_range = get_key_def('vis_batch_range', vis_params['visualization'], None)

    with tqdm(train_loader, desc=f'Iterating train batches with {device.type}') as _tqdm:
        for batch_index, data in enumerate(_tqdm):
            progress_log.open('a', buffering=1).write(tsv_line(ep_idx, 'trn', batch_index, len(train_loader), time.time()))

            inputs = data['sat_img'].to(device)
            labels = data['map_img'].to(device)

            if inputs.shape[1] == 4 and any("module.modelNIR" in s for s in model.state_dict().keys()):
                ############################
                # Test Implementation of the NIR
                ############################
                # TODO: remove after the merge of Remy branch with no visualization option
                # TODO: or change it to match the reste of the implementation
                inputs_NIR = inputs[:,-1,...] # Need to be change for a more elegant way
                inputs_NIR.unsqueeze_(1) # add a channel to get [:, 1, :, :]
                inputs = inputs[:,:-1, ...] # Need to be change                 
                inputs = [inputs, inputs_NIR]
                ############################
                # Test Implementation of the NIR
                ############################

            # forward
            optimizer.zero_grad()
            outputs = model(inputs)
            # added for torchvision models that output an OrderedDict with outputs in 'out' key.
            # More info: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
            if isinstance(outputs, OrderedDict):
                outputs = outputs['out']

            if vis_batch_range and vis_at_train:
                min_vis_batch, max_vis_batch, increment = vis_batch_range
                if batch_index in range(min_vis_batch, max_vis_batch, increment):
                    vis_path = progress_log.parent.joinpath('visualization')
                    if ep_idx == 0:
                        tqdm.write(f'Visualizing on train outputs for batches in range {vis_batch_range}. All images will be saved to {vis_path}\n')
                    vis_from_batch(params, inputs, outputs,
                                   batch_index=batch_index,
                                   vis_path=vis_path,
                                   labels=labels,
                                   dataset='trn',
                                   ep_num=ep_idx+1)

            loss = criterion(outputs, labels)

            train_metrics['loss'].update(loss.item(), batch_size)

            if device.type == 'cuda' and debug:
                res, mem = gpu_stats(device=device.index)
                _tqdm.set_postfix(
                    OrderedDict(
                        trn_loss=f'{train_metrics["loss"].val:.2f}',
                        gpu_perc=f'{res.gpu} %',
                        gpu_RAM=f'{mem.used / (1024 ** 2):.0f}/{mem.total / (1024 ** 2):.0f} MiB',
                        lr=optimizer.param_groups[0]['lr'],
                        img=data['sat_img'].numpy().shape,
                        smpl=data['map_img'].numpy().shape,
                        bs=batch_size,
                        out_vals=np.unique(outputs[0].argmax(dim=0).detach().cpu().numpy())))

            loss.backward()
            optimizer.step()

    scheduler.step()
    if train_metrics["loss"].avg is not None:
        print(f'Training Loss: {train_metrics["loss"].avg:.4f}')
    return train_metrics


def evaluation(eval_loader, model, criterion, num_classes, batch_size, ep_idx, progress_log, vis_params, batch_metrics=None, dataset='val', device=None, debug=False):
    """
    Evaluate the model and return the updated metrics
    :param eval_loader: data loader
    :param model: model to evaluate
    :param criterion: loss criterion
    :param num_classes: number of classes
    :param batch_size: number of samples to process simultaneously
    :param ep_idx: epoch index (for hypertrainer log)
    :param progress_log: progress log file (for hypertrainer log)
    :param batch_metrics: (int) Metrics computed every (int) batches. If left blank, will not perform metrics.
    :param dataset: (str) 'val or 'tst'
    :param device: device used by pytorch (cpu ou cuda)
    :return: (dict) eval_metrics
    """
    eval_metrics = create_metrics_dict(num_classes)
    model.eval()
    vis_at_eval = get_key_def('vis_at_evaluation', vis_params['visualization'], False)
    vis_batch_range = get_key_def('vis_batch_range', vis_params['visualization'], None)

    with tqdm(eval_loader, dynamic_ncols=True, desc=f'Iterating {dataset} batches with {device.type}') as _tqdm:
        for batch_index, data in enumerate(_tqdm):
            progress_log.open('a', buffering=1).write(tsv_line(ep_idx, dataset, batch_index, len(eval_loader), time.time()))

            with torch.no_grad():
                inputs = data['sat_img'].to(device)
                labels = data['map_img'].to(device)
                labels_flatten = flatten_labels(labels)

                if inputs.shape[1] == 4 and any("module.modelNIR" in s for s in model.state_dict().keys()):
                    ############################
                    # Test Implementation of the NIR
                    ############################
                    # TODO: remove after the merge of Remy branch with no visualization option
                    # TODO: or change it to match the reste of the implementation
                    inputs_NIR = inputs[:,-1,...] # Need to be change for a more elegant way
                    inputs_NIR.unsqueeze_(1) # add a channel to get [:, 1, :, :]
                    inputs = inputs[:,:-1, ...] # Need to be change                 
                    inputs = [inputs, inputs_NIR]
                    ############################
                    # Test Implementation of the NIR
                    ############################

                outputs = model(inputs)
                if isinstance(outputs, OrderedDict):
                    outputs = outputs['out']

                if vis_batch_range and vis_at_eval:
                    min_vis_batch, max_vis_batch, increment = vis_batch_range
                    if batch_index in range(min_vis_batch, max_vis_batch, increment):
                        vis_path = progress_log.parent.joinpath('visualization')
                        if ep_idx == 0 and batch_index == min_vis_batch:
                            tqdm.write(f'Visualizing on {dataset} outputs for batches in range {vis_batch_range}. All '
                                       f'images will be saved to {vis_path}\n')
                        vis_from_batch(params, inputs, outputs,
                                       batch_index=batch_index,
                                       vis_path=vis_path,
                                       labels=labels,
                                       dataset=dataset,
                                       ep_num=ep_idx+1)

                outputs_flatten = flatten_outputs(outputs, num_classes)

                loss = criterion(outputs, labels)

                eval_metrics['loss'].update(loss.item(), batch_size)

                if (dataset == 'val') and (batch_metrics is not None):
                    # Compute metrics every n batches. Time consuming.
                    assert batch_metrics <= len(_tqdm), f"Batch_metrics ({batch_metrics} is smaller than batch size " \
                        f"{len(_tqdm)}. Metrics in validation loop won't be computed"
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

                _tqdm.set_postfix(OrderedDict(dataset=dataset, loss=f'{eval_metrics["loss"].avg:.4f}'))

                if debug and device.type == 'cuda':
                    res, mem = gpu_stats(device=device.index)
                    _tqdm.set_postfix(OrderedDict(device=device, gpu_perc=f'{res.gpu} %',
                                                  gpu_RAM=f'{mem.used/(1024**2):.0f}/{mem.total/(1024**2):.0f} MiB'))

    print(f"{dataset} Loss: {eval_metrics['loss'].avg}")
    if batch_metrics is not None:
        print(f"{dataset} precision: {eval_metrics['precision'].avg}")
        print(f"{dataset} recall: {eval_metrics['recall'].avg}")
        print(f"{dataset} fscore: {eval_metrics['fscore'].avg}")
        print(f"{dataset} iou: {eval_metrics['iou'].avg}")

    return eval_metrics


def main(params, config_path):
    """
    Function to train and validate a models for semantic segmentation or classification.
    :param params: (dict) Parameters found in the yaml config file.
    :param config_path: (str) Path to the yaml config file.

    """
    debug = get_key_def('debug_mode', params['global'], False)
    if debug:
        warnings.warn(f'Debug mode activated. Some debug features may mobilize extra disk space and cause delays in execution.')

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    num_classes = params['global']['num_classes']
    task = params['global']['task']
    assert task == 'segmentation', f"The task should be segmentation. The provided value is {task}"
    num_classes_corrected = num_classes + 1  # + 1 for background # FIXME temporary patch for num_classes problem.

    # INSTANTIATE MODEL AND LOAD CHECKPOINT FROM PATH
    model, checkpoint, model_name = net(params, num_classes_corrected)  # pretrained could become a yaml parameter.
    tqdm.write(f'Instantiated {model_name} model with {num_classes_corrected} output channels.\n')
    bucket_name = get_key_def('bucket_name', params['global'])
    data_path = Path(params['global']['data_path'])
    assert data_path.is_dir(), f'Could not locate data path {data_path}'

    # mlflow tracking path + parameters logging
    set_tracking_uri(get_key_def('mlflow_uri', params['global'], default="./mlruns"))
    set_experiment(get_key_def('mlflow_experiment_name', params['global'], default='gdl-training'))
    run_name = get_key_def('mlflow_run_name', params['global'], default='gdl')
    start_run(run_name=run_name)
    log_params(params['training'])
    log_params(params['global'])
    log_params(params['sample'])

    samples_size = params["global"]["samples_size"]
    overlap = params["sample"]["overlap"]
    min_annot_perc = get_key_def('min_annotated_percent', params['sample']['sampling_method'], 0, expected_type=int)
    num_bands = params['global']['number_of_bands']
    samples_folder_name = (f'samples{samples_size}_overlap{overlap}_min-annot{min_annot_perc}_{num_bands}bands'
                           f'_{run_name}')
    samples_folder = data_path.joinpath(samples_folder_name)

    modelname = config_path.stem
    output_path = samples_folder.joinpath('model') / modelname
    if output_path.is_dir():
        output_path = output_path.joinpath(f"_{now}")
    output_path.mkdir(parents=True, exist_ok=False)
    shutil.copy(str(config_path), str(output_path))
    tqdm.write(f'Model and log files will be saved to: {output_path}\n\n')
    batch_size = params['training']['batch_size']

    if bucket_name:
        from utils.aws import download_s3_files
        bucket, bucket_output_path, output_path, data_path = download_s3_files(bucket_name=bucket_name,
                                                                               data_path=data_path,
                                                                               output_path=output_path)

    since = time.time()
    best_loss = 999
    last_vis_epoch = 0

    progress_log = output_path / 'progress.log'
    if not progress_log.exists():
        progress_log.open('w', buffering=1).write(tsv_line('ep_idx', 'phase', 'iter', 'i_p_ep', 'time'))  # Add header

    trn_log = InformationLogger('trn')
    val_log = InformationLogger('val')
    tst_log = InformationLogger('tst')

    num_devices = params['global']['num_gpus']
    assert num_devices is not None and num_devices >= 0, "missing mandatory num gpus parameter"
    # list of GPU devices that are available and unused. If no GPUs, returns empty list
    max_used_ram = get_key_def('max_used_ram', params['global'], 2000)
    max_used_perc = get_key_def('max_used_perc', params['global'], 15)
    get_key_def('debug_mode', params['global'], False)
    lst_device_ids = get_device_ids(
        num_devices, max_used_ram=max_used_ram, max_used_perc=max_used_perc, debug=debug) \
        if torch.cuda.is_available() else []
    num_devices = len(lst_device_ids) if lst_device_ids else 0
    device = torch.device(f'cuda:{lst_device_ids[0]}' if torch.cuda.is_available() and lst_device_ids else 'cpu')
    print(f"Number of cuda devices requested: {params['global']['num_gpus']}. Cuda devices available: {lst_device_ids}\n")
    if num_devices == 1:
        print(f"Using Cuda device {lst_device_ids[0]}\n")
    elif num_devices > 1:
        # TODO: why are we showing indices [1:-1] for lst_device_ids?
        print(f"Using data parallel on devices: {str(lst_device_ids)[1:-1]}. Main device: {lst_device_ids[0]}\n")
        try:  # For HPC when device 0 not available. Error: Invalid device id (in torch/cuda/__init__.py).
            model = nn.DataParallel(model, device_ids=lst_device_ids)  # DataParallel adds prefix 'module.' to state_dict keys
        except AssertionError:
            warnings.warn(f"Unable to use devices {lst_device_ids}. Trying devices {list(range(len(lst_device_ids)))}")
            device = torch.device('cuda:0')
            lst_device_ids = range(len(lst_device_ids))
            model = nn.DataParallel(model,
                                    device_ids=lst_device_ids)  # DataParallel adds prefix 'module.' to state_dict keys

    else:
        warnings.warn(f"No Cuda device available. This process will only run on CPU\n")

    dontcare = get_key_def("ignore_index", params["training"], -1)
    if dontcare == 0:
        warnings.warn("The 'dontcare' value (or 'ignore_index') used in the loss function cannot be zero;"
                      " all valid class indices should be consecutive, and start at 0. The 'dontcare' value"
                      " will be remapped to -1 while loading the dataset, and inside the config from now on.")
        params["training"]["ignore_index"] = -1

    tqdm.write(f'Creating dataloaders from data in {samples_folder}...\n')
    trn_dataloader, val_dataloader, tst_dataloader = create_dataloader(samples_folder=samples_folder,
                                                                       batch_size=batch_size,
                                                                       num_devices=num_devices,
                                                                       params=params)

    tqdm.write(f'Setting model, criterion, optimizer and learning rate scheduler...\n')
    try:  # For HPC when device 0 not available. Error: Cuda invalid device ordinal.
        model.to(device)
    except RuntimeError:
        warnings.warn(f"Unable to use device. Trying device 0...\n")
        device = torch.device(f'cuda:0' if torch.cuda.is_available() and lst_device_ids else 'cpu')
        model.to(device)
    model, criterion, optimizer, lr_scheduler = set_hyperparameters(params,
                                                                    num_classes_corrected,
                                                                    model,
                                                                    checkpoint,
                                                                    dontcare)

    criterion = criterion.to(device)

    filename = output_path.joinpath('checkpoint.pth.tar')

    # VISUALIZATION: generate pngs of inputs, labels and outputs
    vis_batch_range = get_key_def('vis_batch_range', params['visualization'], None)
    if vis_batch_range is not None:
        # Make sure user-provided range is a tuple with 3 integers (start, finish, increment). Check once for all visualization tasks.
        assert isinstance(vis_batch_range, list) and len(vis_batch_range) == 3 and all(isinstance(x, int) for x in vis_batch_range)
        vis_at_init_dataset = get_key_def('vis_at_init_dataset', params['visualization'], 'val')

        # Visualization at initialization. Visualize batch range before first eopch.
        if get_key_def('vis_at_init', params['visualization'], False):
            tqdm.write(f'Visualizing initialized model on batch range {vis_batch_range} from {vis_at_init_dataset} dataset...\n')
            vis_from_dataloader(params=params,
                                eval_loader=val_dataloader if vis_at_init_dataset == 'val' else tst_dataloader,
                                model=model,
                                ep_num=0,
                                output_path=output_path,
                                dataset=vis_at_init_dataset,
                                device=device,
                                vis_batch_range=vis_batch_range)

    for epoch in range(0, params['training']['num_epochs']):
        print(f'\nEpoch {epoch}/{params["training"]["num_epochs"] - 1}\n{"-" * 20}')

        trn_report = train(train_loader=trn_dataloader,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           scheduler=lr_scheduler,
                           num_classes=num_classes_corrected,
                           batch_size=batch_size,
                           ep_idx=epoch,
                           progress_log=progress_log,
                           vis_params=params,
                           device=device,
                           debug=debug)
        trn_log.add_values(trn_report, epoch, ignore=['precision', 'recall', 'fscore', 'iou'])

        val_report = evaluation(eval_loader=val_dataloader,
                                model=model,
                                criterion=criterion,
                                num_classes=num_classes_corrected,
                                batch_size=batch_size,
                                ep_idx=epoch,
                                progress_log=progress_log,
                                vis_params=params,
                                batch_metrics=params['training']['batch_metrics'],
                                dataset='val',
                                device=device,
                                debug=debug)
        val_loss = val_report['loss'].avg
        if params['training']['batch_metrics'] is not None:
            val_log.add_values(val_report, epoch)
        else:
            val_log.add_values(val_report, epoch, ignore=['precision', 'recall', 'fscore', 'iou'])

        if val_loss < best_loss:
            tqdm.write("save checkpoint\n")
            best_loss = val_loss
            # More info: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-torch-nn-dataparallel-models
            state_dict = model.module.state_dict() if num_devices > 1 else model.state_dict()
            torch.save({'epoch': epoch,
                        'arch': model_name,
                        'model': state_dict,
                        'best_loss': best_loss,
                        'optimizer': optimizer.state_dict()}, filename)
            if epoch == 0:
                log_artifact(filename)
            if bucket_name:
                bucket_filename = bucket_output_path.joinpath('checkpoint.pth.tar')
                bucket.upload_file(filename, bucket_filename)

            # VISUALIZATION: generate png of test samples, labels and outputs for visualisation to follow training performance
            vis_at_checkpoint = get_key_def('vis_at_checkpoint', params['visualization'], False)
            ep_vis_min_thresh = get_key_def('vis_at_ckpt_min_ep_diff', params['visualization'], 4)
            vis_at_ckpt_dataset = get_key_def('vis_at_ckpt_dataset', params['visualization'], 'val')
            if vis_batch_range is not None and vis_at_checkpoint and epoch - last_vis_epoch >= ep_vis_min_thresh:
                if last_vis_epoch == 0:
                    tqdm.write(f'Visualizing with {vis_at_ckpt_dataset} dataset samples on checkpointed model for'
                               f'batches in range {vis_batch_range}')
                vis_from_dataloader(params=params,
                                    eval_loader=val_dataloader if vis_at_ckpt_dataset == 'val' else tst_dataloader,
                                    model=model,
                                    ep_num=epoch+1,
                                    output_path=output_path,
                                    dataset=vis_at_ckpt_dataset,
                                    device=device,
                                    vis_batch_range=vis_batch_range)
                last_vis_epoch = epoch

        if bucket_name:
            save_logs_to_bucket(bucket, bucket_output_path, output_path, now, params['training']['batch_metrics'])

        cur_elapsed = time.time() - since
        print(f'Current elapsed time {cur_elapsed // 60:.0f}m {cur_elapsed % 60:.0f}s')

    # load checkpoint model and evaluate it on test dataset.
    if int(params['training']['num_epochs']) > 0:   # if num_epochs is set to 0, model is loaded to evaluate on test set
        checkpoint = load_checkpoint(filename)
        model, _ = load_from_checkpoint(checkpoint, model)

    if tst_dataloader:
        tst_report = evaluation(eval_loader=tst_dataloader,
                                model=model,
                                criterion=criterion,
                                num_classes=num_classes_corrected,
                                batch_size=batch_size,
                                ep_idx=params['training']['num_epochs'],
                                progress_log=progress_log,
                                vis_params=params,
                                batch_metrics=params['training']['batch_metrics'],
                                dataset='tst',
                                device=device)
        tst_log.add_values(tst_report, params['training']['num_epochs'])

        if bucket_name:
            bucket_filename = bucket_output_path.joinpath('last_epoch.pth.tar')
            bucket.upload_file("output.txt", bucket_output_path.joinpath(f"Logs/{now}_output.txt"))
            bucket.upload_file(filename, bucket_filename)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    print(f'Start\n')
    parser = argparse.ArgumentParser(description='Training execution')
    parser.add_argument('param_file', metavar='DIR',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    config_path = Path(args.param_file)
    params = read_parameters(args.param_file)

    # Limit of the NIR implementation TODO: Update after each version
    modalities = None if 'modalities' not in params['global'] else params['global']['modalities']
    if 'deeplabv3' not in params['global']['model_name'] and modalities is 'RGBN':
        print(
            '\n The NIR modality will only be concatenate at the begining,' /
            ' the implementation of the concatenation point is only available' /
            ' for the deeplabv3 model for now. \n More will follow on demande.\n'
             )

    main(params, config_path)
    print('End of training')

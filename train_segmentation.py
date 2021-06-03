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

from torch.utils.data import DataLoader
from PIL import Image
from sklearn.utils import compute_sample_weight
from utils import augmentation as aug, create_dataset
from utils.logger import InformationLogger, save_logs_to_bucket, tsv_line
from utils.metrics import report_classification, create_metrics_dict, iou
from models.model_choice import net, load_checkpoint
from utils.utils import load_from_checkpoint, get_device_ids, gpu_stats, get_key_def, get_git_hash
from utils.visualization import vis_from_batch
from utils.readers import read_parameters
from mlflow import log_params, set_tracking_uri, set_experiment, log_artifact, start_run


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
    :param samples_folder: path to folder containting .hdf5 files if task is segmentation
    :param batch_size: (int) batch size
    :param num_devices: (int) number of GPUs used
    :param params: (dict) Parameters found in the yaml config file.
    :return: trn_dataloader, val_dataloader, tst_dataloader
    """
    debug = get_key_def('debug_mode', params['global'], False)
    dontcare_val = get_key_def("ignore_index", params["training"], -1)

    assert samples_folder.is_dir(), f'Could not locate: {samples_folder}'
    assert len([f for f in samples_folder.glob('**/*.hdf5')]) >= 1, f"Couldn't locate .hdf5 files in {samples_folder}"
    num_samples, samples_weight = get_num_samples(samples_path=samples_folder, params=params)
    assert num_samples['trn'] >= batch_size and num_samples['val'] >= batch_size, f"Number of samples in .hdf5 files is less than batch size"
    print(f"Number of samples : {num_samples}\n")
    meta_map = get_key_def("meta_map", params["global"], {})
    num_bands = get_key_def("number_of_bands", params["global"], {})
    if not meta_map:
        dataset_constr = create_dataset.SegmentationDataset
    else:
        dataset_constr = functools.partial(create_dataset.MetaSegmentationDataset, meta_map=meta_map)
    datasets = []

    for subset in ["trn", "val", "tst"]:

        datasets.append(dataset_constr(samples_folder, subset, num_bands,
                                       max_sample_count=num_samples[subset],
                                       dontcare=dontcare_val,
                                       radiom_transform=aug.compose_transforms(params, subset, type='radiometric'),
                                       geom_transform=aug.compose_transforms(params, subset, type='geometric',
                                                                             ignore_index=dontcare_val),
                                       totensor_transform=aug.compose_transforms(params, subset, type='totensor'),
                                       params=params,
                                       debug=debug))
    trn_dataset, val_dataset, tst_dataset = datasets

    # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
    num_workers = num_devices * 4 if num_devices > 1 else 4

    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),
                                                             len(samples_weight))

    trn_dataloader = DataLoader(trn_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler,
                                drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=num_workers, shuffle=False,
                                drop_last=True)
    tst_dataloader = DataLoader(tst_dataset, batch_size=1, num_workers=num_workers, shuffle=False,
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
    weights = []
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

    return num_samples, samples_weight


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


def train(train_loader,model,criterion,optimizer,scheduler,num_classes,batch_size,ep_idx,progress_log,vis_params,device,debug=False):
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

        # huh? : NIR stuff ?
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
                _tqdm.set_postfix(OrderedDict(trn_loss=f'{train_metrics["loss"].val:.2f}',
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
    params['global']['git_hash'] = get_git_hash()
    debug = get_key_def('debug_mode', params['global'], False)
    if debug:
        warnings.warn(f'Debug mode activated. Some debug features may mobilize extra disk space and cause delays in execution.')

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    num_classes = params['global']['num_classes']
    task = params['global']['task']
    assert task == 'segmentation', f"The task should be segmentation. The provided value is {task}"
    num_classes_corrected = num_classes + 1  # + 1 for background # FIXME temporary patch for num_classes problem.

    data_path = Path(params['global']['data_path'])
    assert data_path.is_dir(), f'Could not locate data path {data_path}'
    samples_size = params["global"]["samples_size"]
    overlap = params["sample"]["overlap"]
    min_annot_perc = get_key_def('min_annotated_percent', params['sample']['sampling_method'], 0, expected_type=int)
    num_bands = params['global']['number_of_bands']
    experiment_name = get_key_def('mlflow_experiment_name', params['global'], default='gdl-training')
    run_name = get_key_def('mlflow_run_name', params['global'], default='gdl')
    samples_folder_name = (f'samples{samples_size}_overlap{overlap}_min-annot{min_annot_perc}_{num_bands}bands'
                           f'_{experiment_name}')
    samples_folder = data_path.joinpath(samples_folder_name)
    batch_size = params['training']['batch_size']
    num_devices = params['global']['num_gpus']
    # # list of GPU devices that are available and unused. If no GPUs, returns empty list
    max_used_ram = get_key_def('max_used_ram', params['global'], 2000, expected_type=int)
    max_used_perc = get_key_def('max_used_perc', params['global'], 15, expected_type=int)
    lst_device_ids = get_device_ids(
        num_devices, max_used_ram=max_used_ram, max_used_perc=max_used_perc, debug=debug) \
        if torch.cuda.is_available() else []
    num_devices = len(lst_device_ids) if lst_device_ids else 0
    device = torch.device(f'cuda:{lst_device_ids[0]}' if torch.cuda.is_available() and lst_device_ids else 'cpu')

    tqdm.write(f'Creating dataloaders from data in {samples_folder}...\n')
    trn_dataloader, val_dataloader, tst_dataloader = create_dataloader(samples_folder=samples_folder,
                                                                       batch_size=batch_size,
                                                                       num_devices=num_devices,
                                                                       params=params)
# Init Model ___________________________________________________________________________________________________________s
    # INSTANTIATE MODEL AND LOAD CHECKPOINT FROM PATH
    model, model_name, criterion, optimizer, lr_scheduler = net(params, num_classes_corrected)  # pretrained could become a yaml parameter.
    tqdm.write(f'Instantiated {model_name} model with {num_classes_corrected} output channels.\n')
    bucket_name = get_key_def('bucket_name', params['global'])

# MLFlow________________________________________________________________________________________________________________
    # mlflow tracking path + parameters logging
    set_tracking_uri(get_key_def('mlflow_uri', params['global'], default="./mlruns"))
    set_experiment(get_key_def('mlflow_experiment_name', params['global'], default='gdl-training'))
    start_run(run_name=run_name)
    log_params(params['training'])
    log_params(params['global'])
    log_params(params['sample'])

# Output Path __________________________________________________________________________________________________________
    modelname = config_path.stem
    output_path = samples_folder.joinpath('model') / modelname
    if output_path.is_dir():
        output_path = output_path.joinpath(f"_{now}")
    output_path.mkdir(parents=True, exist_ok=False)
    shutil.copy(str(config_path), str(output_path))
    tqdm.write(f'Model and log files will be saved to: {output_path}\n\n')

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
    filename = output_path.joinpath('checkpoint.pth.tar')

# Visualization (input) ________________________________________________________________________________________________________
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

# TRAINING _____________________________________________________________________________________________________________
    print('Starting Training..................................')
    for epoch in range(0, params['training']['num_epochs']):
        print(f'\n\tEpoch {epoch}/{params["training"]["num_epochs"] - 1}\n{"-" * 20}')

        print('\t training...')
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

        print('\t evaluating...')
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
                        'params': params,
                        'model': state_dict,
                        'best_loss': best_loss,
                        'optimizer': optimizer.state_dict()}, filename)
            if epoch == 0:
                log_artifact(filename)
            if bucket_name:
                bucket_filename = bucket_output_path.joinpath('checkpoint.pth.tar')
                bucket.upload_file(filename, bucket_filename)

    # Visualization (output) ________________________________________________________________________________________________________
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

# Testing ______________________________________________________________________________________________________________
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
    parser.add_argument('param_file', help='Path to training parameters stored in yaml')
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

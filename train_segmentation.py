from collections import OrderedDict
from datetime import datetime
from numbers import Number
from pathlib import Path
import shutil
import time
from typing import Sequence

from hydra.utils import to_absolute_path, instantiate
import numpy as np
from omegaconf import DictConfig
import rasterio
from sklearn.utils import compute_sample_weight
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model_choice import read_checkpoint, define_model, adapt_checkpoint_to_dp_model
from tiling_segmentation import Tiler
from utils import augmentation as aug
from dataset import create_dataset
from utils.logger import InformationLogger, tsv_line, get_logger, set_tracker
from utils.loss import verify_weights, define_loss
from utils.metrics import create_metrics_dict, calculate_batch_metrics
from utils.utils import gpu_stats, get_key_def, get_device_ids, set_device
from utils.visualization import vis_from_batch
# Set the logging file
logging = get_logger(__name__)  # import logging


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


def create_dataloader(patches_folder: Path,
                      batch_size: int,
                      gpu_devices_dict: dict,
                      sample_size: int,
                      dontcare_val: int,
                      crop_size: int,
                      num_bands: int,
                      min_annot_perc: int,
                      attr_vals: Sequence,
                      scale: Sequence,
                      cfg: dict,
                      eval_batch_size: int = None,
                      dontcare2backgr: bool = False,
                      compute_sampler_weights: bool = False,
                      debug: bool = False):
    """
    Function to create dataloader objects for training, validation and test datasets.
    @param patches_folder: path to folder containting patches
    @param batch_size: (int) batch size
    @param gpu_devices_dict: (dict) dictionary where each key contains an available GPU with its ram info stored as value
    @param sample_size: (int) size of patches (used to evaluate eval batch-size)
    @param dontcare_val: (int) value in label to be ignored during loss calculation
    @param crop_size: (int) size of one side of the square crop performed on original patch during training
    @param num_bands: (int) number of bands in imagery
    @param min_annot_perc: (int) minimum proportion of ground truth containing non-background information
    @param attr_vals: (Sequence)
    @param scale: (List) imagery data will be scaled to this min and max value (ex.: 0 to 1)
    @param cfg: (dict) Parameters found in the yaml config file.
    @param eval_batch_size: (int) Batch size for evaluation (val and test). Optional, calculated automatically if omitted
    @param dontcare2backgr: (bool) if True, all dontcare values in label will be replaced with 0 (background value)
                            before training
    @param compute_sampler_weights: (bool)
        if True, weights will be computed from dataset patches to oversample the minority class(es) and undersample
        the majority class(es) during training.
    :return: trn_dataloader, val_dataloader, tst_dataloader
    """
    if not patches_folder.is_dir():
        raise FileNotFoundError(f'Could not locate: {patches_folder}')
    experiment_name = patches_folder.stem
    if not len([f for f in patches_folder.glob('*.csv')]) >= 1:
        raise FileNotFoundError(f"Couldn't locate csv file(s) containing list of training data in {patches_folder}")
    num_patches, patches_weight = get_num_patches(patches_path=patches_folder,
                                                  params=cfg,
                                                  min_annot_perc=min_annot_perc,
                                                  attr_vals=attr_vals,
                                                  experiment_name=experiment_name,
                                                  compute_sampler_weights=compute_sampler_weights)
    if not num_patches['trn'] >= batch_size and num_patches['val'] >= batch_size:
        raise ValueError(f"Number of patches is smaller than batch size")
    logging.info(f"Number of patches : {num_patches}\n")
    dataset_constr = create_dataset.SegmentationDataset
    datasets = []

    for subset in ["trn", "val", "tst"]:
        # TODO: should user point to the paths of these csvs directly?
        dataset_file, _ = Tiler.make_dataset_file_name(experiment_name, min_annot_perc, subset, attr_vals)
        dataset_filepath = patches_folder / dataset_file
        datasets.append(dataset_constr(dataset_filepath, subset, num_bands,
                                       max_sample_count=num_patches[subset],
                                       radiom_transform=aug.compose_transforms(params=cfg,
                                                                               dataset=subset,
                                                                               aug_type='radiometric'),
                                       geom_transform=aug.compose_transforms(params=cfg,
                                                                             dataset=subset,
                                                                             aug_type='geometric',
                                                                             dontcare=dontcare_val,
                                                                             crop_size=crop_size),
                                       totensor_transform=aug.compose_transforms(params=cfg,
                                                                                 dataset=subset,
                                                                                 scale=scale,
                                                                                 dontcare2backgr=dontcare2backgr,
                                                                                 dontcare=dontcare_val,
                                                                                 aug_type='totensor'),
                                       debug=debug))
    trn_dataset, val_dataset, tst_dataset = datasets

    # Number of workers
    if cfg.training.num_workers:
        num_workers = cfg.training.num_workers
    else:  # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
        num_workers = len(gpu_devices_dict.keys()) * 4 if len(gpu_devices_dict.keys()) > 1 else 4

    patches_weight = torch.from_numpy(patches_weight)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(patches_weight.type('torch.DoubleTensor'),
                                                             len(patches_weight))

    if gpu_devices_dict and not eval_batch_size:
        max_pix_per_mb_gpu = 280
        eval_batch_size = calc_eval_batchsize(gpu_devices_dict, batch_size, sample_size, max_pix_per_mb_gpu)
    elif not eval_batch_size:
        eval_batch_size = batch_size

    trn_dataloader = DataLoader(trn_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler,
                                drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, num_workers=num_workers, shuffle=False,
                                drop_last=True)
    tst_dataloader = DataLoader(tst_dataset, batch_size=eval_batch_size, num_workers=num_workers, shuffle=False,
                                drop_last=True) if num_patches['tst'] > 0 else None

    if len(trn_dataloader) == 0 or len(val_dataloader) == 0:
        raise ValueError(f"\nTrain and validation dataloader should contain at least one data item."
                         f"\nTrain dataloader's length: {len(trn_dataloader)}"
                         f"\nVal dataloader's length: {len(val_dataloader)}")

    return trn_dataloader, val_dataloader, tst_dataloader


def calc_eval_batchsize(gpu_devices_dict: dict, batch_size: int, sample_size: int, max_pix_per_mb_gpu: int = 280):
    """
    Calculate maximum batch size that could fit on GPU during evaluation based on thumb rule with harcoded
    "pixels per MB of GPU RAM" as threshold. The batch size often needs to be smaller if crop is applied during training
    @param gpu_devices_dict: dictionary containing info on GPU devices as returned by lst_device_ids (utils.py)
    @param batch_size: batch size for training
    @param sample_size: size of patches
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


def get_num_patches(
        patches_path,
        params,
        min_annot_perc,
        attr_vals,
        experiment_name:str,
        compute_sampler_weights=False
):
    """
    Function to retrieve number of patches, either from config file or directly from csv listing all created patches.
    @param patches_path: (str) Path to patches folder
    @param params: (dict) Parameters found in the yaml config file.
    @param min_annot_perc: (int) minimum annotated percentage
    @param attr_vals: (list) attribute values to keep from source ground truth
    @param experiment_name: (str) experiment name
    @param compute_sampler_weights: (bool)
        if True, weights will be computed from dataset patches to oversample the minority class(es) and undersample
        the majority class(es) during training.
    :return: (dict) number of patches for trn, val and tst.
    """
    num_patches = {'trn': 0, 'val': 0, 'tst': 0}
    weights = []
    patches_weight = None
    for dataset in ['trn', 'val', 'tst']:
        dataset_file, _ = Tiler.make_dataset_file_name(experiment_name, min_annot_perc, dataset, attr_vals)
        dataset_filepath = patches_path / dataset_file
        if not dataset_filepath.is_file() and dataset == 'tst':
            num_patches[dataset] = 0
            logging.warning(f"No test set. File not found: {dataset_filepath}")
            continue

        if get_key_def(f"num_{dataset}_patches", params['training'], None) is not None:
            num_patches[dataset] = params['training'][f"num_{dataset}_patches"]

            with open(dataset_filepath, 'r') as datafile:
                file_num_patches = len(datafile.readlines())
            if num_patches[dataset] > file_num_patches:
                raise IndexError(f"The number of training patches in the configuration file ({num_patches[dataset]}) "
                                 f"exceeds the number of patches in the training dataset ({file_num_patches}).")
        else:
            with open(dataset_filepath, 'r') as datafile:
                num_patches[dataset] = len(datafile.readlines())

        with open(dataset_filepath, 'r') as datafile:
            datalist = datafile.readlines()
            if dataset == 'trn':
                if not compute_sampler_weights:
                    patches_weight = np.ones(num_patches[dataset])
                else:
                    dontcare = get_key_def("ignore_index", params['dataset'], default=-1)
                    for x in tqdm(range(num_patches[dataset]), desc="Computing sample weights"):
                        label_file = datalist[x].split(';')[1]
                        with rasterio.open(label_file, 'r') as label_handle:
                            label = label_handle.read()
                        label = np.where(label == dontcare, 0, label)
                        unique_labels = np.unique(label)
                        weights.append(''.join([str(int(i)) for i in unique_labels]))
                        patches_weight = compute_sample_weight('balanced', weights)
            logging.debug(patches_weight.shape)
            logging.debug(np.unique(patches_weight))

    return num_patches, patches_weight


def vis_from_dataloader(vis_params,
                        eval_loader,
                        model,
                        ep_num,
                        output_path,
                        dataset='',
                        scale=None,
                        device=None,
                        vis_batch_range=None):
    """
    Use a model and dataloader to provide outputs that can then be sent to vis_from_batch function to visualize performances of model, for example.
    :param vis_params: (dict) Parameters found in the yaml config file useful for visualization
    :param eval_loader: data loader
    :param model: model to evaluate
    :param ep_num: epoch index (for file naming purposes)
    :param dataset: (str) 'val or 'tst'
    :param device: device used by pytorch (cpu ou cuda)
    :param vis_batch_range: (int) max number of patches to perform visualization on

    :return:
    """
    vis_path = output_path.joinpath(f'visualization')
    logging.info(f'Visualization figures will be saved to {vis_path}\n')
    min_vis_batch, max_vis_batch, increment = vis_batch_range

    model.eval()
    with tqdm(eval_loader, dynamic_ncols=True) as _tqdm:
        for batch_index, data in enumerate(_tqdm):
            if vis_batch_range is not None and batch_index in range(min_vis_batch, max_vis_batch, increment):
                with torch.no_grad():
                    inputs = data["image"].to(device)
                    labels = data["mask"].to(device)

                    outputs = model(inputs)
                    if isinstance(outputs, OrderedDict):
                        outputs = outputs['out']

                    vis_from_batch(vis_params, inputs, outputs,
                                   batch_index=batch_index,
                                   vis_path=vis_path,
                                   labels=labels,
                                   dataset=dataset,
                                   ep_num=ep_num,
                                   scale=scale)
    logging.info(f'Saved visualization figures.\n')


def training(train_loader,
          model,
          criterion,
          optimizer,
          scheduler,
          num_classes,
          batch_size,
          ep_idx,
          progress_log,
          device,
          scale,
          vis_params,
          debug=False):
    """
    Train the model and return the metrics of the training epoch

    :param train_loader: training data loader
    :param model: model to train
    :param criterion: loss criterion
    :param optimizer: optimizer to use
    :param scheduler: learning rate scheduler
    :param num_classes: number of classes
    :param batch_size: number of patches to process simultaneously
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

    for batch_index, data in enumerate(tqdm(train_loader, desc=f'Iterating train batches with {device.type}')):
        progress_log.open('a', buffering=1).write(tsv_line(ep_idx, 'trn', batch_index, len(train_loader), time.time()))

        inputs = data["image"].to(device)
        labels = data["mask"].to(device)

        # forward
        optimizer.zero_grad()
        outputs = model(inputs)
        # added for torchvision models that output an OrderedDict with outputs in 'out' key.
        # More info: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
        if isinstance(outputs, OrderedDict):
            outputs = outputs['out']

        # vis_batch_range: range of batches to perform visualization on. see README.md for more info.
        # vis_at_eval: (bool) if True, will perform visualization at eval time, as long as vis_batch_range is valid
        if vis_params['vis_batch_range'] and vis_params['vis_at_train']:
            min_vis_batch, max_vis_batch, increment = vis_params['vis_batch_range']
            if batch_index in range(min_vis_batch, max_vis_batch, increment):
                vis_path = progress_log.parent.joinpath('visualization')
                if ep_idx == 0:
                    logging.info(f'Visualizing on train outputs for batches in range {vis_params["vis_batch_range"]}. '
                                 f'All images will be saved to {vis_path}\n')
                vis_from_batch(vis_params, inputs, outputs,
                               batch_index=batch_index,
                               vis_path=vis_path,
                               labels=labels,
                               dataset='trn',
                               ep_num=ep_idx + 1,
                               scale=scale)

        loss = criterion(outputs, labels) if num_classes > 1 else criterion(outputs, labels.unsqueeze(1).float())

        train_metrics['loss'].update(loss.item(), batch_size)

        if device.type == 'cuda' and debug:
            res, mem = gpu_stats(device=device.index)
            logging.debug(OrderedDict(trn_loss=f"{train_metrics['loss'].average():.2f}",
                                      gpu_perc=f"{res['gpu']} %",
                                      gpu_RAM=f"{mem['used'] / (1024 ** 2):.0f}/{mem['total'] / (1024 ** 2):.0f} MiB",
                                      lr=optimizer.param_groups[0]['lr'],
                                      img=data["image"].numpy().shape,
                                      smpl=data["mask"].numpy().shape,
                                      bs=batch_size,
                                      out_vals=np.unique(outputs[0].argmax(dim=0).detach().cpu().numpy()),
                                      gt_vals=np.unique(labels[0].detach().cpu().numpy())))

        loss.backward()
        optimizer.step()

    scheduler.step()
    # if train_metrics["loss"].avg is not None:
    #     logging.info(f'Training Loss: {train_metrics["loss"].avg:.4f}')
    return train_metrics


def evaluation(eval_loader,
               model,
               criterion,
               num_classes,
               batch_size,
               ep_idx,
               progress_log,
               scale,
               vis_params,
               batch_metrics=None,
               dataset='val',
               device=None,
               debug=False
               ):
    """
    Evaluate the model and return the updated metrics
    :param eval_loader: data loader
    :param model: model to evaluate
    :param criterion: loss criterion
    :param num_classes: number of classes
    :param batch_size: number of patches to process simultaneously
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

    for batch_index, data in enumerate(tqdm(eval_loader, dynamic_ncols=True, desc=f'Iterating {dataset} '
                                                                                  f'batches with {device.type}')):
        progress_log.open('a', buffering=1).write(tsv_line(ep_idx, dataset, batch_index, len(eval_loader), time.time()))

        with torch.no_grad():
            inputs = data["image"].to(device)
            labels = data["mask"].to(device)
            outputs = model(inputs)

            if isinstance(outputs, OrderedDict):
                outputs = outputs['out']

            # vis_batch_range: range of batches to perform visualization on. see README.md for more info.
            # vis_at_eval: (bool) if True, will perform visualization at eval time, as long as vis_batch_range is valid
            if vis_params['vis_batch_range'] and vis_params['vis_at_eval']:
                min_vis_batch, max_vis_batch, increment = vis_params['vis_batch_range']
                if batch_index in range(min_vis_batch, max_vis_batch, increment):
                    vis_path = progress_log.parent.joinpath('visualization')
                    if ep_idx == 0 and batch_index == min_vis_batch:
                        logging.info(f'\nVisualizing on {dataset} outputs for batches in range '
                                     f'{vis_params["vis_batch_range"]} images will be saved to {vis_path}\n')
                    vis_from_batch(vis_params, inputs, outputs,
                                   batch_index=batch_index,
                                   vis_path=vis_path,
                                   labels=labels,
                                   dataset=dataset,
                                   ep_num=ep_idx + 1,
                                   scale=scale)

            loss = criterion(outputs, labels.unsqueeze(1).float())

            eval_metrics['loss'].update(loss.item(), batch_size)

            if (dataset == 'val') and (batch_metrics is not None):
                # Compute metrics every n batches. Time-consuming.
                if not batch_metrics <= len(eval_loader):
                    logging.error(f"\nBatch_metrics ({batch_metrics}) is smaller than batch size "
                                  f"{len(eval_loader)}. Metrics in validation loop won't be computed")
                if (batch_index + 1) % batch_metrics == 0:  # +1 to skip val loop at very beginning
                    eval_metrics = calculate_batch_metrics(
                        predictions=outputs,
                        gts=labels,
                        n_classes=num_classes,
                        metric_dict=eval_metrics
                    )

            elif dataset == 'tst':
                eval_metrics = calculate_batch_metrics(
                    predictions=outputs,
                    gts=labels,
                    n_classes=num_classes,
                    metric_dict=eval_metrics
                )

            logging.debug(OrderedDict(dataset=dataset, loss=f'{eval_metrics["loss"].avg:.4f}'))

            if debug and device.type == 'cuda':
                res, mem = gpu_stats(device=device.index)
                logging.debug(OrderedDict(
                    device=device, gpu_perc=f"{res['gpu']} %",
                    gpu_RAM=f"{mem['used']/(1024**2):.0f}/{mem['total']/(1024**2):.0f} MiB"
                ))

    if eval_metrics['loss'].average():
        logging.info(f"\n{dataset} Loss: {eval_metrics['loss'].average():.4f}")
    if batch_metrics is not None or dataset == 'tst':
        logging.info(f"\n{dataset} precision: {eval_metrics['precision'].average():.4f}")
        logging.info(f"\n{dataset} recall: {eval_metrics['recall'].average():.4f}")
        logging.info(f"\n{dataset} fscore: {eval_metrics['fscore'].average():.4f}")
        logging.info(f"\n{dataset} iou: {eval_metrics['iou'].average():.4f}")

    return eval_metrics


def train(cfg: DictConfig) -> None:
    """
    Function to train and validate a model for semantic segmentation.

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
    :param cfg: (dict) Parameters found in the yaml config file.
    """
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # MANDATORY PARAMETERS
    class_keys = len(get_key_def('classes_dict', cfg['dataset']).keys())
    num_classes = class_keys if class_keys == 1 else class_keys + 1  # +1 for background(multiclass mode)
    modalities = get_key_def('bands', cfg['dataset'], default=("red", "blue", "green"), expected_type=Sequence)
    num_bands = len(modalities)
    batch_size = get_key_def('batch_size', cfg['training'], expected_type=int)
    eval_batch_size = get_key_def('eval_batch_size', cfg['training'], expected_type=int, default=batch_size)
    num_epochs = get_key_def('max_epochs', cfg['training'], expected_type=int)

    # OPTIONAL PARAMETERS
    debug = get_key_def('debug', cfg)
    task = get_key_def('task',  cfg['general'], default='segmentation')
    dontcare_val = get_key_def("ignore_index", cfg['dataset'], default=-1)
    scale = get_key_def('scale_data', cfg['augmentation'], default=[0, 1])
    batch_metrics = get_key_def('batch_metrics', cfg['training'], default=None)
    crop_size = get_key_def('crop_size', cfg['augmentation'], default=None)
    compute_sampler_weights = get_key_def('compute_sampler_weights', cfg['training'], default=False, expected_type=bool)

    # MODEL PARAMETERS
    checkpoint_stack = [""]
    class_weights = get_key_def('class_weights', cfg['dataset'], default=None)
    if cfg.loss.is_binary and not num_classes == 1:
        raise ValueError(f"Parameter mismatch: a binary loss was chosen for a {num_classes}-class task")
    elif not cfg.loss.is_binary and num_classes == 1:
        raise ValueError(f"Parameter mismatch: a multiclass loss was chosen for a 1-class (binary) task")
    del cfg.loss.is_binary  # prevent exception at instantiation
    optimizer = get_key_def('optimizer_name', cfg['optimizer'], default='adam', expected_type=str)  # TODO change something to call the function
    pretrained = get_key_def('pretrained', cfg['model'], default=True, expected_type=(bool, str))
    train_state_dict_path = get_key_def('state_dict_path', cfg['training'], default=None, expected_type=str)
    state_dict_strict = get_key_def('state_dict_strict_load', cfg['training'], default=True, expected_type=bool)
    dropout_prob = get_key_def('factor', cfg['scheduler']['params'], default=None, expected_type=float)
    # if error
    if train_state_dict_path and not Path(train_state_dict_path).is_file():
        raise logging.critical(
            FileNotFoundError(f'\nCould not locate pretrained checkpoint for training: {train_state_dict_path}')
        )
    if class_weights:
        verify_weights(num_classes, class_weights)
    # Read the concatenation point if requested model is deeplabv3 dualhead
    conc_point = get_key_def('conc_point', cfg['model'], None)
    step_size = get_key_def('step_size', cfg['scheduler']['params'], default=4, expected_type=int)
    gamma = get_key_def('gamma', cfg['scheduler']['params'], default=0.9, expected_type=float)

    # GPU PARAMETERS
    num_devices = get_key_def('num_gpus', cfg['training'], default=0)
    if num_devices and not num_devices >= 0:
        raise ValueError("\nMissing mandatory num gpus parameter")
    max_used_ram = get_key_def('max_used_ram', cfg['training'], default=15)
    max_used_perc = get_key_def('max_used_perc', cfg['training'], default=15)

    # LOGGING PARAMETERS
    run_name = get_key_def(['tracker', 'run_name'], cfg, default='gdl')
    tracker_uri = get_key_def(['tracker', 'uri'], cfg, default=None, expected_type=str)
    experiment_name = get_key_def('project_name', cfg['general'], default='gdl-training')

    # PARAMETERS FOR PATCHES
    patches_size = get_key_def("patch_size", cfg['tiling'], expected_type=int, default=256)
    min_annot_perc = get_key_def('min_annot_perc', cfg['tiling'], expected_type=Number, default=0)
    attr_vals = get_key_def('attribute_values', cfg['dataset'], None, expected_type=(Sequence, int))

    data_path = get_key_def('raw_data_dir', cfg['dataset'], to_path=True, validate_path_exists=True)
    tiling_root_dir = get_key_def('tiling_data_dir', cfg['tiling'], default=data_path, to_path=True,
                                 validate_path_exists=True)
    logging.info("\nThe tiling directory used '{}'".format(tiling_root_dir))

    tiling_dir = tiling_root_dir / experiment_name

    # visualization parameters
    vis_at_train = get_key_def('vis_at_train', cfg['visualization'], default=False)
    vis_at_eval = get_key_def('vis_at_evaluation', cfg['visualization'], default=False)
    vis_batch_range = get_key_def('vis_batch_range', cfg['visualization'], default=None)
    vis_at_checkpoint = get_key_def('vis_at_checkpoint', cfg['visualization'], default=False)
    ep_vis_min_thresh = get_key_def('vis_at_ckpt_min_ep_diff', cfg['visualization'], default=1)
    vis_at_ckpt_dataset = get_key_def('vis_at_ckpt_dataset', cfg['visualization'], 'val')
    colormap_file = get_key_def('colormap_file', cfg['visualization'], None)
    heatmaps = get_key_def('heatmaps', cfg['visualization'], False)
    heatmaps_inf = get_key_def('heatmaps', cfg['inference'], False)
    grid = get_key_def('grid', cfg['visualization'], False)
    mean = get_key_def('mean', cfg['augmentation']['normalization'])
    std = get_key_def('std', cfg['augmentation']['normalization'])
    vis_params = {'colormap_file': colormap_file, 'heatmaps': heatmaps, 'heatmaps_inf': heatmaps_inf, 'grid': grid,
                  'mean': mean, 'std': std, 'vis_batch_range': vis_batch_range, 'vis_at_train': vis_at_train,
                  'vis_at_eval': vis_at_eval, 'ignore_index': dontcare_val, 'inference_input_path': None}

    # automatic model naming with unique id for each training
    config_path = None
    for list_path in cfg.general.config_path:
        if list_path['provider'] == 'main':
            config_path = list_path['path']
    default_output_path = Path(to_absolute_path(f'{tiling_dir}/model/{experiment_name}/{run_name}'))
    output_path = get_key_def('save_weights_dir', cfg['general'], default=default_output_path, to_path=True)
    if output_path.is_dir():
        last_mod_time_suffix = datetime.fromtimestamp(output_path.stat().st_mtime).strftime('%Y%m%d-%H%M%S')
        archive_output_path = output_path.parent / f"{output_path.stem}_{last_mod_time_suffix}"
        shutil.move(output_path, archive_output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    logging.info(f'\nModel will be saved to: {output_path}')
    if debug:
        logging.warning(f'\nDebug mode activated. Some debug features may mobilize extra disk space and '
                        f'cause delays in execution.')
    if dontcare_val < 0 and vis_batch_range:
        logging.warning(f'\nVisualization: expected positive value for ignore_index, got {dontcare_val}.'
                        f'\nWill be overridden to 255 during visualization only. Problems may occur.')

    # overwrite dontcare values in label if loss doens't implement ignore_index
    dontcare2backgr = False if 'ignore_index' in cfg.loss.keys() else True

    # Will check if batch size needs to be a lower value only if cropping patches during training
    calc_eval_bs = True if crop_size else False

    # Set device(s)
    gpu_devices_dict = get_device_ids(num_devices)
    device = set_device(gpu_devices_dict=gpu_devices_dict)

    # INSTANTIATE MODEL AND LOAD CHECKPOINT FROM PATH
    checkpoint = read_checkpoint(train_state_dict_path)
    model = define_model(
        net_params=cfg.model,
        in_channels=num_bands,
        out_classes=num_classes,
        main_device=device,
        devices=list(gpu_devices_dict.keys()),
        checkpoint_dict=checkpoint,
        checkpoint_dict_strict_load=state_dict_strict
    )

    criterion = define_loss(loss_params=cfg.loss, class_weights=class_weights)
    criterion = criterion.to(device)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

    logging.info(f'\nInstantiated {cfg.model._target_} model with {num_bands} input channels and {num_classes} output '
                 f'classes.')

    logging.info(f'Creating dataloaders from data in {tiling_dir}...\n')
    trn_dataloader, val_dataloader, tst_dataloader = create_dataloader(patches_folder=tiling_dir,
                                                                       batch_size=batch_size,
                                                                       eval_batch_size=eval_batch_size,
                                                                       gpu_devices_dict=gpu_devices_dict,
                                                                       sample_size=patches_size,
                                                                       dontcare_val=dontcare_val,
                                                                       crop_size=crop_size,
                                                                       num_bands=num_bands,
                                                                       min_annot_perc=min_annot_perc,
                                                                       attr_vals=attr_vals,
                                                                       scale=scale,
                                                                       cfg=cfg,
                                                                       dontcare2backgr=dontcare2backgr,
                                                                       compute_sampler_weights=compute_sampler_weights,
                                                                       debug=debug)

    # Save tracking
    set_tracker(mode='train', type='mlflow', task='segmentation', experiment_name=experiment_name, run_name=run_name,
                tracker_uri=tracker_uri, params=cfg,
                keys2log=['general', 'training', 'dataset', 'model', 'optimizer', 'scheduler', 'augmentation'])
    trn_log, val_log, tst_log = [InformationLogger(dataset) for dataset in ['trn', 'val', 'tst']]

    since = time.time()
    best_loss = 999
    last_vis_epoch = 0

    progress_log = output_path / 'progress.log'
    if not progress_log.exists():
        progress_log.open('w', buffering=1).write(tsv_line('ep_idx', 'phase', 'iter', 'i_p_ep', 'time'))  # Add header

    # VISUALIZATION: generate pngs of inputs, labels and outputs
    if vis_batch_range is not None:
        # Make sure user-provided range is a tuple with 3 integers (start, finish, increment).
        # Check once for all visualization tasks.
        if not len(vis_batch_range) == 3 and all(isinstance(x, int) for x in vis_batch_range):
            raise logging.critical(
                ValueError(f'\nVis_batch_range expects three integers in a list: start batch, end batch, increment.'
                           f'Got {vis_batch_range}')
            )
        vis_at_init_dataset = get_key_def('vis_at_init_dataset', cfg['visualization'], 'val')

        # Visualization at initialization. Visualize batch range before first eopch.
        if get_key_def('vis_at_init', cfg['visualization'], False):
            logging.info(f'\nVisualizing initialized model on batch range {vis_batch_range} '
                         f'from {vis_at_init_dataset} dataset...\n')
            vis_from_dataloader(vis_params=vis_params,
                                eval_loader=val_dataloader if vis_at_init_dataset == 'val' else tst_dataloader,
                                model=model,
                                ep_num=0,
                                output_path=output_path,
                                dataset=vis_at_init_dataset,
                                scale=scale,
                                device=device,
                                vis_batch_range=vis_batch_range)

    for epoch in range(0, num_epochs):
        logging.info(f'\nEpoch {epoch}/{num_epochs - 1}\n' + "-" * len(f'Epoch {epoch}/{num_epochs - 1}'))
        # creating trn_report
        trn_report = training(train_loader=trn_dataloader,
                              model=model,
                              criterion=criterion,
                              optimizer=optimizer,
                              scheduler=lr_scheduler,
                              num_classes=num_classes,
                              batch_size=batch_size,
                              ep_idx=epoch,
                              progress_log=progress_log,
                              device=device,
                              scale=scale,
                              vis_params=vis_params,
                              debug=debug)
        if 'trn_log' in locals():  # only save the value if a tracker is setup
            trn_log.add_values(trn_report, epoch, ignore=['precision', 'recall', 'fscore', 'iou'])
        val_report = evaluation(eval_loader=val_dataloader,
                                model=model,
                                criterion=criterion,
                                num_classes=num_classes,
                                batch_size=batch_size,
                                ep_idx=epoch,
                                progress_log=progress_log,
                                batch_metrics=batch_metrics,
                                dataset='val',
                                device=device,
                                scale=scale,
                                vis_params=vis_params,
                                debug=debug)
        val_loss = val_report['loss'].average()
        if 'val_log' in locals():  # only save the value if a tracker is setup
            if batch_metrics is not None:
                val_log.add_values(val_report, epoch)
            else:
                val_log.add_values(val_report, epoch, ignore=['precision', 'recall', 'fscore', 'iou'])

        if val_loss < best_loss:
            logging.info("\nSave checkpoint with a validation loss of {:.4f}".format(val_loss))  # only allow 4 decimals
            # create the checkpoint file
            checkpoint_tag = checkpoint_stack.pop()
            filename = output_path.joinpath(checkpoint_tag)
            if filename.is_file():
                filename.unlink()
            val_loss_string = f'{val_loss:.2f}'.replace('.', '-')
            modalities_str = [str(band) for band in modalities]
            checkpoint_tag = f'{experiment_name}_{num_classes}_{"_".join(modalities_str)}_{val_loss_string}.pth.tar'
            filename = output_path.joinpath(checkpoint_tag)
            checkpoint_stack.append(checkpoint_tag)
            best_loss = val_loss
            # More info:
            # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-torch-nn-dataparallel-models
            state_dict = model.module.state_dict() if num_devices > 1 else model.state_dict()
            torch.save({'epoch': epoch,
                        'params': cfg,
                        'model_state_dict': state_dict,
                        'best_loss': best_loss,
                        'optimizer_state_dict': optimizer.state_dict()}, filename)

            # VISUALIZATION: generate pngs of img patches, labels and outputs as alternative to follow training
            if vis_batch_range is not None and vis_at_checkpoint and epoch - last_vis_epoch >= ep_vis_min_thresh:
                if last_vis_epoch == 0:
                    logging.info(f'\nVisualizing with {vis_at_ckpt_dataset} dataset patches on checkpointed model for'
                                 f'batches in range {vis_batch_range}')
                vis_from_dataloader(vis_params=vis_params,
                                    eval_loader=val_dataloader if vis_at_ckpt_dataset == 'val' else tst_dataloader,
                                    model=model,
                                    ep_num=epoch+1,
                                    output_path=output_path,
                                    dataset=vis_at_ckpt_dataset,
                                    scale=scale,
                                    device=device,
                                    vis_batch_range=vis_batch_range)
                last_vis_epoch = epoch

        cur_elapsed = time.time() - since
        # logging.info(f'\nCurrent elapsed time {cur_elapsed // 60:.0f}m {cur_elapsed % 60:.0f}s')

    # load checkpoint model and evaluate it on test dataset.
    if int(cfg['general']['max_epochs']) > 0:   # if num_epochs is set to 0, model is loaded to evaluate on test set
        checkpoint = read_checkpoint(filename)
        checkpoint = adapt_checkpoint_to_dp_model(checkpoint, model)
        model.load_state_dict(state_dict=checkpoint['model_state_dict'])

    if tst_dataloader:
        tst_report = evaluation(eval_loader=tst_dataloader,
                                model=model,
                                criterion=criterion,
                                num_classes=num_classes,
                                batch_size=batch_size,
                                ep_idx=num_epochs,
                                progress_log=progress_log,
                                batch_metrics=batch_metrics,
                                dataset='tst',
                                scale=scale,
                                vis_params=vis_params,
                                device=device)
        if 'tst_log' in locals():  # only save the value if a tracker is set                                     up
            tst_log.add_values(tst_report, num_epochs)


def main(cfg: DictConfig) -> None:
    """
    Function to manage details about the training on segmentation task.

    -------
    1. Pre-processing TODO
    2. Training process

    -------
    :param cfg: (dict) Parameters found in the yaml config file.
    """
    # Preprocessing
    # HERE the code to do for the preprocessing for the segmentation

    # execute the name mode (need to be in this file for now)
    train(cfg)

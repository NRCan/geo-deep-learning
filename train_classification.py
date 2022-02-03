import time
import h5py
import torch
import warnings
import torchvision
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from omegaconf import DictConfig
from omegaconf.errors import ConfigKeyError

try:
    from pynvml import *
except ModuleNotFoundError:
    warnings.warn(f"The python Nvidia management library could not be imported. Ignore if running on CPU only.")

from torch.utils.data import DataLoader
from utils.logger import InformationLogger, save_logs_to_bucket, tsv_line, dict_path
from utils.metrics import report_classification, create_metrics_dict, iou
from models.model_choice import net, load_checkpoint, verify_weights
from utils.utils import load_from_checkpoint, get_device_ids, gpu_stats, get_key_def, read_modalities
# Set the logging file
from utils import utils

logging = utils.get_logger(__name__)  # import logging


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


def create_classif_dataloader(data_path, batch_size, num_devices):
    """
    Function to create dataloader objects for training, validation and test datasets.
    :param data_path: (str) path to the samples folder
    :param batch_size: (int) batch size
    :param num_devices: (int) number of GPUs used
    :return: trn_dataloader, val_dataloader, tst_dataloader
    """
    num_samples = {}
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
    tst_dataset = torchvision.datasets.ImageFolder(os.path.join(data_path, "tst"),
                                                   transform=transforms.Compose(
                                                       [transforms.Resize(299), transforms.ToTensor()]),
                                                   loader=loader)
    # FIXME assert that f is a file
    num_samples['tst'] = len([f for f in Path(data_path).joinpath('tst').glob('**/*')])

    # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
    num_workers = num_devices * 4 if num_devices > 1 else 4

    # Shuffle must be set to True.
    trn_dataloader = DataLoader(trn_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    tst_dataloader = DataLoader(tst_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False) if num_samples['tst'] > 0 else None

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
        if params['training'][f"num_{i}_samples"]:
            num_samples[i] = params['training'][f"num_{i}_samples"]

            with h5py.File(os.path.join(samples_path, f"{i}_samples.hdf5"), 'r') as hdf5_file:
                file_num_samples = len(hdf5_file['map_img'])
            if num_samples[i] > file_num_samples:
                raise IndexError(f"The number of training samples in the configuration file ({num_samples[i]}) "
                                 f"exceeds the number of samples in the hdf5 training dataset ({file_num_samples}).")
        else:
            with h5py.File(os.path.join(samples_path, f"{i}_samples.hdf5"), "r") as hdf5_file:
                num_samples[i] = len(hdf5_file['map_img'])

    return num_samples


def training(train_loader, model, criterion, optimizer, scheduler, num_classes, batch_size, ep_idx, progress_log, device, debug=False):
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
    :return: Updated training loss
    """
    model.train()
    train_metrics = create_metrics_dict(num_classes)

    with tqdm(train_loader, desc=f'Iterating train batches with {device.type}') as _tqdm:
        for batch_index, data in enumerate(_tqdm):
            progress_log.open('a', buffering=1).write(tsv_line(ep_idx, 'trn', batch_index, len(train_loader), time.time()))

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            train_metrics['loss'].update(loss.item(), batch_size)

            loss.backward()
            optimizer.step()

    scheduler.step()
    logging.info(f'Training Loss: {train_metrics["loss"].avg:.4f}')
    return train_metrics


def evaluation(eval_loader,
               model,
               criterion,
               num_classes,
               batch_size,
               ep_idx,
               progress_log,
               batch_metrics=None,
               dataset='val',
               ignore_index=None,
               device=None,
               debug=False):
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

    with tqdm(eval_loader, dynamic_ncols=True, desc=f'Iterating {dataset} batches with {device.type}') as _tqdm:
        for batch_index, data in enumerate(_tqdm):
            progress_log.open('a', buffering=1).write(tsv_line(ep_idx, dataset, batch_index, len(eval_loader), time.time()))

            with torch.no_grad():
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels_flatten = labels

                outputs = model(inputs)
                outputs_flatten = outputs

                loss = criterion(outputs, labels)

                eval_metrics['loss'].update(loss.item(), batch_size)

                if (dataset == 'val') and (batch_metrics is not None):
                    # Compute metrics every n batches. Time consuming.
                    assert batch_metrics <= len(_tqdm), f"Batch_metrics ({batch_metrics} is smaller than batch size " \
                        f"{len(_tqdm)}. Metrics in validation loop won't be computed"
                    if (batch_index+1) % batch_metrics == 0:   # +1 to skip val loop at very beginning
                        a, segmentation = torch.max(outputs_flatten, dim=1)
                        eval_metrics = report_classification(segmentation, labels_flatten, batch_size, eval_metrics,
                                                             ignore_index=ignore_index)
                elif dataset == 'tst':
                    a, segmentation = torch.max(outputs_flatten, dim=1)
                    eval_metrics = report_classification(segmentation, labels_flatten, batch_size, eval_metrics,
                                                         ignore_index=ignore_index)

                _tqdm.set_postfix(OrderedDict(dataset=dataset, loss=f'{eval_metrics["loss"].avg:.4f}'))

                if debug and device.type == 'cuda':
                    res, mem = gpu_stats(device=device.index)
                    _tqdm.set_postfix(OrderedDict(device=device, gpu_perc=f'{res.gpu} %',
                                                  gpu_RAM=f'{mem.used/(1024**2):.0f}/{mem.total/(1024**2):.0f} MiB'))

    logging.info(f"{dataset} Loss: {eval_metrics['loss'].avg}")
    if batch_metrics is not None:
        logging.info(f"{dataset} precision: {eval_metrics['precision'].avg}")
        logging.info(f"{dataset} recall: {eval_metrics['recall'].avg}")
        logging.info(f"{dataset} fscore: {eval_metrics['fscore'].avg}")

    return eval_metrics


def main(cfg: DictConfig) -> None:
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
    since = time.time()
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    best_loss = 999

    # MANDATORY PARAMETERS
    num_classes = len(get_key_def('classes_dict', cfg['dataset']).keys())
    num_bands = len(read_modalities(cfg.dataset.modalities))
    batch_size = get_key_def('batch_size', cfg['training'], expected_type=int)
    eval_batch_size = get_key_def('eval_batch_size', cfg['training'], expected_type=int, default=batch_size)
    num_epochs = get_key_def('max_epochs', cfg['training'], expected_type=int)
    model_name = get_key_def('model_name', cfg['model'], expected_type=str).lower()

    # OPTIONAL PARAMETERS
    debug = get_key_def('debug', cfg)
    task = get_key_def('task', cfg['general'], default='segmentation')
    dontcare_val = get_key_def("ignore_index", cfg['dataset'], default=-1)
    bucket_name = get_key_def('bucket_name', cfg['AWS'])
    batch_metrics = get_key_def('batch_metrics', cfg['training'], default=None)

    # MODEL PARAMETERS
    class_weights = get_key_def('class_weights', cfg['dataset'], default=None)
    loss_fn = get_key_def('loss_fn', cfg['training'], default='CrossEntropy')
    optimizer = get_key_def('optimizer_name', cfg['optimizer'], default='adam', expected_type=str)  # TODO change something to call the function
    pretrained = get_key_def('pretrained', cfg['model'], default=True, expected_type=bool)
    train_state_dict_path = get_key_def('state_dict_path', cfg['general'], default=None, expected_type=str)
    dropout_prob = get_key_def('factor', cfg['scheduler']['params'], default=None, expected_type=float)
    # if error
    if train_state_dict_path and not Path(train_state_dict_path).is_file():
        raise logging.critical(
            FileNotFoundError(f'\nCould not locate pretrained checkpoint for training: {train_state_dict_path}')
        )
    if class_weights:
        verify_weights(num_classes, class_weights)

    # GPU PARAMETERS
    num_devices = get_key_def('num_gpus', cfg['training'], default=0)
    if num_devices and not num_devices >= 0:
        raise logging.critical(ValueError("\nmissing mandatory num gpus parameter"))

    # LOGGING PARAMETERS TODO reimplement mlflow

    try:
        data_path = Path(str(cfg.dataset.raw_data_dir)).resolve(strict=True)
        logging.info("\nThe data directory used '{}'".format(data_path))
    except FileNotFoundError:
        logging.info(
            f"\nThe data directory '{cfg.dataset.raw_data_dir}' doesn't exist, please change the path.")

    # automatic model naming with unique id for each training
    config_path = None
    for list_path in cfg.general.config_path:
        if list_path['provider'] == 'main':
            config_path = list_path['path']
    config_name = str(cfg.general.config_name)
    model_id = config_name
    output_path = Path(f'model/{model_id}')
    output_path.mkdir(parents=True, exist_ok=False)
    logging.info(f'\nModel and log files will be saved to: {os.getcwd()}/{output_path}')
    if debug:
        logging.warning(f'\nDebug mode activated. Some debug features may mobilize extra disk space and '
                        f'cause delays in execution.')

    logging.info(f'Instantiated {model_name} model with {num_classes} output channels.\n')

    logging.info(f'Creating dataloaders from data in {Path(data_path)}...\n')
    trn_dataloader, val_dataloader, tst_dataloader = create_classif_dataloader(data_path=data_path,
                                                                               batch_size=batch_size,
                                                                               num_devices=num_devices,)

    if bucket_name:
        from utils.aws import download_s3_files
        bucket, bucket_output_path, output_path, data_path = download_s3_files(bucket_name=bucket_name,
                                                                               data_path=data_path,  # FIXME
                                                                               output_path=output_path)

    progress_log = output_path / 'progress.log'
    if not progress_log.exists():
        progress_log.open('w', buffering=1).write(tsv_line('ep_idx', 'phase', 'iter', 'i_p_ep', 'time'))  # Add header
    trn_log = InformationLogger('trn')
    val_log = InformationLogger('val')
    tst_log = InformationLogger('tst')

    # INSTANTIATE MODEL AND LOAD CHECKPOINT FROM PATH
    model, model_name, criterion, optimizer, lr_scheduler, device, gpu_devices_dict = \
        net(model_name=model_name,
            num_bands=num_bands,
            num_channels=num_classes,
            dontcare_val=dontcare_val,
            num_devices=num_devices,
            train_state_dict_path=train_state_dict_path,
            pretrained=pretrained,
            dropout_prob=dropout_prob,
            loss_fn=loss_fn,
            optimizer=optimizer,
            net_params=cfg)
    logging.info(f'Instantiated {model_name} model with {num_classes} output channels.\n')

    filename = os.path.join(output_path, 'checkpoint.pth.tar')

    for epoch in range(0, num_epochs):
        logging.info(f'\nEpoch {epoch}/{num_epochs - 1}\n{"-" * 20}')

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
                              debug=debug)
        trn_log.add_values(trn_report, epoch, ignore=['precision', 'recall', 'fscore', 'iou'])

        val_report = evaluation(eval_loader=val_dataloader,
                                model=model,
                                criterion=criterion,
                                num_classes=num_classes,
                                batch_size=eval_batch_size,
                                ep_idx=epoch,
                                progress_log=progress_log,
                                batch_metrics=cfg['training']['batch_metrics'],
                                dataset='val',
                                device=device,
                                debug=debug)
        val_loss = val_report['loss'].avg
        if cfg['training']['batch_metrics'] is not None:
            val_log.add_values(val_report, epoch, ignore=['iou'])
        else:
            val_log.add_values(val_report, epoch, ignore=['precision', 'recall', 'fscore', 'iou'])

        if val_loss < best_loss:
            logging.info("save checkpoint\n")
            best_loss = val_loss
            # More info: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-torch-nn-dataparallel-models
            state_dict = model.module.state_dict() if num_devices > 1 else model.state_dict()
            torch.save({'epoch': epoch,
                        'params': cfg,
                        'model': state_dict,
                        'best_loss': best_loss,
                        'optimizer': optimizer.state_dict()}, filename)

            if bucket_name:
                bucket_filename = os.path.join(bucket_output_path, 'checkpoint.pth.tar')
                bucket.upload_file(filename, bucket_filename)

        if bucket_name:
            save_logs_to_bucket(bucket, bucket_output_path, output_path, batch_metrics)

        cur_elapsed = time.time() - since
        logging.info(f'Current elapsed time {cur_elapsed // 60:.0f}m {cur_elapsed % 60:.0f}s')

    # load checkpoint model and evaluate it on test dataset.
    if num_epochs > 0:  # if num_epochs is set to 0, model is loaded to evaluate on test set
        checkpoint = load_checkpoint(filename)
        model, _ = load_from_checkpoint(checkpoint, model)

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
                                device=device)
        tst_log.add_values(tst_report, num_epochs, ignore=['iou'])

        if bucket_name:
            bucket_filename = os.path.join(bucket_output_path, 'last_epoch.pth.tar')
            bucket.upload_file("output.txt", os.path.join(bucket_output_path, f"Logs/{now}_output.txt"))
            bucket.upload_file(filename, bucket_filename)

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))



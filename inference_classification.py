from typing import List, Sequence

import torch
# import torch should be first. Unclear issue, mentionned here: https://github.com/pytorch/pytorch/issues/2083
import numpy as np
import os
import csv
import heapq
from PIL import Image
import torchvision
from omegaconf.errors import ConfigKeyError
from tqdm import tqdm
from pathlib import Path

from utils.logger import dict_path
from models.model_choice import net
from utils.utils import load_from_checkpoint, get_device_ids, get_key_def, list_input_images, read_modalities, \
    find_first_file
from utils.verifications import add_background_to_num_class, validate_num_classes, assert_crs_match

try:
    import boto3
except ModuleNotFoundError:
    pass
# Set the logging file
from utils import utils
logging = utils.get_logger(__name__)


def classifier(img_list, weights_file_name, classes, model, device, working_folder):
    """
    Classify images by class
    :param img_list:
    :param model:
    :param device:
    :return:
    """
    classes_file = weights_file_name.split('/')[:-1]
    class_csv = ''
    for c in classes_file:
        class_csv = class_csv + c + '/'

    classified_results = np.empty((0, 2 + len(classes)))

    for image in img_list:
        img_name = os.path.basename(image['tif'])  # TODO: pathlib
        model.eval()
        img = Image.open(image['tif']).resize((299, 299), resample=Image.BILINEAR)
        to_tensor = torchvision.transforms.ToTensor()

        img = to_tensor(img)
        img = img.unsqueeze(0)
        with torch.no_grad():
            img = img.to(device)
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)

        top5 = heapq.nlargest(5, outputs.cpu().numpy()[0])
        top5_loc = []
        for i in top5:
            top5_loc.append(np.where(outputs.cpu().numpy()[0] == i)[0][0])
        logging.info(f"Image {img_name} classified as {classes[0][predicted]}")
        logging.info('Top 5 classes:')
        for i in range(0, 5):
            logging.info(f"\t{classes[0][top5_loc[i]]} : {top5[i]}")
        classified_results = np.append(classified_results, [np.append([image['tif'], classes[0][predicted]],
                                                                      outputs.cpu().numpy()[0])], axis=0)
    csv_results = 'classification_results.csv'

    np.savetxt(os.path.join(working_folder, csv_results), classified_results, fmt='%s',  # TODO: pathlib
               delimiter=',')


def main(params: dict) -> None:
    """
    Function to manage details about the inference on segmentation task.
    1. Read the parameters from the config given.
    2. Read and load the state dict from the previous training or the given one.
    3. Make the inference on the data specifies in the config.
    -------
    :param params: (dict) Parameters found in the yaml config file.
    """
    # since = time.time()

    # PARAMETERS
    mode = get_key_def('mode', params, expected_type=str)
    task = get_key_def('task', params['general'], expected_type=str)
    model_name = get_key_def('model_name', params['model'], expected_type=str).lower()
    classes = list(get_key_def('classes_dict', params['dataset']).keys())
    modalities = read_modalities(get_key_def('modalities', params['dataset'], expected_type=str))
    num_bands = len(modalities)
    debug = get_key_def('debug', params, default=False, expected_type=bool)
    # SETTING OUTPUT DIRECTORY
    try:
        state_dict = Path(params['inference']['state_dict_path']).resolve(strict=True)
    except FileNotFoundError:
        logging.info(
            f"\nThe state dict path directory '{params['inference']['state_dict_path']}' don't seem to be find," +
            f"we will try to locate a state dict path in the '{params['general']['save_weights_dir']}' " +
            f"specify during the training phase"
        )
        try:
            state_dict = Path(params['general']['save_weights_dir']).resolve(strict=True)
        except FileNotFoundError:
            raise logging.critical(
                f"\nThe state dict path directory '{params['general']['save_weights_dir']}'" +
                f" don't seem to be find either, please specify the path to a state dict"
            )
    # TODO add more detail in the parent folder
    working_folder = state_dict.parent.joinpath(f'inference_{num_bands}bands')
    logging.info("\nThe state dict path directory used '{}'".format(working_folder))
    Path.mkdir(working_folder, parents=True, exist_ok=True)

    # LOGGING PARAMETERS TODO put option not just mlflow
    experiment_name = get_key_def('project_name', params['general'], default='gdl-training')
    try:
        tracker_uri = get_key_def('uri', params['tracker'], default=None, expected_type=str)
        Path(tracker_uri).mkdir(exist_ok=True)
        run_name = get_key_def('run_name', params['tracker'], default='gdl')  # TODO change for something meaningful
        run_name = '{}_{}_{}'.format(run_name, mode, task)
        logging.info(f'\nInference and log files will be saved to: {working_folder}')
        # TODO change to fit whatever inport
        from mlflow import log_params, set_tracking_uri, set_experiment, start_run, log_artifact, log_metrics
        # tracking path + parameters logging
        set_tracking_uri(tracker_uri)
        set_experiment(experiment_name)
        start_run(run_name=run_name)
        log_params(dict_path(params, 'general'))
        log_params(dict_path(params, 'dataset'))
        log_params(dict_path(params, 'data'))
        log_params(dict_path(params, 'model'))
        log_params(dict_path(params, 'inference'))
    # meaning no logging tracker as been assigned or it doesnt exist in config/logging
    except ConfigKeyError:
        logging.info(
            "\nNo logging tracker as been assigned or the yaml config doesnt exist in 'config/tracker'."
            "\nNo tracker file will be save in that case."
        )

    # MANDATORY PARAMETERS
    img_dir_or_csv = get_key_def(
        'img_dir_or_csv_file', params['inference'], default=params['general']['raw_data_csv'], expected_type=str
    )
    if not (Path(img_dir_or_csv).is_dir() or Path(img_dir_or_csv).suffix == '.csv'):
        raise logging.critical(
            FileNotFoundError(
                f'\nCouldn\'t locate .csv file or directory "{img_dir_or_csv}" containing imagery for inference'
            )
        )
    # load the checkpoint
    try:
        # Sort by modification time (mtime) descending
        sorted_by_mtime_descending = sorted(
            [os.path.join(state_dict, x) for x in os.listdir(state_dict)], key=lambda t: -os.stat(t).st_mtime
        )
        last_checkpoint_save = find_first_file('checkpoint.pth.tar', sorted_by_mtime_descending)
        if last_checkpoint_save is None:
            raise FileNotFoundError
        # change the state_dict
        state_dict = last_checkpoint_save
    except FileNotFoundError as e:
        logging.error(f"\nNo file name 'checkpoint.pth.tar' as been found at '{state_dict}'")
        raise e

    # OPTIONAL PARAMETERS
    dontcare_val = get_key_def("ignore_index", params["training"], default=-1, expected_type=int)
    num_devices = get_key_def('num_gpus', params['training'], default=0, expected_type=int)
    default_max_used_ram = 25
    max_used_ram = get_key_def('max_used_ram', params['training'], default=default_max_used_ram, expected_type=int)
    max_used_perc = get_key_def('max_used_perc', params['training'], default=25, expected_type=int)

    # benchmark (ie when gkpgs are inputted along with imagery)
    dontcare = get_key_def("ignore_index", params["training"], -1)
    attribute_field = get_key_def('attribute_field', params['dataset'], None, expected_type=str)
    attr_vals = get_key_def('attribute_values', params['dataset'], None, expected_type=Sequence)

    if debug:
        logging.warning(f'\nDebug mode activated. Some debug features may mobilize extra disk space and '
                        f'cause delays in execution.')

    # Assert that all values are integers (ex.: to benchmark single-class model with multi-class labels)
    if attr_vals:
        for item in attr_vals:
            if not isinstance(item, int):
                raise ValueError(f'\nValue "{item}" in attribute_values is {type(item)}, expected int.')

    logging.info(f'\nInferences will be saved to: {working_folder}\n\n')
    if not (0 <= max_used_ram <= 100):
        logging.warning(f'\nMax used ram parameter should be a percentage. Got {max_used_ram}. '
                        f'Will set default value of {default_max_used_ram} %')
        max_used_ram = default_max_used_ram

    # AWS
    bucket_name = get_key_def('bucket_name', params['AWS'])

    # list of GPU devices that are available and unused. If no GPUs, returns empty dict
    gpu_devices_dict = get_device_ids(num_devices,
                                      max_used_ram_perc=max_used_ram,
                                      max_used_perc=max_used_perc)
    if gpu_devices_dict:
        logging.info(f"\nNumber of cuda devices requested: {num_devices}. "
                     f"\nCuda devices available: {gpu_devices_dict}. "
                     f"\nUsing {list(gpu_devices_dict.keys())[0]}\n\n")
        device = torch.device(f'cuda:{list(range(len(gpu_devices_dict.keys())))[0]}')
    else:
        logging.warning(f"\nNo Cuda device available. This process will only run on CPU")
        device = torch.device('cpu')

    # CONFIGURE MODEL
    num_classes_backgr = add_background_to_num_class(task, len(classes))
    model, loaded_checkpoint, model_name = net(model_name=model_name,
                                               num_bands=num_bands,
                                               num_channels=num_classes_backgr,
                                               dontcare_val=dontcare_val,
                                               num_devices=1,
                                               net_params=params,
                                               inference_state_dict=state_dict)
    try:
        model.to(device)
    except RuntimeError:
        logging.info(f"\nUnable to use device. Trying device 0")
        device = torch.device(f'cuda' if gpu_devices_dict else 'cpu')
        model.to(device)

    # CREATE LIST OF INPUT IMAGES FOR INFERENCE
    try:
        # check if the data folder exist
        raw_data_dir = get_key_def('raw_data_dir', params['dataset'])
        my_data_path = Path(raw_data_dir).resolve(strict=True)
        logging.info("\nImage directory used '{}'".format(my_data_path))
        data_path = Path(my_data_path)
    except FileNotFoundError:
        raw_data_dir = get_key_def('raw_data_dir', params['dataset'])
        raise logging.critical(
            "\nImage directory '{}' doesn't exist, please change the path".format(raw_data_dir)
        )
    list_img = list_input_images(
        img_dir_or_csv, bucket_name, glob_patterns=["*.tif", "*.TIF"], in_case_of_path=str(data_path)
    )

    # VALIDATION: anticipate problems with imagery and label (if provided) before entering main for loop
    valid_gpkg_set = set()
    for info in tqdm(list_img, desc='Validating imagery'):
        # validate_raster(info['tif'], num_bands, meta_map)
        if 'gpkg' in info.keys() and info['gpkg'] and info['gpkg'] not in valid_gpkg_set:
            validate_num_classes(vector_file=info['gpkg'],
                                 num_classes=len(classes),
                                 attribute_name=attribute_field,
                                 ignore_index=dontcare,
                                 attribute_values=attr_vals)
            assert_crs_match(info['tif'], info['gpkg'])
            valid_gpkg_set.add(info['gpkg'])

    logging.info('\nSuccessfully validated imagery')
    if valid_gpkg_set:
        logging.info('\nSuccessfully validated label data for benchmarking')

    classifier(list_img, state_dict, classes, model, device, working_folder)  # FIXME: why don't we load from checkpoint?
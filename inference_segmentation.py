# Licensed under the MIT License.
# Authors: Victor Alhassan, Rémi Tavon

# Adapted from: https://github.com/microsoft/torchgeo/blob/3f7e525fbd01dddd25804e7a1b7634269ead1760/evaluate.py
# Also see: https://gist.github.com/calebrob6/7b226eb73877187f85fb5e1621bb7971

# Hardware requirements: 64 Gb RAM (cpu), 8 Gb GPU RAM.

"""CCMEO model inference script."""
import argparse
import os
from numbers import Number
from pathlib import Path
from typing import Dict, Any, Sequence, Union

import numpy as np
import rasterio
from hydra.utils import instantiate
from omegaconf import DictConfig
from pandas.io.common import is_url
from pytorch_lightning import LightningModule, seed_everything
import torch
from rasterio.plot import reshape_as_raster
from torch import autocast, Tensor
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.transforms import Compose
from tqdm import tqdm

from dataset.aoi import AOI, aois_from_csv
from inference.InferenceDataModule import InferenceDataModule, preprocess, pad
from models.model_choice import define_model_architecture
from utils.geoutils import create_new_raster_from_base
from utils.logger import get_logger
from utils.utils import _window_2D, get_device_ids, get_key_def, set_device, override_model_params_from_checkpoint, \
    checkpoint_converter, read_checkpoint, extension_remover, class_from_heatmap, stretch_heatmap, ckpt_is_compatible

# Set the logging file
logging = get_logger(__name__)


# TODO merge with GDL then remove this class
class InferenceTask(LightningModule):
    """
    Inspiration: torchgeo.trainers.SemanticSegmentationTask
    """
    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        self.model = define_model_architecture(
            self.hparams["model"],
            self.hparams["model"]["in_channels"],
            self.hparams["model"]["classes"]
        )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model.
        Keyword Args:
            segmentation_model: Name of the segmentation model type to use
            encoder_name: Name of the encoder model backbone to use
            encoder_weights: None or "imagenet" to use imagenet pretrained weights in
                the encoder model
            in_channels: Number of channels in input image
            num_classes: Number of semantic classes to predict
        Raises:
            ValueError: if kwargs arguments are invalid
        """
        super().__init__()
        self.save_hyperparameters()  # creates `self.hparams` from kwargs
        self.model = None

        self.config_task()

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Inference step - outputs prediction for non-labeled imagery.
        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y_hat = self.forward(x)

        return y_hat

    def forward(self, x: Tensor) -> Any:  # type: ignore[override]
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return self.model(x)


def auto_batch_size_finder(datamodule, device, model, single_class_mode=False, max_used_ram: int = 95):
    """
    Auto batch size finder scales batch size to find the largest batch size that fits into memory.
    Pytorch Lightning's alternative (not implemented for inference):
    https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html#auto-scaling-of-batch-size
    @param datamodule:
        data module that will be used for main prediction task
    @param device:
        the device to put data on
    @param model:
        model that will be used for main prediction task
    @param single_class_mode:
        set to True if training a single-class model
    @param max_used_ram:
        max % of RAM the batch size should use
    @return: int
        largest batch size before max_used_ram threshold is reached for GPU memory
    """
    src_size = datamodule.inference_dataset.aoi.raster_meta['height'] \
               * datamodule.inference_dataset.aoi.raster_meta['width']
    bs_min, bs_max, bs_step = 4, 256, 4
    window_spline_2d = create_spline_window(datamodule.patch_size).to(device)
    _tqdm = tqdm(range(bs_min, bs_max, bs_step), desc=f"Finding batch size filling GPU to {max_used_ram} % or less")
    for trial in _tqdm:
        batch_size_trial = trial
        _tqdm.set_postfix_str(f"Trying: {batch_size_trial}")
        datamodule.batch_size = batch_size_trial
        if batch_size_trial * datamodule.patch_size ** 2 > src_size:
            logging.info(f"Reached maximum batch size for image size. "
                         f"Batch size tuned to {batch_size_trial - bs_step}.")
            return batch_size_trial - bs_step
        eval_gen2tune = eval_batch_generator(
            model=model,
            dataloader=datamodule.predict_dataloader(),
            device=device,
            single_class_mode=single_class_mode,
            window_spline_2d=window_spline_2d,
            pad=datamodule.pad_size,
            verbose=False,
        )
        _ = next(eval_gen2tune)
        free, total = torch.cuda.mem_get_info(device)
        if (total - free) / total > max_used_ram / 100:
            logging.info(f"Reached GPU RAM threshold of {int(max_used_ram)}%. Batch size tuned to {batch_size_trial}.")
            return batch_size_trial


def create_spline_window(window_size: int, power=1):
    """
    Create a float-tensor window for smoothing during inference.
    Adapted from : https://github.com/Vooban/Smoothly-Blend-Image-Patches
    @param window_size:
        Size of one side of window to create. Currently only implemented for square predictions
    @param power: power to apply to values in window
    @return: returns a torch.Tensor array
    """
    window_spline_2d = _window_2D(window_size=window_size, power=power)
    window_spline_2d = torch.as_tensor(np.moveaxis(window_spline_2d, 2, 0), ).float()
    return window_spline_2d


def eval_batch_generator(
        model: LightningModule,
        dataloader: Any,
        device: torch.device,
        single_class_mode: bool,
        window_spline_2d,
        pad,
        verbose: bool = True,
) -> Any:
    """Runs an adapted version of test loop without label data over a dataloader and returns prediction.
    Args:
        model: the model used for inference
        dataloader: the dataloader to get samples from
        device: the device to put data on
        single_class_mode: set to True if training a single-class model
        window_spline_2d: spline window to use for smoothing overlapping predictions
        pad: padding value used in transforms
        verbose: if True, print progress bar. Should be set to False when auto scaling batch size.
    Returns:
        the prediction for a dataloader batch
    """
    batch_output = {}
    for batch in tqdm(dataloader, disable=not verbose):
        batch_output['bbox'] = batch['bbox']
        batch_output['crs'] = batch['crs']
        inputs = batch["image"].to(device)
        with torch.no_grad(), autocast(device_type=device.type):
            outputs = model(inputs)
        if single_class_mode:
            outputs = torch.sigmoid(outputs)
        else:
            outputs = F.softmax(outputs, dim=1)
        outputs = outputs[..., pad:-pad, pad:-pad]
        outputs = torch.mul(outputs, window_spline_2d)
        batch_output['data'] = outputs.permute(0, 2, 3, 1).cpu().numpy().astype('float16')
        yield batch_output


def save_heatmap(heatmap: np.ndarray, outpath: Union[str, Path], src: rasterio.DatasetReader):
    """
    Write a heatmap as array to disk
    @param heatmap:
        array of dtype float containing probability map for each class,
        after sigmoid or softmax operation (expects values between 0 and 1)
    @param outpath:
        path to desired output file
    @param src:
        a rasterio file handler (aka DatasetReader) from source imagery. Contains metadata to write heatmap
    @return:
    """
    heatmap_arr = stretch_heatmap(heatmap_arr=heatmap, out_max=100)
    heatmap_arr = reshape_as_raster(heatmap_arr).astype(np.uint8)
    create_new_raster_from_base(input_raster=src, output_raster=outpath, write_array=heatmap_arr)


def main(params):
    """High-level pipeline.
    Runs a model checkpoint on non-labeled imagery and saves results to file.
    Args:
        params: configuration parameters
    """
    logging.debug(f"\nSetting inference parameters")
    # Main params
    raw_data_csv = get_key_def('raw_data_csv', params['inference'], expected_type=str, default=None,
                               validate_path_exists=True)
    item_url = get_key_def('input_stac_item', params['inference'], expected_type=str, validate_path_exists=True)
    root = get_key_def('root_dir', params['inference'], default="inference", to_path=True)
    root.mkdir(exist_ok=True)
    data_dir = get_key_def('raw_data_dir', params['dataset'], default="data", to_path=True, validate_path_exists=True)
    models_dir = get_key_def('checkpoint_dir', params['inference'], default=root / 'checkpoints', to_path=True)
    models_dir.mkdir(exist_ok=True)
    outname = get_key_def('output_name', params['inference'], default=f"{Path(item_url).stem}_pred")
    outname = extension_remover(outname)
    outpath = root / f"{outname}.tif"
    checkpoint = get_key_def('state_dict_path', params['inference'], to_path=True,
                             validate_path_exists=True, wildcard='*pth.tar')
    download_data = get_key_def('download_data', params['inference'], default=False, expected_type=bool)
    save_heatmap_bool = get_key_def('save_heatmap', params['inference'], default=False, expected_type=bool)
    heatmap_threshold = get_key_def('heatmap_threshold', params['inference'], default=50, expected_type=int)
    heatmap_name = get_key_def('heatmap_name', params['inference'], default=f"{outpath.stem}_heatmap",
                               expected_type=str)
    outpath_heat = root / f"{heatmap_name}.tif"

    # Create yaml to use pytorch lightning model management
    if is_url(checkpoint):
        load_state_dict_from_url(url=checkpoint, map_location='cpu', model_dir=models_dir)
        checkpoint = models_dir / Path(checkpoint).name
    if not ckpt_is_compatible(checkpoint):
        checkpoint = checkpoint_converter(in_pth_path=checkpoint, out_dir=models_dir)
    checkpoint_dict = read_checkpoint(checkpoint, out_dir=models_dir, update=False)
    params = override_model_params_from_checkpoint(params=params, checkpoint_params=checkpoint_dict["hyper_parameters"])

    # TODO: remove if no old models are used in production.
    # Covers old single-class models with 2 input channels rather than 1 (ie with background)
    single_class_mode = get_key_def('state_dict_single_mode', params['inference'], default=True, expected_type=bool)
    # Dataset params
    bands_requested = get_key_def('bands', params['dataset'], default=("red", "blue", "green"), expected_type=Sequence)
    if item_url and [bands for bands in bands_requested if isinstance(bands, int)]:
        # TODO: rethink this patch
        bands = ['red', 'green', 'blue', 'nir']
        bands_requested = [bands[i-1] for i in bands_requested]
        logging.warning(f"Got stac item as input imagery for inference, but model's requested bands are integers."
                        f"\nWill request: {bands_requested}")
    classes_dict = get_key_def('classes_dict', params['dataset'], expected_type=DictConfig)
    classes_dict = {k: v for k, v in classes_dict.items() if v}  # Discard keys where value is None
    # +1 for background if multiclass mode
    num_classes = len(classes_dict) if len(classes_dict) == 1 and single_class_mode else len(classes_dict) + 1

    # Hardware
    num_devices = get_key_def('gpu', params['inference'], default=1, expected_type=(int, bool))
    device_id = get_key_def('gpu_id', params['inference'], default=None, expected_type=int)
    max_used_ram = get_key_def('max_used_ram', params['inference'], default=100, expected_type=int)
    if not (0 <= max_used_ram <= 100):
        raise ValueError(f'\nMax used ram parameter should be a percentage. Got {max_used_ram}.')
    max_used_perc = get_key_def('max_used_perc', params['inference'], default=25, expected_type=int)
    # list of GPU devices that are available and unused. If no GPUs, returns empty dict
    gpu_devices_dict = get_device_ids(num_devices, max_used_ram_perc=max_used_ram, max_used_perc=max_used_perc)
    if device_id:
        logging.warning(
            f"\nWill use GPU with id {device_id}. This parameter has priority over \"inference.gpu\" parameter")
        device = torch.device(f'cuda:{device_id}')
    else:
        device = set_device(gpu_devices_dict=gpu_devices_dict)
    num_workers_default = 0 if device == 'cpu' else 4
    num_workers = get_key_def('num_workers', params['inference'], default=num_workers_default, expected_type=int)

    # Sampling, batching and augmentations configuration
    batch_size = get_key_def('batch_size', params['inference'], default=None, expected_type=int)
    chip_size = get_key_def('chunk_size', params['inference'], default=512, expected_type=int)
    auto_batch_size = True if not batch_size else False
    if auto_batch_size and device.type == 'cpu':
        logging.warning(f"Auto batch size not implemented for cpu execution. Batch size will default to 1.")
        batch_size = 1
    auto_bs_threshold = get_key_def('auto_batch_size_threshold', params['inference'], default=95, expected_type=int)
    stride_default = int(chip_size / 2) if chip_size else 256
    stride = get_key_def('stride', params['inference'], default=stride_default, expected_type=int)
    if chip_size and stride > chip_size * 0.75:
        logging.warning(f"Setting a large stride (more than 75% of chip size) will interfere with "
                        f"spline window smoothing operations and may result in poor quality extraction.")
    pad_size = get_key_def('pad', params['inference'], default=16, expected_type=int)
    clahe_clip_limit = get_key_def('clahe_enhance_clip_limit', params['augmentation'], expected_type=Number, default=0)
    test_transforms_cfg = get_key_def('test_transforms', params['inference'],
                                      default=Compose([pad(pad_size, mode='reflect'), preprocess]),
                                      expected_type=(dict, DictConfig))

    test_transforms = instantiate(test_transforms_cfg)

    # TODO: tune TTA with optuna
    tta_transforms = get_key_def('tta_transforms', params['inference'], default=None, expected_type=(dict, DictConfig))

    seed = 123
    seed_everything(seed)

    # GET LIST OF INPUT IMAGES FOR INFERENCE
    if raw_data_csv and item_url:
        raise ValueError(f"Input imagery should be a csv of stac item. Got inputs from both \"raw_data_csv\" and "
                         f"\"input stac item\"")
    if raw_data_csv is not None:
        list_aois = aois_from_csv(csv_path=raw_data_csv, bands_requested=bands_requested, download_data=download_data,
                                  data_dir=data_dir, for_multiprocessing=True,
                                  equalize_clahe_clip_limit=clahe_clip_limit)
    elif item_url:
        list_aois = [
            AOI(raster=str(item_url), raster_bands_request=bands_requested, split='inference',
                aoi_id=Path(item_url).stem, download_data=download_data, root_dir=data_dir, for_multiprocessing=True,
                equalize_clahe_clip_limit=clahe_clip_limit)
        ]
    else:
        raise NotImplementedError(f"No valid input provided. Set input with \"raw_data_csv\" or \"input stac item\"")

    logging.debug(f"\nInstantiating model with pretrained weights from {checkpoint}")
    try:
        model = InferenceTask.load_from_checkpoint(checkpoint)
    except (KeyError, AssertionError) as e:
        logging.warning(f"\nModel checkpoint is not compatible with pytorch-ligthning's load_from_checkpoint method:\n"
                        f"Key error: {e}\n")
        raise e

    model.freeze()
    model.eval()
    model = model.to(device)

    logging.debug(f"\nInstantiating test-time augmentations")
    if tta_transforms:
        model = instantiate(tta_transforms, model=model)

    # LOOP THROUGH LIST OF INPUT IMAGES
    for aoi in tqdm(list_aois, desc='Inferring from images', position=0, leave=True):
        dm = InferenceDataModule(aoi=aoi,
                                 outpath=outpath,
                                 patch_size=chip_size,
                                 stride=stride,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 pad=pad_size,
                                 save_heatmap=save_heatmap_bool,
                                 )
        dm.setup(test_transforms=test_transforms)

        height, width = aoi.raster_meta['height'], aoi.raster_meta['width']

        if auto_batch_size and device.type != "cpu":
            batch_size = auto_batch_size_finder(datamodule=dm,
                                                device=device,
                                                model=model,
                                                single_class_mode=single_class_mode,
                                                max_used_ram=auto_bs_threshold,
                                                )
            # Set final batch size from auto found value
            dm.batch_size = batch_size

        window_spline_2d = create_spline_window(chip_size).to(device)
        logging.debug(f"\nInstantiating inference generator for looping over imagery chips")
        eval_gen = eval_batch_generator(
            model=model,
            dataloader=dm.predict_dataloader(),
            device=device,
            single_class_mode=single_class_mode,
            window_spline_2d=window_spline_2d,
            pad=dm.pad_size,
        )

        # Create a numpy memory map to write results from per-chip inference to full-size prediction
        tempfile = root / f"{Path(aoi.raster_name).stem}.dat"
        fp = np.memmap(tempfile, dtype='float16', mode='w+', shape=(height, width, num_classes))
        transform = aoi.raster_meta['transform']
        for inference_prediction in eval_gen:
            # iterate through the batch and paste the predictions where they belong
            for i in range(len(inference_prediction['bbox'])):
                bb = inference_prediction["bbox"][i]
                col_min, row_min = ~transform * (bb.minx, bb.maxy)  # top left
                col_min, row_min = round(col_min), round(row_min)
                right = col_min + chip_size if col_min + chip_size <= width else width
                bottom = row_min + chip_size if row_min + chip_size <= height else height

                pred = inference_prediction['data'][i]
                # Write prediction on top of existing prediction for smoothing purposes
                fp[row_min:bottom, col_min:right, :] = \
                    fp[row_min:bottom, col_min:right, :] + pred[:bottom - row_min, :right - col_min]
        fp.flush()
        del fp

        logging.debug(f"\nReading full-size prediction memory-map and writing to final raster after argmax operation "
                      f"(discard confidence info)")
        fp = np.memmap(tempfile, dtype='float16', mode='r', shape=(height, width, num_classes))
        pred_img = class_from_heatmap(heatmap_arr=fp, heatmap_threshold=heatmap_threshold)
        pred_img = pred_img[np.newaxis, :, :].astype(np.uint8)

        if not aoi.raster:  # in case of multiprocessing
            aoi.raster = rasterio.open(aoi.raster_multiband)
        create_new_raster_from_base(input_raster=aoi.raster, output_raster=outpath, write_array=pred_img)

        logging.info(f'\nInference completed on {aoi.raster_name}'
                     f'\nFinal prediction written to {outpath}')

        if dm.save_heatmap:
            logging.info(f"\nSaving heatmap...")
            save_heatmap(heatmap=fp, outpath=outpath_heat, src=aoi.raster)
            logging.info(f'\nSaved heatmap to {outpath_heat}')

        if outpath.is_file():
            logging.debug(f"\nDeleting temporary .dat file {tempfile}...")
            fp.flush()
            del fp
            os.remove(tempfile)

        return outpath, pred_img


if __name__ == "__main__":  # serves as back up
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-stac-item",
        required=True,
        help="url or path to stac item pointing to imagery stored one band / file and implementing stac's eo extension",
        metavar="STAC")
    parser.add_argument(
        "--state-dict-path",
        required=True,
        help="path to the checkpoint file to test",
        metavar="CKPT")
    parser.add_argument(
        "--root-dir",
        help="root directory where dataset downloaded if desired and outputs will be written",
        default="data",
        metavar="DIR")
    args = parser.parse_args()
    params = {"inference": {"input_stac_item": args.input_stac_item, "state_dict_path": args.state_dict_path,
                            "root_dir": args.root_dir},
              "dataset": {}}
    main(params)

# Licensed under the MIT License.
# Adapted from: https://github.com/microsoft/torchgeo/blob/3f7e525fbd01dddd25804e7a1b7634269ead1760/evaluate.py

# Hardware requirements: 64 Gb RAM (cpu), 8 Gb GPU RAM.

"""CCMEO model inference script."""
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Sequence, List, Union

import numpy as np
import rasterio
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule, seed_everything
import segmentation_models_pytorch as smp
import ttach as tta
import torch
from torch import autocast
import torch.nn.functional as F
from torchgeo.trainers import SemanticSegmentationTask
from tqdm import tqdm
from ttach import SegmentationTTAWrapper

import inference_postprocess
from inference.InferenceDataModule import InferenceDataModule
from models.model_choice import load_checkpoint
from utils.logger import get_logger
from utils.utils import _window_2D, get_device_ids, get_key_def, set_device

# Set the logging file
logging = get_logger(__name__)


# TODO merge with GDL then remove this class
class InferenceTask(SemanticSegmentationTask):
    def config_task(self) -> None:  # TODO: use Hydra
        """Configures the task based on kwargs parameters passed to the constructor."""
        if self.hparams["segmentation_model"] == "unet_pretrained":
            self.model = smp.Unet(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
            )
        elif self.hparams["segmentation_model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
            )
        elif self.hparams["segmentation_model"] == "manet_pretrained":
            self.model = smp.MAnet(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
            )
        else:
            raise ValueError(
                f"Model type '{self.hparams['segmentation_model']}' is not valid."
            )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function.
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
        super().__init__(ignore_zeros=True)  # softcode once torchgeo's issue #444 is resolved
        self.save_hyperparameters()  # creates `self.hparams` from kwargs

        self.config_task()

    def inference_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Inference step - outputs prediction for non-labeled imagery.
        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y_hat = self.forward(x)
        # y_hat_hard = y_hat.argmax(dim=1)

        return y_hat


def gdl2pl_checkpoint(in_pth_path: str, out_pth_path: str = None):
    """
    Converts a geo-deep-learning/pytorch checkpoint (from v2.0.0+) to a pytorch lightning checkpoint.
    The outputted model should remain compatible with geo-deep-learning's checkpoint loading.
    @param in_pth_path: path to input checkpoint
    @param out_pth_path: path where pytorch-lightning adapted checkpoint should be written. Default to same as input,
                         but with "pl_" prefix in front of name
    @return: path to outputted checkpoint and path to outputted yaml to use when loading with pytorch lightning
    """
    if not out_pth_path:
        out_pth_path = Path(in_pth_path).parent / f'pl_{Path(in_pth_path).name}'
    if Path(out_pth_path).is_file():
        return out_pth_path
    # load with geo-deep-learning method
    checkpoint = load_checkpoint(in_pth_path)

    hparams = OmegaConf.create()
    if 'general' in checkpoint['params'].keys():  # for models saved as of v2.0.0
        class_keys = len(get_key_def('classes_dict', checkpoint['params']['dataset']).keys())
        num_classes = class_keys if class_keys == 1 else class_keys + 1  # +1 for background(multiclass mode)
        # Store hyper parameters to checkpoint as expected by pytorch lightning
        hparams["segmentation_model"] = checkpoint['params']['model']['model_name']  # TODO temporary
        hparams["in_channels"] = len(checkpoint['params']['dataset']['modalities'])
    else:  # for models saved before v2.0.0
        num_classes = int(checkpoint['params']['global']['num_classes']) + 1
        # Store hyper parameters to checkpoint as expected by pytorch lightning
        hparams["segmentation_model"] = checkpoint['params']['global']['model_name']  # TODO temporary
        hparams["in_channels"] = checkpoint['params']['global']['number_of_bands']

    hparams["encoder_name"] = "resnext50_32x4d"  # TODO temporary
    hparams["encoder_weights"] = 'imagenet'
    hparams["num_classes"] = num_classes
    hparams["ignore_zeros"] = True  #False if num_classes == 1 else True

    # adapt to what pytorch lightning expects: add "model" prefix to model keys
    if not list(checkpoint['model'].keys())[0].startswith('model'):
        new_state_dict = {}
        new_state_dict['model'] = checkpoint['model'].copy()
        new_state_dict['model'] = {'model.'+k: v for k, v in checkpoint['model'].items()}    # Very flimsy
        checkpoint['model'] = new_state_dict['model']

    # keep all keys and copy model weights to new state_dict key
    pl_checkpoint = checkpoint.copy()
    pl_checkpoint['state_dict'] = pl_checkpoint['model']
    pl_checkpoint['hyper_parameters'] = hparams
    torch.save(pl_checkpoint, out_pth_path)

    return out_pth_path


def auto_chip_size_finder(datamodule, device, model, tta_transforms, single_class_mode=False, max_used_ram: int = 95):
    """
    TODO
    @param datamodule:
    @param device:
    @param model:
    @param tta_transforms: test-time transforms as implemented run_eval_loop()
    @param single_class_mode:
    @param max_used_ram:
    @return:
    """
    chip_size_init = chip_size_trial = 256
    _tqdm = tqdm(range(32), desc=f"Finding chip size filling GPU to {max_used_ram} % or less")
    for trial in _tqdm:
        chip_size_trial += chip_size_init*(trial+1)
        _tqdm.set_postfix_str(f"Trying: {chip_size_trial}")
        datamodule.patch_size = chip_size_trial
        datamodule.stride = int(chip_size_trial / 2)
        if chip_size_trial > min(datamodule.inference_dataset.src.shape):
            logging.info(f"Reached maximum chip size for image size. "
                         f"Chip size tuned to {chip_size_trial-chip_size_init}.")
            return chip_size_trial-chip_size_init
        window_spline_2d = create_spline_window(chip_size_trial).to(device)
        eval_gen2tune = run_eval_loop(
            model=model,
            dataloader=datamodule.predict_dataloader(),
            device=device,
            single_class_mode=single_class_mode,
            window_spline_2d=window_spline_2d,
            pad=datamodule.pad_size,
            tta_transforms=tta_transforms,
        )
        _ = next(eval_gen2tune)
        free, total = torch.cuda.mem_get_info(device)
        if (total-free)/total > max_used_ram/100:
            logging.info(f"Reached GPU RAM threshold of {int(max_used_ram)}%. Chip size tuned to {chip_size_trial}.")
            return chip_size_trial


def create_spline_window(chip_size, power=1):
    window_spline_2d = _window_2D(window_size=chip_size, power=power)
    window_spline_2d = torch.as_tensor(np.moveaxis(window_spline_2d, 2, 0), ).float()
    return window_spline_2d


def run_eval_loop(
    model: LightningModule,
    dataloader: Any,
    device: torch.device,
    single_class_mode: bool,
    window_spline_2d,
    pad,
    tta_transforms: Union[List, str] = "horizontal_flip",
    tta_merge_mode: str = 'max',
) -> Any:
    """Runs an adapted version of test loop without label data over a dataloader and returns prediction.
    Args:
        model: the model used for inference
        dataloader: the dataloader to get samples from
        device: the device to put data on
        single_class_mode: TODO
        window_spline_2d: TODO
        pad: TODO
        tta_transforms:
        tta_merge_mode: TODO
    Returns:
        the prediction for a dataloader batch
    """
    # initialize test time augmentation
    transforms_dict = {"horizontal_flip": tta.Compose([tta.HorizontalFlip()]), "d4": tta.aliases.d4_transform()}
    if tta_transforms not in transforms_dict.keys():
        raise ValueError(f"TTA transform {tta_transforms} is not implemented. Choose between {transforms_dict.keys()}")
    transforms = transforms_dict[tta_transforms]  # TODO: tune with optuna
    model = SegmentationTTAWrapper(model, transforms, merge_mode=tta_merge_mode)

    batch_output = {}
    for batch in tqdm(dataloader):
        batch_output['bbox'] = batch['bbox']
        batch_output['crs'] = batch['crs']
        inputs = batch["image"].to(device)
        with autocast(device_type=device.type):
            outputs = model(inputs)
        if single_class_mode:
            outputs = outputs.squeeze(dim=0)
        else:
            outputs = F.softmax(outputs, dim=1).squeeze(dim=0)
        outputs = outputs[:, pad:-pad, pad:-pad]
        outputs = torch.mul(outputs, window_spline_2d)
        if single_class_mode:
            outputs = torch.sigmoid(outputs)
        outputs = outputs.permute(1, 2, 0).cpu().numpy().astype('float16')
        batch_output['data'] = outputs
        yield batch_output


def main(params):
    """High-level pipeline.
    Runs a model checkpoint on non-labeled imagery and saves results to file.
    Args:
        params: configuration parameters
    """
    # Main params
    item_url = get_key_def('input_stac_item', params['inference'], expected_type=str)  #, to_path=True, validate_path_exists=True) TODO implement for url
    checkpoint = get_key_def('state_dict_path', params['inference'], expected_type=str, to_path=True, validate_path_exists=True)
    root = get_key_def('root_dir', params['inference'], default="data", to_path=True, validate_path_exists=True)
    model_name = get_key_def('model_name', params['model'], expected_type=str).lower()  # TODO couple with model_choice.py
    download_data = get_key_def('download_data', params['inference'], default=False, expected_type=bool)

    # Dataset params
    modalities = get_key_def('modalities', params['dataset'], default=("red", "blue", "green"), expected_type=Sequence)
    num_bands = len(modalities)
    class_keys = len(get_key_def('classes_dict', params['dataset']).keys())
    num_classes = class_keys + 1  # if class_keys == 1 else class_keys + 1  # +1 for background(multiclass mode)
    single_class_mode = False  # if num_classes > 2 else True TODO bug fix in GDL

    # Hardware
    num_devices = get_key_def('gpu', params['inference'], default=1, expected_type=(int, bool))  # TODO implement >1
    max_used_ram = get_key_def('max_used_ram', params['inference'], default=50, expected_type=int)
    if not (0 <= max_used_ram <= 100):
        raise ValueError(f'\nMax used ram parameter should be a percentage. Got {max_used_ram}.')
    max_used_perc = get_key_def('max_used_perc', params['inference'], default=25, expected_type=int)
    # list of GPU devices that are available and unused. If no GPUs, returns empty dict
    gpu_devices_dict = get_device_ids(num_devices, max_used_ram_perc=max_used_ram, max_used_perc=max_used_perc)
    device = set_device(gpu_devices_dict=gpu_devices_dict)
    num_workers_default = len(gpu_devices_dict.keys()) * 4 if len(gpu_devices_dict.keys()) > 1 else 4
    num_workers = get_key_def('num_workers', params['inference'], default=num_workers_default, expected_type=int)

    # Sampling, batching and augmentations configuration
    chip_size = get_key_def('chunk_size', params['inference'], default=None, expected_type=int)
    auto_chip_size = True if not chip_size else False
    auto_cs_threshold = get_key_def('auto_chunk_size_threshold', params['inference'], default=95, expected_type=int)
    stride_default = int(chip_size / 2) if chip_size else 256
    stride = get_key_def('stride', params['inference'], default=stride_default, expected_type=int)
    if chip_size and stride > chip_size*0.75:
        logging.warning(f"Setting a large stride (more than 75% of chip size) will interfere with "
                        f"spline window smoothing operations and may result in poor quality extraction.")
    pad = get_key_def('pad', params['inference'], default=16, expected_type=int)
    # for inference, batch size should be equal to number of GPUs being used TODO implement bs>1
    batch_size = num_devices if num_devices else 1
    # TODO implement with hydra for all possible ttach transforms
    tta_transforms = get_key_def('tta_transforms', params['inference'], default="horizontal_flip", expected_type=str)
    tta_merge_mode = get_key_def('tta_merge_mode', params['inference'], default="max", expected_type=str)

    seed = 123
    seed_everything(seed)

    try:
        model = InferenceTask.load_from_checkpoint(checkpoint)
    except (KeyError, AssertionError) as e:
        logging.warning(f"\nModel checkpoint is not compatible with pytorch-ligthning's load_from_checkpoint method:\n"
                        f"Key error: {e}\n"
                        f"Will try to use geo-deep-learning to pytorch lightning adapter...")
        checkpoint = gdl2pl_checkpoint(checkpoint)
        # Create yaml to use pytorch lightning model management
        model = InferenceTask.load_from_checkpoint(checkpoint)

    model.freeze()
    model.eval()

    dm = InferenceDataModule(root_dir=root,
                             item_path=item_url,
                             outpath=root/f"{Path(item_url).stem}_pred.tif",
                             bands=modalities,
                             patch_size=chip_size,
                             stride=stride,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             download=download_data,
                             seed=seed,
                             pad=pad,
                             )  # TODO: test batch_size>1 for multi-gpu implementation
    dm.setup()

    model = model.to(device)

    h, w = [side for side in dm.inference_dataset.src.shape]

    if auto_chip_size and device != "cpu":
        chip_size = auto_chip_size_finder(datamodule=dm,
                                          device=device,
                                          model=model,
                                          tta_transforms=tta_transforms,
                                          single_class_mode=single_class_mode,
                                          max_used_ram=auto_cs_threshold,
                                          )
        # Set final chip size and stride from auto found values
        dm.patch_size = chip_size
        dm.stride = int(chip_size / 2)

    window_spline_2d = create_spline_window(chip_size).to(device)
    # Instantiate inference generator for looping over imagery chips
    eval_gen = run_eval_loop(
        model=model,
        dataloader=dm.predict_dataloader(),
        device=device,
        single_class_mode=single_class_mode,
        window_spline_2d=window_spline_2d,
        pad=dm.pad_size,
        tta_transforms=tta_transforms,
        tta_merge_mode=tta_merge_mode,
    )

    # Create a numpy memory map to write results from per-chip inference to full-size prediction
    tempfile = root / f"{Path(dm.inference_dataset.outpath).stem}.dat"
    fp = np.memmap(tempfile, dtype='float16', mode='w+', shape=(h, w, num_classes))
    for i, inference_prediction in enumerate(eval_gen):
        for bbox in inference_prediction['bbox']:
            col_min, *_, row_min = dm.sampler.chip_indices_from_bbox(bbox, dm.inference_dataset.src)
            right = col_min + chip_size if col_min + chip_size <= w else w
            bottom = row_min + chip_size if row_min + chip_size <= h else h
            # Write prediction on top of existing prediction for smoothing purposes
            fp[row_min:bottom, col_min:right, :] = \
                fp[row_min:bottom, col_min:right, :] + inference_prediction['data'][:bottom-row_min, :right-col_min]
    fp.flush()
    del fp

    # Read full-size prediction memory-map and write to final raster after argmax operation (discard confidence info)
    fp = np.memmap(tempfile, dtype='float16', mode='r', shape=(h, w, num_classes))
    pred_img = fp.argmax(axis=-1).astype('uint8')
    pred_img = pred_img[np.newaxis, :, :].astype(np.uint8)
    dm.inference_dataset.create_outraster()  # Unknown bug when moved further down
    outpath = Path(dm.inference_dataset.outpath)
    meta = rasterio.open(outpath).meta
    with rasterio.open(outpath, 'w+', **meta) as dest:
        dest.write(pred_img)

    # Postprocess final raster prediction (polygonization and simplification)
    inference_postprocess.main(params)

    logging.info(f'\nInference completed on {dm.inference_dataset.item_url}'
                 f'\nFinal prediction written to {dm.inference_dataset.outpath}')


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
        metavar="DIR")
    args = parser.parse_args()
    params = {"inference": {"input_stac_item": args.input_stac_item, "state_dict_path": args.state_dict_path,
              "root_dir": args.root_dir},
              "dataset": {'classes_dict': {'Building': 1}}}  # TODO softcode?
    main(params)

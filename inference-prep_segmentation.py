# Licensed under the MIT License.
# Authors: Victor Alhassan, Rémi Tavon

# Adapted from: https://github.com/microsoft/torchgeo/blob/3f7e525fbd01dddd25804e7a1b7634269ead1760/evaluate.py
# Also see: https://gist.github.com/calebrob6/7b226eb73877187f85fb5e1621bb7971

# Hardware requirements: 64 Gb RAM (cpu), 8 Gb GPU RAM.

"""CCMEO model inference script."""
import argparse
from pathlib import Path
from typing import Dict, Any, Sequence

import rasterio
from hydra.utils import instantiate
from omegaconf import DictConfig
from pandas.io.common import is_url
from pytorch_lightning import LightningModule, seed_everything
import torch
from torch import Tensor
from torch.hub import load_state_dict_from_url
from torchvision.transforms import Compose

from inference.InferenceDataModule import InferenceDataModule, preprocess, pad
from models.model_choice import define_model_architecture
from utils.geoutils import create_new_raster_from_base
from utils.logger import get_logger
from utils.utils import get_device_ids, get_key_def, set_device, override_model_params_from_checkpoint, \
    checkpoint_converter, extension_remover

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


def main(params):
    """High-level pipeline.
    Runs a model checkpoint on non-labeled imagery and saves results to file.
    Args:
        params: configuration parameters
    """
    seed = 123
    seed_everything(seed)

    logging.debug(f"\nSetting inference parameters")
    # Main params
    item_url = get_key_def('input_stac_item', params['inference'], expected_type=str, to_path=True, validate_path_exists=True)
    root = get_key_def('root_dir', params['inference'], default="inference", to_path=True)
    root.mkdir(exist_ok=True)
    data_dir = get_key_def('raw_data_dir', params['dataset'], default="data", to_path=True, validate_path_exists=True)
    models_dir = get_key_def('checkpoint_dir', params['inference'], default=root / 'checkpoints', to_path=True)
    models_dir.mkdir(exist_ok=True)
    outname = get_key_def('output_name', params['inference'], default=f"{Path(item_url).stem}_pred")
    outname = extension_remover(outname)
    outpath = root / f"{outname}.tif"
    checkpoint = get_key_def('state_dict_path', params['inference'], expected_type=str, to_path=True, validate_path_exists=True)
    download_data = get_key_def('download_data', params['inference'], default=False, expected_type=bool)
    save_heatmap_bool = get_key_def('save_heatmap', params['inference'], default=False, expected_type=bool)

    # Create yaml to use pytorch lightning model management
    if is_url(checkpoint):
        load_state_dict_from_url(url=checkpoint, map_location='cpu', model_dir=models_dir)
        checkpoint = models_dir / Path(checkpoint).name

    logging.debug(f"\nInstantiating model with pretrained weights from {checkpoint}")
    try:
        model = InferenceTask.load_from_checkpoint(checkpoint)
    except (KeyError, AssertionError) as e:
        logging.warning(f"\nModel checkpoint is not compatible with pytorch-ligthning's load_from_checkpoint method:\n"
                        f"Key error: {e}\n")
        try:
            logging.warning(
                f"\nTry to convert model checkpoint to be compatible with pytorch-ligthning's load_from_checkpoint method"
            )
            checkpoint = checkpoint_converter(in_pth_path=checkpoint, out_dir=models_dir)
            model = InferenceTask.load_from_checkpoint(checkpoint)
        except (KeyError, AssertionError) as e:
            logging.warning(f"\nModel checkpoint fail to be compatible with pytorch-ligthning's load_from_checkpoint method:\n"
                        f"Key error: {e}\n")
            raise e

    params = override_model_params_from_checkpoint(params=params, checkpoint_params=model.hparams)

    # Dataset params
    bands_requested = get_key_def('bands', params['dataset'], default=("red", "blue", "green"), expected_type=Sequence)

    # Sampling, batching and augmentations configuration
    batch_size = get_key_def('batch_size', params['inference'], default=None, expected_type=int)
    chip_size = get_key_def('chunk_size', params['inference'], default=512, expected_type=int)
    stride_default = int(chip_size / 2) if chip_size else 256
    stride = get_key_def('stride', params['inference'], default=stride_default, expected_type=int)
    if chip_size and stride > chip_size * 0.75:
        logging.warning(f"Setting a large stride (more than 75% of chip size) will interfere with "
                        f"spline window smoothing operations and may result in poor quality extraction.")
    pad_size = get_key_def('pad', params['inference'], default=16, expected_type=int)
    test_transforms_cfg = get_key_def('test_transforms', params['inference'],
                                  default=Compose([pad(pad_size, mode='reflect'), preprocess]),
                                  expected_type=(dict, DictConfig))

    # FIXME: temporary implementation of clahe enhancement applied to entire aoi, not tile by tile
    ####
    test_transforms_list = [transform for transform in test_transforms_cfg['transforms'] if 'enhance' not in transform['_target_']]
    test_transform_clahe_cfg = [transform for transform in test_transforms_cfg['transforms'] if 'enhance' in transform['_target_']]
    test_transforms_cfg['transforms'] = test_transforms_list
    test_transform_clahe = instantiate(test_transform_clahe_cfg[0]) if test_transform_clahe_cfg else None
    ####

    test_transforms = instantiate(test_transforms_cfg)

    dm = InferenceDataModule(root_dir=data_dir,
                             item_path=item_url,
                             outpath=outpath,
                             bands=bands_requested,
                             patch_size=chip_size,
                             stride=stride,
                             batch_size=batch_size,
                             num_workers=0,
                             download=download_data,
                             seed=seed,
                             pad=pad_size,
                             save_heatmap=save_heatmap_bool,
                             )
    dm.setup(test_transforms=test_transforms)

    # FIXME: temporary implementation of clahe enhancement applied to entire aoi, not tile by tile
    ####
    clip_limit = get_key_def('clahe_enhance_clip_limit', params['augmentation'], default=0, expected_type=int)
    if test_transform_clahe is not None and clip_limit > 0:
        if not dm.inference_dataset.download:
            raise NotImplementedError(
                f"Temporary CLAHE enhancement on entire AOI requires data to be downloaded locally\n"
                f"Got 'inference.download_data' = {dm.inference_dataset.download}"
            )
        logging.info(f"Will use CLAHE-enhanced AOI")
        for cname in dm.inference_dataset.bands:
            single_band = dm.inference_dataset.bands_dict[cname]['href']
            single_band_enhced = single_band.parent / f'{single_band.stem}_clahe{clip_limit}.tif'
            if single_band_enhced.is_file():
                logging.info(f"Using existing: {single_band_enhced}")
            else:
                logging.info(f"Enhancing {single_band}.\nOutput: {single_band_enhced}")
                single_band_rio = rasterio.open(single_band)
                single_band_np = single_band_rio.read()
                sample = {'image': torch.from_numpy(single_band_np)}
                sample = test_transform_clahe(sample)
                create_new_raster_from_base(single_band_rio, single_band_enhced, sample['image'])
            dm.inference_dataset.bands_dict[cname]['href'] = single_band_enhced
    ####


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

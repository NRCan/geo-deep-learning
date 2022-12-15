import logging
import os
from pathlib import Path
from typing import OrderedDict

import torch
import torchvision.models
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path, instantiate
from torch import nn

import models.unet
from models import unet
from models.model_choice import read_checkpoint, adapt_checkpoint_to_dp_model, define_model, define_model_architecture
from utils.utils import get_device_ids, set_device, ckpt_is_compatible


class TestModelsZoo(object):
    """Tests all geo-deep-learning's models instantiation and forward method"""
    def test_net(self) -> None:
        with initialize(config_path="../../config", job_name="test_ci"):
            for model_config in Path(to_absolute_path(f"../../config/model")).glob('*.yaml'):
                print(model_config)
                cfg = compose(config_name="gdl_config_template",
                              overrides=[f"model={model_config.stem}"],
                              return_hydra_config=True)
                hconf = HydraConfig()
                hconf.set_config(cfg)
                del cfg.loss.is_binary  # prevent exception at instantiation
                rand_img = torch.rand((2, 4, 64, 64))
                print(cfg.model._target_)
                if cfg.model._target_ == 'models.deeplabv3_dualhead.DeepLabV3_dualhead':
                    for layer in ['conv1', 'maxpool', 'layer2', 'layer3', 'layer4']:
                        logging.info(layer)
                        cfg.model.conc_point = layer
                        model = define_model_architecture(
                            net_params=cfg.model,
                            in_channels=4,
                            out_classes=4,
                        )
                        output = model(rand_img)
                        print(output.shape)
                else:
                    model = define_model_architecture(
                        net_params=cfg.model,
                        in_channels=4,
                        out_classes=4,
                    )
                    output = model(rand_img)
                    print(output.shape)


class TestCkptIsCompatible(object):
    """
    Tests checkpoint compatibility utility with incompatible checkpoint
    """
    filename = "tests/utils/gdl_pre20_test.pth.tar"
    assert not ckpt_is_compatible(filename)


class TestAdaptCheckpoint2DpModel(object):
    """
    Tests adapting a checkpoint to a DataParallel model, then loading it to model
    """
    filename = "tests/utils/gdl_current_test.pth.tar"
    filename_out = "tests/utils/gdl_current_test_temp.pth.tar"
    checkpoint = read_checkpoint(filename)
    dummy_model = torchvision.models.resnet18()
    checkpoint['model_state_dict'] = dummy_model.state_dict()
    torch.save(checkpoint, filename_out)
    checkpoint = read_checkpoint(filename_out)
    num_devices = 1
    gpu_devices_dict = get_device_ids(num_devices)
    device_ids = list(gpu_devices_dict.keys()) if len(gpu_devices_dict.keys()) >= 1 else None
    dummy_dp_model = nn.DataParallel(dummy_model, device_ids=device_ids)
    checkpoint = adapt_checkpoint_to_dp_model(checkpoint, dummy_dp_model)
    dummy_dp_model.load_state_dict(checkpoint['model_state_dict'])
    os.remove(filename_out)


class TestDefineModelMultigpu(object):
    """
    Tests defining model architecture with weights from provided checkpoint and pushing to multiple devices if possible
    """
    dummy_model = unet.UNet(4, 4, True, 0.5)
    filename = "tests/utils/gdl_current_test.pth.tar"
    filename_out = "tests/utils/gdl_current_test_temp.pth.tar"
    checkpoint = read_checkpoint(filename)
    checkpoint['model_state_dict'] = dummy_model.state_dict()
    torch.save(checkpoint, filename_out)

    gpu_devices_dict = get_device_ids(4)
    device = set_device(gpu_devices_dict=gpu_devices_dict)
    if len(gpu_devices_dict.keys()) == 0:
        logging.critical(f"No GPUs available. Cannot perform multi-gpu testing.")
    else:
        define_model(
            net_params={'_target_': 'models.unet.UNet'},
            in_channels=4,
            out_classes=4,
            main_device=device,
            devices=list(gpu_devices_dict.keys()),
            state_dict_path=filename_out,
            state_dict_strict_load=True,
        )
    os.remove(filename_out)

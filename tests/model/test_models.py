import logging
import os
from pathlib import Path

import torch
import torchvision.models
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from torch import nn

from models import unet
from models.model_choice import read_checkpoint, adapt_checkpoint_to_dp_model, define_model, define_model_architecture
from utils.optimizer import create_optimizer
from utils.utils import get_device_ids, set_device


class Test_Models_Zoo(object):
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
                if cfg.model['model_name'] == 'deeplabv3_resnet101_dualhead':
                    for layer in ['conv1', 'maxpool', 'layer2', 'layer3', 'layer4']:
                        logging.info(layer)
                        model = define_model_architecture(
                            model_name=cfg.model['model_name'],
                            num_bands=4,
                            num_channels=4,
                            conc_point=layer,
                        )
                        output = model(rand_img)
                        print(output.shape)
                else:
                    model = define_model_architecture(
                        model_name=cfg.model['model_name'],
                        num_bands=4,
                        num_channels=4,
                    )
                    output = model(rand_img)
                    print(output.shape)


class test_read_checkpoint():
    """
    Tests reading a checkpoint saved outside GDL into memory
    """
    dummy_model = torchvision.models.resnet18()
    dummy_optimizer = create_optimizer(dummy_model.parameters())
    filename = "test.pth.tar"
    torch.save(dummy_model.state_dict(), filename)
    read_checkpoint(filename)
    # test gdl's checkpoints at version <=2.0.1
    torch.save({'epoch': 999,
                'params': {'model': 'resnet18'},
                'model': dummy_model.state_dict(),
                'best_loss': 0.1,
                'optimizer': dummy_optimizer.state_dict()}, filename)
    read_checkpoint(filename)
    os.remove(filename)


class test_adapt_checkpoint_to_dp_model():
    """
    Tests adapting a generic checkpoint to a DataParallel model, then loading it to model
    """
    dummy_model = torchvision.models.resnet18()
    filename = "test.pth.tar"
    num_devices = 1
    gpu_devices_dict = get_device_ids(num_devices)
    torch.save(dummy_model.state_dict(), filename)
    checkpoint = read_checkpoint(filename)
    device_ids = list(gpu_devices_dict.keys()) if len(gpu_devices_dict.keys()) >= 1 else None
    dummy_dp_model = nn.DataParallel(dummy_model, device_ids=device_ids)
    checkpoint = adapt_checkpoint_to_dp_model(checkpoint, dummy_dp_model)
    dummy_dp_model.load_state_dict(checkpoint['model_state_dict'])
    os.remove(filename)


class test_define_model_multigpu():
    """
    Tests defining model architecture with weights from provided checkpoint and pushing to multiple devices if possible
    """
    dummy_model = unet.UNet(4, 4, True, 0.5)
    filename = "test.pth.tar"
    torch.save(dummy_model.state_dict(), filename)

    gpu_devices_dict = get_device_ids(4)
    device = set_device(gpu_devices_dict=gpu_devices_dict)
    if len(gpu_devices_dict.keys()) == 0:
        logging.critical(f"No GPUs available. Cannot perform multi-gpu testing.")
    else:
        define_model(
            model_name='unet',
            num_bands=4,
            num_classes=4,
            dropout_prob=0.5,
            conc_point=None,
            main_device=device,
            devices=list(gpu_devices_dict.keys()),
            state_dict_path=filename,
            state_dict_strict_load=True,
        )

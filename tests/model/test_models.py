import logging
import os
from pathlib import Path

import torch
import torchvision.models
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path, instantiate
from torch import nn

import models.unet
from models import unet
from models.model_choice import read_checkpoint, adapt_checkpoint_to_dp_model, define_model, define_model_architecture
from utils.utils import get_device_ids, set_device


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


class TestReadCheckpoint(object):
    """
    Tests reading a checkpoint saved outside GDL into memory
    """
    var = 4
    dummy_model = models.unet.UNetSmall(classes=var, in_channels=var)
    dummy_optimizer = instantiate({'_target_': 'torch.optim.Adam'}, params=dummy_model.parameters())
    filename = "test.pth.tar"
    torch.save(dummy_model.state_dict(), filename)
    read_checkpoint(filename)
    # test gdl's checkpoints at version <=2.0.1
    torch.save({'epoch': 999,
                'params': {
                    'global': {'num_classes': var, 'model_name': 'unet_small', 'number_of_bands': var}
                },
                'model': dummy_model.state_dict(),
                'best_loss': 0.1,
                'optimizer': dummy_optimizer.state_dict()}, filename)
    read_checkpoint(filename)
    os.remove(filename)


class TestAdaptCheckpoint2DpModel(object):
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


class TestDefineModelMultigpu(object):
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
        checkpoint = read_checkpoint(filename)
        model = define_model(
            net_params={'_target_': 'models.unet.UNet'},
            in_channels=4,
            out_classes=4,
            main_device=device,
            devices=list(gpu_devices_dict.keys()),
            checkpoint_dict=checkpoint
        )

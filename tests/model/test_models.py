import logging
from collections import OrderedDict
from pathlib import Path

import torch
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path

from models.model_choice import net


class Test_Models(object):
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
                        model, model_name, criterion, optimizer, lr_scheduler, device, gpu_devices_dict = net(
                            model_name=cfg.model['model_name'],
                            num_bands=4,
                            num_channels=4,
                            num_devices=0,
                            net_params={'training': None, 'optimizer': {'params': None},
                                        'scheduler': {'params': None}},
                            inference_state_dict=None,
                            conc_point=layer,
                            loss_fn={'_target_': 'torch.nn.CrossEntropyLoss'},
                            optimizer='sgd',
                        )
                        output = model(rand_img)
                        print(output.shape)
                else:
                    model, model_name, criterion, optimizer, lr_scheduler, device, gpu_devices_dict = net(
                        model_name=cfg.model['model_name'],
                        num_bands=4,
                        num_channels=4,
                        num_devices=0,
                        net_params={'training': None, 'optimizer': {'params': None},
                                    'scheduler': {'params': None}},
                        inference_state_dict=None,
                        loss_fn={'_target_': 'torch.nn.CrossEntropyLoss'},
                        optimizer='sgd',
                        )
                    output = model(rand_img)
                    print(output.shape)

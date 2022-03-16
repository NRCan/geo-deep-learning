import logging
from collections import OrderedDict
from pathlib import Path

import torch
import torchvision.models
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path

from models.model_choice import net


class Test_Models(object):
    """Tests geo-deep-learning's pipeline"""
    def test_net(self) -> None:
        with initialize(config_path="../../config", job_name="test_ci"):
            for model_config in Path(to_absolute_path(f"../../config/model")).glob('*.yaml'):
                if model_config.stem in ['fastrcnn', 'fcn_resnet101', 'ternauset', 'inception']:
                    logging.warning(f"These models will be removed in PR #289")  # TODO remove when #289 is merged
                    continue
                print(model_config)
                cfg = compose(config_name="gdl_config_template",
                              overrides=[f"model={model_config.stem}"],
                              return_hydra_config=True)
                hconf = HydraConfig()
                hconf.set_config(cfg)
                del cfg.loss.is_binary  # prevent exception at instantiation
                rand_img = torch.rand((2, 4, 64, 64))
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
                if isinstance(output, OrderedDict):  # TODO temporary fix for torchvision models
                    output = output['out']
                logging.info(output.shape)
import torch
import torch.nn as nn
from collections import OrderedDict
from models import TernausNet, unet, checkpointed_unet, inception, densenet, resnet


def maxpool_level(model, num_bands, size):
    """Calculate and return the number of maxpool inside the model definition.
    This function is useful during classification in order to calculate the number of pixel required as context.
    """
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))

    input_size = (num_bands, size, size)
    x = torch.rand(1, *input_size).type(torch.FloatTensor)

    summary = OrderedDict()
    hooks = []
    model.apply(register_hook)
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    maxpool_count = 0
    for layer in summary:
        if layer.startswith("MaxPool2d"):
            maxpool_count += 1
    return {'MaxPoolCount': maxpool_count}


def net(net_params, rtn_level=False):
    """Define the neural net"""
    model_name = net_params['global']['model_name'].lower()
    state_dict_path = ''
    num_classes = net_params['global']['num_classes']
    number_of_bands = net_params['global']['number_of_bands']

    # Semantic Segmentation models
    if model_name == 'unetsmall':
        model = unet.UNetSmall(num_classes, number_of_bands, net_params['models']['unetsmall']['dropout'],
                               net_params['models']['unetsmall']['probability'])
        if net_params['models']['unetsmall']['pretrained']:
            state_dict_path = net_params['models']['unetsmall']['pretrained']

    elif model_name == 'unet':
        model = unet.UNet(num_classes, number_of_bands, net_params['models']['unet']['dropout'],
                          net_params['models']['unet']['probability'])
        if net_params['models']['unet']['pretrained']:
            state_dict_path = net_params['models']['unet']['pretrained']

    elif model_name == 'ternausnet':
        model = TernausNet.ternausnet(num_classes, net_params['models']['ternausnet']['pretrained'])

    elif model_name == 'checkpointed_unet':
        model = checkpointed_unet.UNetSmall(num_classes, number_of_bands, net_params['models']['checkpointed_unet']['dropout'],
                                            net_params['models']['checkpointed_unet']['probability'])
        if net_params['models']['checkpointed_unet']['pretrained']:
            state_dict_path = net_params['models']['checkpointed_unet']['pretrained']

    # Classification Task models
    elif model_name == 'inception':
        model = inception.CustomInception3(num_classes, number_of_bands)
        if net_params['models']['inception']['pretrained']:
            state_dict_path = net_params['models']['inception']['pretrained']

    elif model_name == 'resnet18':
        model = resnet.resnet18(num_classes=num_classes, band_num=number_of_bands)
        if net_params['models']['resnet18']['pretrained']:
            state_dict_path = net_params['models']['resnet18']['pretrained']
    elif model_name == 'resnet34':
        model = resnet.resnet34(num_classes=num_classes, band_num=number_of_bands)
        if net_params['models']['resnet34']['pretrained']:
            state_dict_path = net_params['models']['resnet34']['pretrained']
    elif model_name == 'resnet50':
        model = resnet.resnet50(num_classes=num_classes, band_num=number_of_bands)
        if net_params['models']['resnet50']['pretrained']:
            state_dict_path = net_params['models']['resnet50']['pretrained']
    elif model_name == 'resnet101':
        model = resnet.resnet101(num_classes=num_classes, band_num=number_of_bands)
        if net_params['models']['resnet101']['pretrained']:
            state_dict_path = net_params['models']['resnet101']['pretrained']
    elif model_name == 'resnet152':
        model = resnet.resnet152(num_classes=num_classes, band_num=number_of_bands)
        if net_params['models']['resnet152']['pretrained']:
            state_dict_path = net_params['models']['resnet152']['pretrained']

    elif model_name == 'densenet121':
        model = densenet.densenet121(num_classes=num_classes, band_num=number_of_bands)
        if net_params['models']['densenet121']['pretrained']:
            state_dict_path = net_params['models']['densenet121']['pretrained']
    elif model_name == 'densenet161':
        model = densenet.densenet161(num_classes=num_classes, band_num=number_of_bands)
        if net_params['models']['densenet161']['pretrained']:
            state_dict_path = net_params['models']['densenet161']['pretrained']
    elif model_name == 'densenet169':
        model = densenet.densenet169(num_classes=num_classes, band_num=number_of_bands)
        if net_params['models']['densenet169']['pretrained']:
            state_dict_path = net_params['models']['densenet169']['pretrained']
    elif model_name == 'densenet201':
        model = densenet.densenet201(num_classes=num_classes, band_num=number_of_bands)
        if net_params['models']['densenet201']['pretrained']:
            state_dict_path = net_params['models']['densenet201']['pretrained']

    else:
        raise ValueError('The model name in the config.yaml is not defined.')

    if rtn_level:
        lvl = maxpool_level(model, net_params['global']['number_of_bands'], 256)
        return model, state_dict_path, lvl['MaxPoolCount']
    else:
        return model, state_dict_path

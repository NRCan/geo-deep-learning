from models import TernausNet, unet, checkpointed_unet, inception


def net(net_params, inference=False):
    """Define the neural net"""
    model_name = net_params['global']['model_name'].lower()
    state_dict_path = ''
    if model_name == 'unetsmall':
        model = unet.UNetSmall(net_params['global']['num_classes'],
                                       net_params['global']['number_of_bands'],
                                       net_params['models']['unetsmall']['dropout'],
                                       net_params['models']['unetsmall']['probability'])
        if net_params['models']['unetsmall']['pretrained']:
            state_dict_path = net_params['models']['unetsmall']['pretrained']
    elif model_name == 'unet':
        model = unet.UNet(net_params['global']['num_classes'],
                                  net_params['global']['number_of_bands'],
                                  net_params['models']['unet']['dropout'],
                                  net_params['models']['unet']['probability'])
        if net_params['models']['unet']['pretrained']:
            state_dict_path = net_params['models']['unet']['pretrained']
    elif model_name == 'ternausnet':
        model = TernausNet.ternausnet(net_params['global']['num_classes'],
                                      net_params['models']['ternausnet']['pretrained'])
    elif model_name == 'checkpointed_unet':
        model = checkpointed_unet.UNetSmall(net_params['global']['num_classes'],
                                       net_params['global']['number_of_bands'],
                                       net_params['models']['unetsmall']['dropout'],
                                       net_params['models']['unetsmall']['probability'])
        if net_params['models']['unetsmall']['pretrained']:
            state_dict_path = net_params['models']['unetsmall']['pretrained']
    elif model_name == 'inception':
        model = inception.Inception3(net_params['global']['num_classes'],
                                     net_params['global']['number_of_bands'])
        if net_params['models']['inception']['pretrained']:
            state_dict_path = net_params['models']['inception']['pretrained']
    else:
        raise ValueError('The model name in the config.yaml is not defined.')
    if inference:
        state_dict_path = net_params['inference']['state_dict_path']

    return model, state_dict_path, model_name

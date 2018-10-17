from models import TernausNet, unet


def net(net_params):
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
    else:
        raise ValueError('The model name in the yaml.config is not defined.')
    return model, state_dict_path

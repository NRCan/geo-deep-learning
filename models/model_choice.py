import torchvision.models as models
from models import TernausNet, unet, checkpointed_unet, inception
from utils.utils import chop_layer, load_from_checkpoint

# TODO: add in read me that parameter 'pretrained' means user provides his own .pth. Otherwise, will load pretrained weight (ex.:COCO2017)
# TODO: check config.yaml structure. Could be simplified (ex.: single line for pretrained path
def net(net_params, inference=False):
    """Define the neural net"""
    model_name = net_params['global']['model_name'].lower()
    num_classes = net_params['global']['num_classes']
    msg = f'Number of bands specified incompatible with this model. Requires 3 band data.'
    state_dict_path = ''
    if model_name == 'unetsmall':
        model = unet.UNetSmall(num_classes,
                                       net_params['global']['number_of_bands'],
                                       net_params['models']['unetsmall']['dropout'],
                                       net_params['models']['unetsmall']['probability'])
        if net_params['models']['unetsmall']['pretrained']:
            state_dict_path = net_params['models']['unetsmall']['pretrained']
    elif model_name == 'unet':
        model = unet.UNet(num_classes,
                                  net_params['global']['number_of_bands'],
                                  net_params['models']['unet']['dropout'],
                                  net_params['models']['unet']['probability'])
        if net_params['models']['unet']['pretrained']:
            state_dict_path = net_params['models']['unet']['pretrained']
    elif model_name == 'ternausnet':
        model = TernausNet.ternausnet(num_classes,
                                      net_params['models']['ternausnet']['pretrained'])
    elif model_name == 'checkpointed_unet':
        model = checkpointed_unet.UNetSmall(num_classes,
                                       net_params['global']['number_of_bands'],
                                       net_params['models']['unetsmall']['dropout'],
                                       net_params['models']['unetsmall']['probability'])
        if net_params['models']['unetsmall']['pretrained']:
            state_dict_path = net_params['models']['unetsmall']['pretrained']
    elif model_name == 'inception':
        model = inception.Inception3(num_classes,
                                     net_params['global']['number_of_bands'])
        if net_params['models']['inception']['pretrained']:
            state_dict_path = net_params['models']['inception']['pretrained']
    elif model_name == 'fcn_resnet101':
        assert net_params['global']['number_of_bands'], msg
        model = models.segmentation.fcn_resnet50(pretrained=True, progress=True, num_classes=num_classes, aux_loss=None)

        if net_params['models']['fcn_resnet101']['pretrained']:
            state_dict_path = net_params['models']['fcn_resnet101']['pretrained']
    elif model_name == 'deeplabv3_resnet50':
        assert net_params['global']['number_of_bands'], msg
        model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True,
                                                       num_classes=num_classes, aux_loss=None)
        if net_params['models']['deeplabv3_resnet50']['pretrained']:
            state_dict_path = net_params['models']['deeplabv3_resnet50']['pretrained']
    elif model_name == 'deeplabv3_resnet101':
        assert net_params['global']['number_of_bands'], msg
        # pretrained on coco (21 classes)
        coco_model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True,
                                                        num_classes=21, aux_loss=None)
        model = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True,
                                                        num_classes=num_classes, aux_loss=None)

        # TODO: START HERE
        # TODO: is this best implementation? At least should become a function.
        chopped_dict = chop_layer(coco_model.state_dict(), layer_name='classifier.4')
        chopped_dict = chop_layer(coco_model.state_dict(), layer_name='aux_classifier.4')

        model_dict = model.state_dict()
        model_dict.update(chopped_dict)
        # load the new state dict
        model.load_state_dict(model_dict)

        if net_params['models']['deeplabv3_resnet101']['pretrained']:
            state_dict_path = net_params['models']['deeplabv3_resnet101']['pretrained']
    else:
        raise ValueError('The model name in the config.yaml is not defined.')
    if inference:
        state_dict_path = net_params['inference']['state_dict_path']

    return model, state_dict_path, model_name

## **Models available**

## Train from Scratch
- [Inception V3](https://arxiv.org/pdf/1512.00567.pdf)
- [Ternausnet](https://arxiv.org/abs/1801.05746.pdf)
- [Unet](https://arxiv.org/abs/1505.04597.pdf)
- Unet small (less deep version of [Unet](https://arxiv.org/abs/1505.04597.pdf))
- Checkpointed Unet (same as [Unet](https://arxiv.org/abs/1505.04597.pdf) small, but uses less GPU memory and recomputes data during the backward pass)

## Pre-trained (torch vision by default pretrained on coco dataset)
- [FCN with backbone resnet101](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
- [Deeplabv3 with backbone resnet101](https://arxiv.org/abs/1706.05587.pdf)
- Experimental: [Deeplabv3 with backbone resnet101](https://arxiv.org/abs/1706.05587.pdf)  adapted for RGB-NIR (4 Bands)

## Segmentation Models from Pytorch Library
Those models are pretrained on `imagenet`.
- [unet_pretrained](https://arxiv.org/abs/1801.05746.pdf)
- [pan_pretrained](https://arxiv.org/abs/1805.10180.pdf)
- [fpn_pretrained](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)
- [pspnet_pretrained](https://arxiv.org/abs/1612.01105.pdf)
- [deeplabv3+_pretrained](https://arxiv.org/pdf/1802.02611.pdf)
- [spacenet_unet_efficientnetb5_pretrained](https://github.com/SpaceNetChallenge/SpaceNet_SAR_Buildings_Solutions/tree/master/1-zbigniewwojna)
- [spacenet_unet_senet152_pretrained](https://github.com/SpaceNetChallenge/SpaceNet_SAR_Buildings_Solutions/tree/master/2-MaksimovKA)
- [spacenet_unet_baseline_pretrained]() (In the article of SpaceNet, the baseline is originally pretrained on `SN6 PS-RGB Imagery` if you want to you can give those weights in parameters, but we don't have it in **GDL**).

Models from this [library](https://github.com/qubvel/segmentation_models.pytorch) support any number of image bands and offers modular encoder architectures. Check the official [github repo](https://github.com/qubvel/segmentation_models.pytorch) for more details.  

If you want to add a new model base on this [library](https://github.com/qubvel/segmentation_models.pytorch), you can add it in the `dict` name `lm_smp` in [`model_choice.py`](model_choice.py). First, add the main architecture under the key `fct`. For the parameters, put the name of the parameters as a key and the value as the value of the `dict`, like follow:
```
'pan_pretrained': {
        'fct': smp.PAN, 'params': {
            'encoder_name':'se_resnext101_32x4d',
        }},
```
Take note that the parameters `encoder_weights`, `in_channels`, `classes` and `activation` are taken care in the [`net()`](model_choice.py#L144) function of the [`model_choice.py`](model_choice.py).

## New Models
To add a new model, be sure that it can be called in `model_choice.py` with all the particularities needed to be trained.

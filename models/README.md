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
- [unet_pretrained](https://arxiv.org/abs/1801.05746.pdf)
- [pan_pretrained](https://arxiv.org/abs/1805.10180.pdf)
- [fpn_pretrained](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)
- [pspnet_pretrained](https://arxiv.org/abs/1612.01105.pdf)
- [deeplabv3+_pretrained](https://arxiv.org/pdf/1802.02611.pdf)

Models from this [library](https://github.com/qubvel/segmentation_models.pytorch) support any number of image bands and offers modular encoder architectures. Check the official [github repo](https://github.com/qubvel/segmentation_models.pytorch) for more details.  

## New Models
To add a new model, be sure that it can be call in `model_choice.py` with all the particularities needed to be train.

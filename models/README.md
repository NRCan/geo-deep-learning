make a list of all the models we have plus the ref for each with maybe a photo

## Models available

Models: Train from Scratch
- [Unet](https://arxiv.org/abs/1505.04597)
- Unet small (less deep version of Unet)
- Checkpointed Unet (same as Unet small, but uses less GPU memory and recomputes data during the backward pass)
- [Ternausnet](https://arxiv.org/abs/1801.05746)

Models: Pre-trained (torch vision)
- [Deeplabv3 (backbone: resnet101, optional: pretrained on coco dataset)](https://arxiv.org/abs/1706.05587)
- Experimental: Deeplabv3 (default: pretrained on coco dataset) adapted for RGB-NIR(4 Bands) supported
- [FCN (backbone: resnet101, optional: pretrained on coco dataset)](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

Models: Segmentation Models Pytorch Library

The following highly configurable models are offered from this easy to use [library](https://github.com/qubvel/segmentation_models.pytorch).

- unet_pretrained
- [pan_pretrained](https://arxiv.org/abs/1805.10180)
- [fpn_pretrained](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)
- [pspnet_pretrained](https://arxiv.org/abs/1612.01105)
- [deeplabv3+_pretrained](https://arxiv.org/pdf/1802.02611.pdf)

Models from this library support any number of image bands and offers modular encoder architectures. Check the official [github repo](https://github.com/qubvel/segmentation_models.pytorch) for more details.  

.. _modelindex:

Models
++++++

Most modern deep learning models are based on artificial neural networks,
specifically convolutional neural networks (**CNN**).
During the training process, algorithms use unknown elements in the input
distribution to extract features, group objects, and discover useful data patterns.
Much like training machines for self-learning, this occurs at multiple levels,
using the algorithms to make a inference on a image without annotation at the end.

While not one network is considered perfect, some algorithms are better suited to 
perform specific tasks or extract specific patterns.

Here some models available in **GDL**. 

Segmentation
------------

`UNet <https://arxiv.org/abs/1505.04597.pdf>`_
=====

*Unet* is a fully convolution neural network for image semantic segmentation.
Consist of encoder and decoder parts connected with skip connections.
Encoder extract features of different spatial resolution (skip connections) 
which are used by decoder to define accurate segmentation mask. 
Use concatenation for fusing decoder blocks with skip connections.

Here some implementation found in the config `model <https://github.com/NRCan/geo-deep-learning/tree/develop/config/model>`_ folder.

.. autoclass:: models.unet.UNetSmall
   :members:
   :special-members:

.. autoclass:: models.unet.UNet
   :members:
   :special-members:

And an implementation from 
`smp model library <https://smp.readthedocs.io/en/latest/models.html#unet>`_. 
Plus, the folder contains some specific combinaisons the *smp model* like :
unet++, unet pretrained on imagenet, unet with senet154 encoder, unet with resnext101 encoder and more.
We invite you to see the config `model <https://github.com/NRCan/geo-deep-learning/tree/develop/config/model>`_ 
folder to the complete list on different combinaisons.

`DeepLabV3 <https://arxiv.org/abs/1706.05587>`_
==========

*DeepLabV3* implementation of *Rethinking Atrous Convolution for Semantic Image Segmentation* paper from 
`smp model library <https://smp.readthedocs.io/en/latest/models.html#deeplabv3>`_.

Also from the same library, another version of *DeepLabV3*, named *DeepLabV3+* of the
*Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation* paper.

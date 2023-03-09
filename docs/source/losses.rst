.. _lossindex:

Losses
++++++

Loss functions are one of the most important aspects of neural networks,
as they, along with the optimization functions, are directly responsible
for fitting the model to the given training data.

The choice of the loss function is very important, since each usecase
and each model will probably require a different loss.

Here some losses available in **GDL**. 

Segmentation
------------

`Cross Entropy Loss (multiclass) <https://en.wikipedia.org/wiki/Cross_entropy>`_
================================

Also called logarithmic loss, log loss or logistic loss.
Each predicted class probability is compared to the current class desired
output 0 or 1 and a score/loss is calculated that penalizes the probability
based on how far it is form the actual expected value.

We are using the smp losses implementation
`here <https://smp.readthedocs.io/en/latest/losses.html#softcrossentropyloss>`_.


`Boundary Loss (multiclass) <https://arxiv.org/abs/1905.07852.pdf>`_
===========================

A differentiable surrogate of a metric accounting accuracy of boundary detection.

.. autoclass:: losses.boundary_loss.BoundaryLoss
   :members:
   :special-members:

`Dice Loss (binary & multiclass) <https://medium.com/ai-salon/understanding-dice-loss-for-crisp-boundary-detection-bb30c2e5f62b>`_
================================

A loss using the dice coefficient that estimates the fraction of contour
length that needs correction.

For the binary and multiclasses version, the configuration call the ``DiceLoss`` from 
`smp losses library <https://smp.readthedocs.io/en/latest/losses.html#softcrossentropyloss>`_.

`Focal Loss (multiclass) <https://arxiv.org/pdf/1708.02002.pdf>`_
========================

The focal loss focuses training on a sparse set of hard examples
and prevents the vast number of easy negatives from overwhelming the
detector during training.  

The configuration call the ``FocalLoss`` from 
`smp losses library <https://smp.readthedocs.io/en/latest/losses.html#focalloss>`_.

`Lovasz-Softmax Loss (binary & multiclass) <https://arxiv.org/pdf/1705.08790.pdf>`_
==========================================

A tractable surrogate for the optimization of the intersection-over-union 
measure in neural networks.

For the binary and multiclasses version, the configuration call the ``LovaszLoss`` from 
`smp losses library <https://smp.readthedocs.io/en/latest/losses.html#lovaszloss>`_.

`Ohem Loss (multiclass) <https://github.com/openseg-group/OCNet.pytorch>`_
=======================

A loss that calculate where the hard pixels are defined as the pixels 
associated with probabilities smaller than a certain value over the 
correct classes.

.. autoclass:: losses.ohem_loss.OhemCrossEntropy2d
   :members:
   :special-members:

`Softbce Loss (binary) <https://smp.readthedocs.io/en/latest/losses.html#softbcewithlogitsloss>`_
=======================

Drop-in replacement for ``torch.nn.BCEWithLogitsLoss``
with few additions: ``ignore_index`` and ``label_smoothing``

We are using the segmentation models pytorch implementation
`here <https://smp.readthedocs.io/en/latest/losses.html#softcrossentropyloss>`_.

`Duo Loss (multiclass) <https://github.com/openseg-group/OCNet.pytorch>`_
=======================

This loss is a combinaison of the :py:func:`losses.lovasz_loss.LovaszSoftmax` and 
:py:func:`losses.boundary_loss.BoundaryLoss`.

.. autoclass:: losses.duo_loss.DuoLoss
   :members:
   :special-members:

# Segmentation Losses

This document provides an overview of the different loss functions used available in `segmentation_losses.py`.

## Loss Functions

Here are the loss functions available: 

1. `DiceLoss`: This is a Dice loss function. It can be used in binary, multi-class, and multi-label modes. The mode can be specified as an argument.

2. `FocalLoss`: This is a Focal loss function. It can be used in binary, multi-class, and multi-label modes. The mode can be specified as an argument.

3. `JaccardLoss`: This is a Jaccard loss function, also known as Intersection over Union (IoU) loss. It can be used in binary, multi-class, and multi-label modes. The mode can be specified as an argument.

4. `LovaszLoss`: This is a Lovasz loss function. It can be used in binary, multi-class, and multi-label modes. The mode can be specified as an argument.

5. `SoftCELoss`: This is a Soft Cross Entropy loss function. It can only be used in multi-class mode and does not take a "mode" argument.

6. `SoftBCELoss`: This is a Soft Binary Cross Entropy loss function. It can only be used in binary mode and does not take a "mode" argument.

7. `MCCLoss`: This is a Matthews Correlation Coefficient loss function. It can only be used in binary mode and does not take a "mode" argument.





## Variable Names

When implementing these loss functions in the `segmentation_segformer` script, you can use the following variable names:

- `cross_entropy_loss`: Variable to store the cross-entropy loss value.
- `dice_loss`: Variable to store the dice loss value.
- `focal_loss`: Variable to store the focal loss value.
- `lovasz_softmax_loss`: Variable to store the Lovász-Softmax loss value.

Make sure to initialize these variables appropriately and update them during the training process.

For more detailed information on each loss function and their implementation, please refer to the corresponding sections in this document.



## Usage in segmentation_segformer.py

To use these loss functions in the `segmentation_segformer.py` script, you can import them as follows:

```python
from segmentation_losses import DiceLoss, FocalLoss, JaccardLoss, LovaszLoss, SoftCELoss, SoftBCELoss, MCCLoss
```

Then, you can use them in your model training code. For example, if you want to use `DiceLoss` as your loss function, you can do:

```python
loss = DiceLoss()
```

And then use `loss` in your training loop to compute the loss and backpropagate the gradients.
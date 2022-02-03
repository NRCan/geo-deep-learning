## **Losses available**
### - [Cross Entropy Loss](https://en.wikipedia.org/wiki/Cross_entropy)
Also called logarithmic loss, log loss or logistic loss. Each predicted class probability is compared to the current class desired output 0 or 1 and a score/loss is calculated that penalizes the probability based on how far it is form the actual expected value.
### - [Boundary Loss](https://arxiv.org/abs/1905.07852.pdf)
A differentiable surrogate of a metric accounting accuracy of boundary detection.
### - [Dice Loss](https://medium.com/ai-salon/understanding-dice-loss-for-crisp-boundary-detection-bb30c2e5f62b)
A loss using the dice coefficient that estimates the fraction of contour length that needs correction.
### - [Focal Loss](https://arxiv.org/pdf/1708.02002.pdf)
The focal loss focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training.  
### - [Lovasz-Softmax Loss](https://arxiv.org/pdf/1705.08790.pdf)
 A tractable surrogate for the optimization of the intersection-over-union measure in neural networks.
### - [Ohem Loss](https://github.com/openseg-group/OCNet.pytorch)
A loss that calculate where the hard pixels are defined as the pixels associated with probabilities smaller than a certain value over the correct classes.

## New Losses
To add a new loss, be sure that the loss is callable in the `__init__.py`.

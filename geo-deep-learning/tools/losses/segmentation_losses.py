import segmentation_models_pytorch as smp

# Binary, Multi-class, Multi-label mode, takes argument "mode".
DiceLoss = smp.losses.DiceLoss()
FocalLoss = smp.losses.FocalLoss()
JaccardLoss = smp.losses.JaccardLoss()
LovaszLoss = smp.losses.LovaszLoss()

# Multi-class mode only, does not take argument "mode".
SoftCELoss = smp.losses.SoftCrossEntropyLoss()

# Binary mode only, does not take argument "mode".
SoftBCELoss = smp.losses.SoftBCEWithLogitsLoss()
MCCLoss = smp.losses.MCCLoss()
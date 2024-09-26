import segmentation_models_pytorch as smp

# Binary, Multi-class, Multi-label mode, takes argument "mode".
DiceLoss = smp.losses.DiceLoss(mode="multiclass")
FocalLoss = smp.losses.FocalLoss(mode="multiclass")
JaccardLoss = smp.losses.JaccardLoss(mode="multiclass")
LovaszLoss = smp.losses.LovaszLoss(mode="multiclass")

# Multi-class mode only, does not take argument "mode".
SoftCELoss = smp.losses.SoftCrossEntropyLoss()

# Binary mode only, does not take argument "mode".
SoftBCELoss = smp.losses.SoftBCEWithLogitsLoss()
MCCLoss = smp.losses.MCCLoss()
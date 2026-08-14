"""Comprehensive segmentation metrics for binary and multi-class segmentation."""

import logging
from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)

logger = logging.getLogger(__name__)


class SegmentationMetrics(Metric):
    """
    Comprehensive segmentation metrics for per-class evaluation.
    
    Computes IoU, Dice/F1, Precision, Recall, and diagnostic statistics
    for each class in binary or multi-class segmentation.
    """

    def __init__(
        self,
        num_classes: int,
        class_labels: list[str] | None = None,
        ignore_index: int | None = None,
        threshold: float = 0.5,
        compute_diagnostics: bool = True,
    ) -> None:
        """
        Initialize segmentation metrics.

        Args:
            num_classes: Number of classes (including background)
            class_labels: Optional list of class names
            ignore_index: Index to ignore in metrics computation
            threshold: Threshold for binary classification (unused for multiclass)
            compute_diagnostics: Whether to compute diagnostic statistics

        """
        super().__init__()
        self.num_classes = num_classes
        self.class_labels = class_labels or [str(i) for i in range(num_classes)]
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.compute_diagnostics = compute_diagnostics

        # Use binary metrics if num_classes == 2 (background + 1 class)
        self.is_binary = num_classes == 2

        if self.is_binary:
            # Binary metrics for class 1 (positive class)
            self.iou = BinaryJaccardIndex(ignore_index=ignore_index)
            self.dice = BinaryF1Score(ignore_index=ignore_index)
            self.precision = BinaryPrecision(ignore_index=ignore_index)
            self.recall = BinaryRecall(ignore_index=ignore_index)
        else:
            # Multiclass metrics
            self.iou = MulticlassJaccardIndex(
                num_classes=num_classes,
                ignore_index=ignore_index,
                average=None,  # Per-class
            )
            self.dice = MulticlassF1Score(
                num_classes=num_classes,
                ignore_index=ignore_index,
                average=None,  # Per-class
            )
            self.precision = MulticlassPrecision(
                num_classes=num_classes,
                ignore_index=ignore_index,
                average=None,  # Per-class
            )
            self.recall = MulticlassRecall(
                num_classes=num_classes,
                ignore_index=ignore_index,
                average=None,  # Per-class
            )

        # State for diagnostics
        if compute_diagnostics:
            self.add_state("total_pixels", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state("tp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
            self.add_state("fp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
            self.add_state("fn", default=torch.zeros(num_classes), dist_reduce_fx="sum")
            self.add_state("tn", default=torch.zeros(num_classes), dist_reduce_fx="sum")
            self.add_state("pred_pixels", default=torch.zeros(num_classes), dist_reduce_fx="sum")
            self.add_state("true_pixels", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update metrics with new predictions and targets.

        Args:
            preds: Predictions (class indices) [B, H, W]
            target: Ground truth (class indices) [B, H, W]

        """
        # Update main metrics
        if self.is_binary:
            # For binary, metrics expect flattened tensors
            self.iou.update(preds, target)
            self.dice.update(preds, target)
            self.precision.update(preds, target)
            self.recall.update(preds, target)
        else:
            self.iou.update(preds, target)
            self.dice.update(preds, target)
            self.precision.update(preds, target)
            self.recall.update(preds, target)

        # Update diagnostic statistics
        if self.compute_diagnostics:
            # Create valid mask (exclude ignore_index)
            if self.ignore_index is not None:
                valid_mask = target != self.ignore_index
            else:
                valid_mask = torch.ones_like(target, dtype=torch.bool)

            valid_preds = preds[valid_mask]
            valid_target = target[valid_mask]

            self.total_pixels += valid_mask.sum()

            # Compute per-class statistics
            for class_idx in range(self.num_classes):
                pred_class = (valid_preds == class_idx)
                true_class = (valid_target == class_idx)

                self.tp[class_idx] += (pred_class & true_class).sum()
                self.fp[class_idx] += (pred_class & ~true_class).sum()
                self.fn[class_idx] += (~pred_class & true_class).sum()
                self.tn[class_idx] += (~pred_class & ~true_class).sum()
                self.pred_pixels[class_idx] += pred_class.sum()
                self.true_pixels[class_idx] += true_class.sum()

    def compute(self) -> dict[str, Any]:
        """
        Compute all metrics.

        Returns:
            Dictionary containing all computed metrics with descriptive keys

        """
        metrics = {}

        # Compute main metrics
        iou_values = self.iou.compute()
        dice_values = self.dice.compute()
        precision_values = self.precision.compute()
        recall_values = self.recall.compute()

        # Handle binary vs multiclass output format
        if self.is_binary:
            # Binary metrics return single values for the positive class
            # Store metrics for both background (class 0) and positive class (class 1)
            # For background, we can compute complement metrics
            metrics["iou_background"] = 1.0 - iou_values  # Approximation
            metrics["iou_" + self.class_labels[1]] = iou_values
            
            metrics["dice_background"] = 1.0 - dice_values  # Approximation
            metrics["dice_" + self.class_labels[1]] = dice_values
            
            metrics["precision_" + self.class_labels[1]] = precision_values
            metrics["recall_" + self.class_labels[1]] = recall_values

            # Add mean metrics
            metrics["mean_iou"] = iou_values
            metrics["mean_dice"] = dice_values

        else:
            # Multiclass metrics return per-class tensors
            for i, label in enumerate(self.class_labels):
                metrics[f"iou_{label}"] = iou_values[i]
                metrics[f"dice_{label}"] = dice_values[i]
                metrics[f"precision_{label}"] = precision_values[i]
                metrics[f"recall_{label}"] = recall_values[i]

            # Add mean metrics
            metrics["mean_iou"] = iou_values.mean()
            metrics["mean_dice"] = dice_values.mean()

        # Add diagnostic statistics
        if self.compute_diagnostics:
            for i, label in enumerate(self.class_labels):
                metrics[f"tp_{label}"] = self.tp[i].float()
                metrics[f"fp_{label}"] = self.fp[i].float()
                metrics[f"fn_{label}"] = self.fn[i].float()
                metrics[f"tn_{label}"] = self.tn[i].float()
                
                # Predicted positive fraction for this class
                if self.total_pixels > 0:
                    metrics[f"pred_fraction_{label}"] = (
                        self.pred_pixels[i].float() / self.total_pixels.float()
                    )
                    metrics[f"true_fraction_{label}"] = (
                        self.true_pixels[i].float() / self.total_pixels.float()
                    )
                else:
                    metrics[f"pred_fraction_{label}"] = torch.tensor(0.0)
                    metrics[f"true_fraction_{label}"] = torch.tensor(0.0)

        return metrics

    def reset(self) -> None:
        """Reset all metrics."""
        super().reset()
        self.iou.reset()
        self.dice.reset()
        self.precision.reset()
        self.recall.reset()


def compute_metrics_from_logits(
    logits: Tensor,
    target: Tensor,
    num_classes: int,
    threshold: float = 0.5,
    class_labels: list[str] | None = None,
    ignore_index: int | None = None,
) -> dict[str, float]:
    """
    Compute comprehensive metrics from model logits.

    Args:
        logits: Model output logits [B, C, H, W] or [B, 1, H, W] for binary
        target: Ground truth [B, H, W] or [B, 1, H, W]
        num_classes: Number of classes (1 for binary, >1 for multiclass)
        threshold: Threshold for binary segmentation
        class_labels: Optional class names
        ignore_index: Optional index to ignore

    Returns:
        Dictionary of computed metrics

    """
    # Ensure target is [B, H, W]
    if target.dim() == 4:
        target = target.squeeze(1)
    target = target.long()

    # Convert logits to predictions
    if num_classes == 1:
        # Binary segmentation
        probs = torch.sigmoid(logits)
        preds = (probs.squeeze(1) > threshold).long()
        actual_num_classes = 2  # Background and foreground
    else:
        # Multi-class segmentation
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        actual_num_classes = num_classes

    # Compute metrics
    metrics_module = SegmentationMetrics(
        num_classes=actual_num_classes,
        class_labels=class_labels,
        ignore_index=ignore_index,
        threshold=threshold,
    )
    metrics_module.update(preds, target)
    return metrics_module.compute()

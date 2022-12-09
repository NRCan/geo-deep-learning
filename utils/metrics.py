from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def create_metrics_dict(num_classes: int) -> Dict:
    """
    Initializes a metrics dictionary.
    Args:
        num_classes: number of classes in the segmentation task. If it equals to 1, then converted to a binary problem.
    Returns:
        Initialized metrics dictionary.

    """
    assert num_classes > 0, f"Wrong number of classes: {num_classes}"
    num_classes = 2 if num_classes == 1 else num_classes
    metrics_dict = {"iou": AverageMeter(),
                    "precision": AverageMeter(),
                    "recall": AverageMeter(),
                    "fscore": AverageMeter()}

    for key in metrics_dict.copy().keys():
        per_class_metrics = {key + "_" + str(i): AverageMeter() for i in range(num_classes)}
        metrics_dict.update(per_class_metrics)

    metrics_dict['loss'] = AverageMeter()
    metrics_dict['iou_nonbg'] = AverageMeter()

    return metrics_dict


class AverageMeter:
    """
    Accumulates individual metrics calculations and their average.
    """
    def __init__(self):
        self.total_metrics_sum = 0
        self.total_batches = 0
        self.total_samples = 0
        self.avg = 0.0

    def update(self, val: float, batch_size: int) -> None:
        self.total_metrics_sum += val
        self.total_batches += 1
        self.total_samples += batch_size
        self.avg = self.total_metrics_sum / self.total_batches

    def average(self) -> float:
        if self.total_batches != 0:
            return self.avg
        return 0.0

    def reset(self) -> None:
        self.total_metrics_sum = 0
        self.total_batches = 0
        self.total_samples = 0
        self.avg = 0.0


def calculate_confusion_matrix(
        label_pred: np.ndarray,
        label_true: np.ndarray,
        n_classes: int) -> np.ndarray:
    """
    Calculates a confusion matrix given true and predicted hard labels.
    Args:
        label_pred: 1-D (flattened) predicted hard labels.
        label_true: 1-D (flattened) ground-truth hard labels.
        n_classes: number of segmentation classes. If it equals to 1, then converted to a binary problem.

    Returns:
        A confusion matrix for the given true and predicted hard labels.
    """
    # Mask ignore index values (-1, 255, etc. will be excluded from the confusion matrix:
    mask = (label_true >= 0) & (label_true < n_classes)

    # Calculate the confusion matrix:
    hist = np.bincount(
        n_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_classes ** 2).reshape(n_classes, n_classes)
    return hist


def calculate_batch_metrics(
        predictions: torch.Tensor,
        gts: torch.Tensor,
        n_classes: int,
        metric_dict: Dict) -> Dict:
    """
    Calculate batch metrics for the given batch and ground-truth labels.
    Update current metrics dictionary.
    Args:
        predictions: predicted logits, the direct outputs from the model.
        gts: ground-truth labels.
        n_classes: number of segmentation classes. If it equals to 1, then converted to a binary problem.
        metric_dict: current metrics dictionary.

    Returns:
        Updated metrics dictionary.
    """
    # Extract the batch size:
    batch_size = predictions.shape[0]

    # Get hard labels from the predicted logits:
    if n_classes == 1:
        label_pred = torch.sigmoid(predictions).clone().cpu().detach().numpy()
        ones_array = np.ones(shape=label_pred.shape, dtype=np.uint8)
        bg_pred = ones_array - label_pred
        label_pred = np.concatenate([bg_pred, label_pred], axis=1)
    elif n_classes > 1:
        label_pred = F.softmax(predictions.clone().cpu().detach(), dim=1).numpy()
    else:
        raise ValueError(f'Number of classes is less than 1 == {n_classes}')

    label_pred = np.array(np.argmax(label_pred, axis=1)).astype(np.uint8)
    label_true = gts.clone().cpu().detach().numpy().copy()

    # Make a problem binary if the number of classes == 1:
    n_classes = 2 if n_classes == 1 else n_classes

    # Initialize an empty batch confusion matrix:
    batch_matrix = np.zeros((n_classes, n_classes))

    # For each sample in the batch, add the confusion matrix the to the batch matrix:
    for lp, lt in zip(label_pred, label_true):
        batch_matrix += calculate_confusion_matrix(lp.flatten(), lt.flatten(), n_classes)

    # Calculate metrics from the confusion matrix:
    iu = np.diag(batch_matrix) / (batch_matrix.sum(axis=1) + batch_matrix.sum(axis=0) - np.diag(batch_matrix))
    precision = np.diag(batch_matrix) / batch_matrix.sum(axis=1)
    recall = np.diag(batch_matrix) / batch_matrix.sum(axis=0)
    f_score = 2 * ((precision * recall) / (precision + recall))

    # Update the metrics dict:
    mean_iu = np.nanmean(iu)
    metric_dict['iou'].update(mean_iu, batch_size)

    mean_iu_nobg = np.nanmean(iu[1:])
    metric_dict['iou_nonbg'].update(mean_iu_nobg, batch_size)

    mean_precision = np.nanmean(precision)
    metric_dict['precision'].update(mean_precision, batch_size)

    mean_recall = np.nanmean(recall)
    metric_dict['recall'].update(mean_recall, batch_size)

    mean_f_score = np.nanmean(f_score)
    metric_dict['fscore'].update(mean_f_score, batch_size)

    cls_list = [str(cls) for cls in range(n_classes)]

    for i, cls_lbl in enumerate(cls_list):
        metric_dict['iou_' + cls_lbl].update(iu[i], batch_size)
        metric_dict['precision_' + cls_lbl].update(precision[i], batch_size)
        metric_dict['recall_' + cls_lbl].update(recall[i], batch_size)
        metric_dict['fscore_' + cls_lbl].update(f_score[i], batch_size)

    return metric_dict


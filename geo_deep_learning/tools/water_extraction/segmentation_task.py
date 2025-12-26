"""Water extraction segmentation task that extends SegmentationUnetPlus."""

import logging
from typing import Any

import torch
from torch import Tensor

from geo_deep_learning.tasks_with_models.segmentation_unetplus import (
    SegmentationUnetPlus,
)

logger = logging.getLogger(__name__)


class WaterExtractionSegmentation(SegmentationUnetPlus):
    """
    Segmentation task for water extraction with proper ignore_index handling.

    Extends SegmentationUnetPlus to handle ignore_index (-1) properly in IoU
    calculation by creating a valid mask that excludes ignored pixels.
    
    Includes comprehensive diagnostics for debugging class imbalance and label issues.
    """

    def __init__(self, *args: Any, ignore_index: int = -1, **kwargs: Any) -> None:
        """
        Initialize water extraction segmentation task.

        Args:
            *args: Positional arguments for SegmentationUnetPlus
            ignore_index: Value to ignore in loss and metrics (default: -1)
            **kwargs: Keyword arguments for SegmentationUnetPlus
        """
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index
        self._logged_label_values = False
        
        # # BACKWARD-COMPATIBILITY ALIAS (NO REBUILD)
        # if not hasattr(self, "model"):
        #     raise RuntimeError(
        #         "SegmentationUnetPlus is expected to define `self.model`, but it does not."
        #     )

    def _log_class_stats(self, y: Tensor, stage: str, batch_idx: int) -> None:
        """
        Log per-batch class pixel counts for debugging.

        Args:
            y: Ground truth labels (B, H, W)
            stage: Stage name ("train" or "val")
            batch_idx: Current batch index
        """
        # y: (B, H, W)
        valid = y != self.ignore_index
        
        if not valid.any():
            self.log(f"{stage}_valid_pixels", 0, prog_bar=True, batch_size=y.shape[0])
            logger.warning(
                "[%s] Batch %d has ZERO valid pixels (all ignored)",
                stage.upper(),
                batch_idx,
            )
            return

        unique, counts = torch.unique(y[valid], return_counts=True)
        stats = dict(zip(unique.tolist(), counts.tolist()))

        total_valid = valid.sum().item()
        water_pixels = stats.get(1, 0)
        water_ratio = water_pixels / max(total_valid, 1)

        self.log(f"{stage}_valid_pixels", total_valid, prog_bar=True, batch_size=y.shape[0])
        self.log(f"{stage}_water_pixels", water_pixels, prog_bar=True, batch_size=y.shape[0])
        self.log(f"{stage}_water_ratio", water_ratio, prog_bar=True, batch_size=y.shape[0])
        self.log(f"{stage}_background_pixels", stats.get(0, 0), batch_size=y.shape[0])

        # Log warning if water ratio is very low
        if water_ratio < 0.001:  # Less than 0.1%
            logger.warning(
                "[%s] Batch %d has very low water ratio: %.4f%% (%d / %d pixels)",
                stage.upper(),
                batch_idx,
                water_ratio * 100,
                water_pixels,
                total_valid,
            )
    def _forward_and_loss(self, batch: dict[str, Any]) -> Tensor:
        """
        Single source of truth for forward + loss.
        Used by BOTH training and validation.
        """
        x = batch["image"]
        y = batch["mask"]

        y_hat = self(x)
        loss = self.loss(y_hat, y)

        return loss

    # def on_train_batch_start(
    #     self,
    #     batch: dict[str, Any],
    #     batch_idx: int,
    # ) -> None:
    #     """
    #     Log unique label values once at the start of training.

    #     Args:
    #         batch: Batch data
    #         batch_idx: Batch index
    #     """
    #     # Only log once at the very first batch
    #     if self.global_step == 0 and not self._logged_label_values:
    #         y = batch["mask"]
    #         unique = torch.unique(y)
    #         self.log("label_unique_count", unique.numel(), batch_size=y.shape[0])
            
    #         unique_list = unique.cpu().tolist()
    #         logger.info("=" * 80)
    #         logger.info("LABEL VALUE DIAGNOSTICS (Step 0)")
    #         logger.info("=" * 80)
    #         logger.info("Unique label values in first batch: %s", unique_list)
    #         logger.info("Expected: [-1, 0, 1] (ignore, background, water)")
            
    #         # Interpret findings
    #         if unique_list == [0]:
    #             logger.error("❌ CRITICAL: Only background (0) found - water missing!")
    #         elif unique_list == [-1, 0]:
    #             logger.error("❌ CRITICAL: Water (1) completely masked out!")
    #         elif set(unique_list) == {-1, 0, 1}:
    #             logger.info("✓ All expected label values present")
    #         else:
    #             logger.warning("⚠️  Unexpected label values: %s", unique_list)
            
    #         logger.info("=" * 80)
    #         self._logged_label_values = True

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> Tensor:
        """
        Run training step with class statistics logging.

        Args:
            batch: Batch data
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        
        # Log class statistics
        #self._log_class_stats(y, "train", batch_idx)
        
        unique_vals = torch.unique(y)
        if torch.any(y < 0):
            self.log("has_ignore_pixels", 1, prog_bar=True)
        else:
            self.log("has_ignore_pixels", 0, prog_bar=True)

        if torch.any(y > self.num_classes - 1):
            self.log("has_invalid_labels", 1, prog_bar=True)
            print("INVALID LABEL VALUES:", unique_vals.tolist())
        
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.log(
            "train_loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )

        return loss

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> Tensor:
        """
        Run validation step with class statistics logging.

        Args:
            batch: Batch data
            batch_idx: Batch index

        Returns:
            Predictions
        """
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        
        # Log class statistics
        #self._log_class_stats(y, "val", batch_idx)
        
        unique_vals = torch.unique(y)
        if torch.any(y < 0):
            self.log("has_ignore_pixels", 1, prog_bar=True)
        else:
            self.log("has_ignore_pixels", 0, prog_bar=True)

        if torch.any(y > self.num_classes - 1):
            self.log("has_invalid_labels", 1, prog_bar=True)
            print("INVALID LABEL VALUES:", unique_vals.tolist())
        
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        
        if loss > 2.0:  # threshold to tune
            image_paths = batch.get("image_path")
            mask_paths = batch.get("label_path")
                
            logger.warning(
                f"[VAL-OUTLIER] batch_idx={batch_idx} "
                f"loss={loss.item():.3f}\n"
                f"images={image_paths}\n"
                f"labels={mask_paths}"
            )
            
        self.log(
            "val_loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        
        if self.num_classes == 1:
            y_hat = (y_hat.sigmoid().squeeze(1) > self.threshold).long()
        else:
            y_hat = y_hat.softmax(dim=1).argmax(dim=1)

        return y_hat

    def test_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        """
        Run test step with proper ignore_index handling.

        Overrides parent test_step to mask out ignore_index pixels before
        computing IoU metrics, ensuring that pixels outside the AOI or with
        invalid LiDAR coverage do not affect the evaluation.

        Args:
            batch: Batch dictionary containing image, mask, and metadata
            batch_idx: Index of the current batch
        """
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        y_hat = self(x)
        
        # Compute loss with ignore_index handling (loss function handles this)
        loss = self.loss(y_hat, y)

        # Convert predictions to class indices
        if self.num_classes == 1:
            y_hat_classes = (y_hat.sigmoid().squeeze(1) > self.threshold).long()
        else:
            y_hat_classes = y_hat.softmax(dim=1).argmax(dim=1)

        # Create valid mask: exclude pixels with ignore_index
        valid_mask = y != self.ignore_index

        # Apply mask to both predictions and ground truth
        y_masked = torch.where(valid_mask, y, torch.zeros_like(y))
        y_hat_masked = torch.where(valid_mask, y_hat_classes, torch.zeros_like(y_hat_classes))

        # Only compute metrics if there are valid pixels
        if valid_mask.any():
            metrics = self.iou_classwise_metric(y_hat_masked, y_masked)
            #self.iou_classwise_metric.reset()
            
            # Log the percentage of valid pixels for debugging
            valid_ratio = valid_mask.float().mean().item()
            metrics["valid_pixel_ratio"] = valid_ratio
        else:
            # If no valid pixels, create empty metrics
            metrics = {}
            logger.warning("No valid pixels in batch %d", batch_idx)

        metrics["test_loss"] = loss

        # Visualization logic (unchanged from parent)
        if self._total_samples_visualized < self.max_samples:
            remaining_samples = self.max_samples - self._total_samples_visualized
            samples_to_visualize = min(remaining_samples, len(x))
            samples_visualized = self._log_visualizations(
                trainer=self.trainer,
                batch=batch,
                outputs=y_hat_classes,
                max_samples=samples_to_visualize,
                artifact_prefix="test",
                epoch_suffix=False,
            )
            self._total_samples_visualized += samples_visualized

        self.log_dict(
            metrics,
            batch_size=batch_size,
            prog_bar=False,
            logger=True,
            on_step=False,
            rank_zero_only=True,
        )

    def on_test_epoch_end(self):
        metrics = self.iou_classwise_metric.compute()
        self.log_dict(metrics, prog_bar=True)
        self.iou_classwise_metric.reset()
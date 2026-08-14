"""Fixed validation visualization callback for epoch-to-epoch comparison."""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from matplotlib.figure import Figure
from torch import Tensor

logger = logging.getLogger(__name__)


class FixedValidationVisualizationCallback(Callback):
    """
    Visualization callback that saves outputs for fixed validation samples every epoch.
    
    This enables reliable epoch-to-epoch visual comparison to track training progress
    and identify when model quality degrades despite improving validation loss.
    """

    def __init__(
        self,
        num_samples: int = 5,
        save_dir: str | Path | None = None,
        seed: int = 42,
        save_probability_maps: bool = True,
        save_overlays: bool = True,
        save_error_maps: bool = True,
    ) -> None:
        """
        Initialize callback.

        Args:
            num_samples: Number of fixed samples to visualize each epoch
            save_dir: Directory to save visualizations (default: logs/val_visualizations)
            seed: Random seed for deterministic sample selection
            save_probability_maps: Whether to save probability heatmaps
            save_overlays: Whether to save overlay visualizations
            save_error_maps: Whether to save prediction error maps

        """
        super().__init__()
        self.num_samples = num_samples
        self.save_dir = Path(save_dir) if save_dir else None
        self.seed = seed
        self.save_probability_maps = save_probability_maps
        self.save_overlays = save_overlays
        self.save_error_maps = save_error_maps
        
        # State to track fixed samples
        self.fixed_samples: list[dict[str, Any]] = []
        self.samples_initialized = False

    def on_validation_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Initialize fixed samples on first validation epoch."""
        if not self.samples_initialized and trainer.is_global_zero:
            # Set seed for reproducibility
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            logger.info(
                "FixedValidationVisualizationCallback: Will initialize fixed samples on first batch"
            )

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Capture fixed samples from first validation batches."""
        if not self.samples_initialized:
            # Collect samples until we have enough
            batch_size = batch["image"].shape[0]
            needed = self.num_samples - len(self.fixed_samples)
            
            if needed > 0:
                samples_from_batch = min(needed, batch_size)
                
                for i in range(samples_from_batch):
                    sample = {
                        "image": batch["image"][i].detach().cpu(),
                        "mask": batch["mask"][i].detach().cpu(),
                        "image_name": batch["image_name"][i] if "image_name" in batch else f"sample_{len(self.fixed_samples)}",
                        "mean": batch["mean"][i].detach().cpu() if "mean" in batch else None,
                        "std": batch["std"][i].detach().cpu() if "std" in batch else None,
                    }
                    self.fixed_samples.append(sample)
                
                logger.info(
                    "Captured %d/%d fixed validation samples",
                    len(self.fixed_samples),
                    self.num_samples,
                )
                
                if len(self.fixed_samples) >= self.num_samples:
                    self.samples_initialized = True
                    logger.info("✓ Fixed validation samples initialized")

    @rank_zero_only
    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Generate visualizations for fixed samples at end of each validation epoch."""
        if not self.samples_initialized or len(self.fixed_samples) == 0:
            logger.warning("No fixed samples to visualize")
            return

        # Determine save directory
        if self.save_dir is None:
            if hasattr(trainer.logger, "log_dir") and trainer.logger.log_dir:
                save_dir = Path(trainer.logger.log_dir) / "val_visualizations_fixed"
            else:
                save_dir = Path("logs") / "val_visualizations_fixed"
        else:
            save_dir = self.save_dir

        epoch_dir = save_dir / f"epoch_{trainer.current_epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        # Put model in eval mode
        pl_module.eval()

        with torch.no_grad():
            for sample_idx, sample in enumerate(self.fixed_samples):
                # Move sample to device
                image = sample["image"].unsqueeze(0).to(pl_module.device)
                mask = sample["mask"].to(pl_module.device)
                image_name = sample["image_name"]

                # Get model predictions (handle different model outputs)
                try:
                    if hasattr(pl_module, 'num_classes'):
                        num_classes = pl_module.num_classes
                    else:
                        num_classes = 1

                    # Forward pass
                    logits = pl_module(image)
                    
                    # Handle models that return named tuples (like DOFA)
                    if hasattr(logits, 'out'):
                        logits = logits.out

                    # Convert to probabilities and predictions
                    if num_classes == 1:
                        probs = torch.sigmoid(logits)
                        threshold = getattr(pl_module, 'threshold', 0.5)
                        preds = (probs.squeeze(1) > threshold).long()
                        water_prob = probs[0, 0].cpu().numpy()  # [H, W]
                    else:
                        probs = torch.softmax(logits, dim=1)
                        preds = probs.argmax(dim=1)
                        water_prob = probs[0, 1].cpu().numpy()  # Water class probability

                    pred_mask = preds[0].cpu().numpy()
                    
                except Exception as e:
                    logger.exception("Error generating predictions for sample %d: %s", sample_idx, e)
                    continue

                # Prepare ground truth mask
                if mask.dim() == 3:
                    gt_mask = mask[0].cpu().numpy()
                else:
                    gt_mask = mask.cpu().numpy()
                
                # Denormalize image for visualization if mean/std available
                vis_image = image[0].cpu()
                if sample["mean"] is not None and sample["std"] is not None:
                    mean = sample["mean"]
                    std = sample["std"]
                    # Denormalize
                    for c in range(vis_image.shape[0]):
                        vis_image[c] = vis_image[c] * std[c] + mean[c]
                
                # Clip and convert to displayable format
                vis_image = torch.clamp(vis_image, 0, 1)
                
                # Convert to RGB if single channel, pad 2 channels, or select first 3 channels
                if vis_image.shape[0] == 1:
                    vis_image = vis_image.repeat(3, 1, 1)
                elif vis_image.shape[0] == 2:
                    vis_image = torch.cat(
                        [vis_image, torch.zeros_like(vis_image[:1])],
                        dim=0,
                    )
                elif vis_image.shape[0] > 3:
                    vis_image = vis_image[:3]
                
                vis_image = vis_image.permute(1, 2, 0).numpy()

                # Save outputs
                sample_name = Path(image_name).stem if isinstance(image_name, str) else f"sample_{sample_idx}"
                
                # 1. Save as numpy arrays for later analysis
                np.savez_compressed(
                    epoch_dir / f"{sample_name}_data.npz",
                    image=vis_image,
                    ground_truth=gt_mask,
                    prediction=pred_mask,
                    water_probability=water_prob,
                    epoch=trainer.current_epoch,
                )

                # 2. Create comprehensive visualization figure
                self._create_visualization_figure(
                    vis_image=vis_image,
                    gt_mask=gt_mask,
                    pred_mask=pred_mask,
                    water_prob=water_prob,
                    sample_name=sample_name,
                    epoch=trainer.current_epoch,
                    save_path=epoch_dir / f"{sample_name}_visualization.png",
                )

                # 3. Save probability map as separate heatmap
                if self.save_probability_maps:
                    self._save_probability_heatmap(
                        water_prob=water_prob,
                        sample_name=sample_name,
                        epoch=trainer.current_epoch,
                        save_path=epoch_dir / f"{sample_name}_probability.png",
                    )

        logger.info("✓ Saved fixed validation visualizations to: %s", epoch_dir)

    def _create_visualization_figure(
        self,
        vis_image: np.ndarray,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        water_prob: np.ndarray,
        sample_name: str,
        epoch: int,
        save_path: Path,
    ) -> None:
        """Create comprehensive visualization figure."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"{sample_name} - Epoch {epoch}", fontsize=16)

        # Row 1: Input, Ground Truth, Prediction
        axes[0, 0].imshow(vis_image)
        axes[0, 0].set_title("Input Image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(gt_mask, cmap="Blues", vmin=0, vmax=1)
        axes[0, 1].set_title("Ground Truth (Water=1)")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(pred_mask, cmap="Blues", vmin=0, vmax=1)
        axes[0, 2].set_title("Prediction (Water=1)")
        axes[0, 2].axis("off")

        # Row 2: Probability Map, Overlay, Error Map
        im = axes[1, 0].imshow(water_prob, cmap="hot", vmin=0, vmax=1)
        axes[1, 0].set_title("Water Probability")
        axes[1, 0].axis("off")
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

        # Overlay: prediction on image
        overlay = vis_image.copy()
        water_mask_colored = np.zeros_like(overlay)
        water_mask_colored[pred_mask == 1] = [0, 0.5, 1.0]  # Blue for water
        overlay = 0.7 * overlay + 0.3 * water_mask_colored
        axes[1, 1].imshow(np.clip(overlay, 0, 1))
        axes[1, 1].set_title("Prediction Overlay")
        axes[1, 1].axis("off")

        # Error map: TP=green, FP=red, FN=yellow, TN=black
        error_map = np.zeros((*gt_mask.shape, 3))
        tp = (gt_mask == 1) & (pred_mask == 1)
        fp = (gt_mask == 0) & (pred_mask == 1)
        fn = (gt_mask == 1) & (pred_mask == 0)
        error_map[tp] = [0, 1, 0]  # Green
        error_map[fp] = [1, 0, 0]  # Red
        error_map[fn] = [1, 1, 0]  # Yellow
        axes[1, 2].imshow(error_map)
        axes[1, 2].set_title("Error Map (TP=Green, FP=Red, FN=Yellow)")
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _save_probability_heatmap(
        self,
        water_prob: np.ndarray,
        sample_name: str,
        epoch: int,
        save_path: Path,
    ) -> None:
        """Save probability map as separate high-quality heatmap."""
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(water_prob, cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"{sample_name} - Water Probability - Epoch {epoch}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

"""Pretraining MAE model."""

import logging
import math
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Sequence

import kornia as krn
import torch
from kornia.augmentation import AugmentationSequential
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from geo_deep_learning.tools.schedulers.lr_scheduler import (
    MaeLRScheduler,
    MaeLRSchedulerFactory,
)
from geo_deep_learning.utils.models import load_weights_from_checkpoint
from models.pretrain.mae import MAEPretrainModel
from geo_deep_learning.tools.visualization import visualize_pretrain
from matplotlib import pyplot as plt

warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed",
)

logger = logging.getLogger(__name__)


class PretrainMAE(LightningModule):
    """Segmentation SegFormer model."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        image_size: Sequence[int] | int,
        in_channels: int,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_pix_loss: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        scheduler_config: dict[str, Any] | None = None,
        weights_from_checkpoint_path: str | None = None,
        max_samples: int = 10,
        **kwargs: object,  # noqa: ARG002
    ) -> None:

    
        """Initialize the model."""
        super().__init__()
        self.save_hyperparameters()
        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_pix_loss = norm_pix_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config or {"interval": "epoch"}
        self.weights_from_checkpoint_path = weights_from_checkpoint_path
    

        self.test_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.test_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.test_fid = FrechetInceptionDistance(feature=2048, normalize=True)
        # Visualization tracking
        self.max_samples = max_samples
        self._total_samples_visualized = 0

    def configure_model(self) -> None:
        """Configure model."""
        self.model = MAEPretrainModel(
            img_size=self.image_size[0],
            patch_size=self.patch_size,
            mask_ratio=self.mask_ratio,
            in_chans=self.in_channels,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            decoder_embed_dim=self.decoder_embed_dim,
            decoder_depth=self.decoder_depth,
            decoder_num_heads=self.decoder_num_heads,
            mlp_ratio=self.mlp_ratio,
            norm_layer=torch.nn.LayerNorm,
            norm_pix_loss=self.norm_pix_loss,
        )
        if self.weights_from_checkpoint_path:
            map_location = self.device
            logger.info(
                "Loading weights from checkpoint: %s",
                self.weights_from_checkpoint_path,
            )
            load_weights_from_checkpoint(
                self.model,
                self.weights_from_checkpoint_path,
                map_location=map_location,
            )

    def configure_optimizers(self) -> list[list[dict[str, Any]]]:
        """Configure optimizers."""
        optimizer = self.optimizer(self.parameters())
        if (
            self.hparams["scheduler"]["class_path"]
            == "torch.optim.lr_scheduler.OneCycleLR"
        ):
            max_lr = (
                self.hparams.get("scheduler", {}).get("init_args", {}).get("max_lr")
            )
            stepping_batches = self.trainer.estimated_stepping_batches
            if stepping_batches > -1:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    total_steps=stepping_batches,
                )
            elif (
                stepping_batches == -1
                and getattr(self.trainer.datamodule, "epoch_size", None) is not None
            ):
                batch_size = self.trainer.datamodule.batch_size
                epoch_size = self.trainer.datamodule.epoch_size
                accumulate_grad_batches = self.trainer.accumulate_grad_batches
                max_epochs = self.trainer.max_epochs
                steps_per_epoch = math.ceil(
                    epoch_size / (batch_size * accumulate_grad_batches),
                )
                buffer_steps = int(steps_per_epoch * accumulate_grad_batches)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    steps_per_epoch=steps_per_epoch + buffer_steps,
                    epochs=max_epochs,
                )
            else:
                stepping_batches = (
                    self.hparams.get("scheduler", {})
                    .get("init_args", {})
                    .get("total_steps")
                )
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    total_steps=stepping_batches,
                )
            scheduler_config = dict(self.scheduler_config)
            scheduler_config.setdefault("interval", "epoch")
            scheduler_config["scheduler"] = scheduler
            return [optimizer], [scheduler_config]

        if (
            self.hparams["scheduler"]["class_path"]
            == "geo_deep_learning.tools.schedulers.lr_scheduler.MaeLRSchedulerFactory"
        ):
            scheduler_cfg = self.hparams.get("scheduler", {})
            init_args = scheduler_cfg.get("init_args", {})
            iteration_per_epoch = self._infer_iteration_per_epoch(
                init_args.get("iteration_per_epoch"),
            )
            max_iterations = self._infer_max_iterations(
                init_args.get("max_iterations"),
                iteration_per_epoch,
            )
            config_lr = init_args.get("lr")
            optimizer_lr = (
                self.hparams.get("optimizer", {}).get("init_args", {}).get("lr")
            )
            base_lr = config_lr if config_lr is not None else optimizer_lr
            if base_lr is None:
                msg = (
                    "MaeLRScheduler requires `lr` either in scheduler init args or optimizer init args."
                )
                raise ValueError(msg)
            scheduler_factory = MaeLRSchedulerFactory(
                accum_iter=init_args["accum_iter"],
                min_lr=init_args["min_lr"],
                warmup_epochs=init_args["warmup_epochs"],
                lr=config_lr,
                iteration_per_epoch=iteration_per_epoch,
                max_iterations=max_iterations,
            )
            scheduler = scheduler_factory(
                optimizer,
                lr=base_lr,
                iteration_per_epoch=iteration_per_epoch,
                max_iterations=max_iterations,
            )
            scheduler_config = dict(self.scheduler_config)
            scheduler_config["interval"] = "step"
            scheduler_config["scheduler"] = scheduler
            return [optimizer], [scheduler_config]

        scheduler = self.scheduler(optimizer)
        return [optimizer], [{"scheduler": scheduler, **self.scheduler_config}]

    def _infer_iteration_per_epoch(self, provided: int | None) -> int:
        if provided is not None:
            if provided <= 0:
                msg = "`iteration_per_epoch` must be a positive integer."
                raise ValueError(msg)
            return provided

        trainer = self.trainer
        if trainer is None:
            msg = "Trainer reference is required to infer iteration_per_epoch."
            raise RuntimeError(msg)

        iteration_per_epoch = None
        num_training_batches = getattr(trainer, "num_training_batches", None)
        if isinstance(num_training_batches, (list, tuple)):
            iteration_per_epoch = num_training_batches[0]
        elif isinstance(num_training_batches, int):
            iteration_per_epoch = num_training_batches

        if iteration_per_epoch is None:
            datamodule = getattr(trainer, "datamodule", None)
            if datamodule is not None:
                train_dataloader = datamodule.train_dataloader()
                try:
                    iteration_per_epoch = len(train_dataloader)
                except TypeError as exc:
                    msg = (
                        "Unable to infer iteration_per_epoch automatically. "
                        "Please set `iteration_per_epoch` explicitly in the scheduler init args."
                    )
                    raise ValueError(msg) from exc

        if iteration_per_epoch is None or iteration_per_epoch <= 0:
            msg = (
                "Unable to determine a valid `iteration_per_epoch`. "
                "Please set it explicitly in the scheduler init args."
            )
            raise ValueError(msg)

        return iteration_per_epoch

    def _infer_max_iterations(
        self,
        provided: int | None,
        iteration_per_epoch: int,
    ) -> int:
        if provided is not None:
            if provided <= 0:
                msg = "`max_iterations` must be a positive integer."
                raise ValueError(msg)
            return provided

        trainer = self.trainer
        if trainer is None:
            msg = "Trainer reference is required to infer max_iterations."
            raise RuntimeError(msg)

        max_steps = getattr(trainer, "max_steps", None)
        if max_steps is not None and max_steps > 0:
            return max_steps

        max_epochs = getattr(trainer, "max_epochs", None)
        if max_epochs is None or max_epochs <= 0:
            msg = (
                "Unable to infer `max_iterations`. "
                "Provide `max_iterations` or ensure `max_epochs` is set."
            )
            raise ValueError(msg)

        return max_epochs * iteration_per_epoch

    def lr_scheduler_step(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler | MaeLRScheduler,
        metric: Tensor | None,
    ) -> None:
        if isinstance(scheduler, MaeLRScheduler):
            scheduler.step()
            return
        super().lr_scheduler_step(scheduler, metric)

    def forward(self, image: Tensor) -> Tensor:
        """Forward pass."""
        return self.model(image)

    
    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run training step."""
        x = batch["image"]
 
        batch_size = x.shape[0]
        loss, _, mask = self(x)
      
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
        self.log(
            "train_mask_ratio",
            mask.float().mean(),
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        
        # Log learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log(
            "learning_rate",
            current_lr,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )

        return loss


    def test_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Run test step."""
        x = batch["image"]
        batch_size = x.shape[0]
        loss, pred, mask = self(x)
        
        # Unpatchify the prediction to get reconstructed images
        reconstructed = self.model.unpatchify(pred)
        
        # Normalize images to [0, 1] range for metrics
        # Use the combined min/max of both images and reconstructions for consistent normalization
        combined_min = min(x.min().item(), reconstructed.min().item())
        combined_max = max(x.max().item(), reconstructed.max().item())
        combined_range = combined_max - combined_min
        
        if combined_range > 0:
            images_normalized = (x - combined_min) / combined_range
            reconstructed_normalized = (reconstructed - combined_min) / combined_range
        else:
            images_normalized = x - combined_min
            reconstructed_normalized = reconstructed - combined_min
        
        # Clamp to [0, 1] to ensure valid range
        images_normalized = torch.clamp(images_normalized, 0, 1)
        reconstructed_normalized = torch.clamp(reconstructed_normalized, 0, 1)
        
        # Convert to uint8 format for FID (required by torchmetrics)
        images_uint8 = (images_normalized * 255).byte()
        reconstructed_uint8 = (reconstructed_normalized * 255).byte()
        
        # Update metrics
        self.test_ssim.update(reconstructed_normalized, images_normalized)
        self.test_psnr.update(reconstructed_normalized, images_normalized)
        self.test_fid.update(images_uint8, real=True)
        self.test_fid.update(reconstructed_uint8, real=False)
        
        self.log(
            "test_loss",
            loss,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "test_mask_ratio",
            mask.float().mean(),
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        # Log visualizations
        if self._total_samples_visualized < self.max_samples:
            remaining_samples = self.max_samples - self._total_samples_visualized
            samples_to_visualize = min(remaining_samples, len(x))
            samples_visualized = self._log_visualizations(
                trainer=self.trainer,
                batch=batch,
                images=x,
                mask=mask,
                reconstructed=reconstructed,
                max_samples=samples_to_visualize,
                artifact_prefix="test",
                epoch_suffix=False,
            )
            self._total_samples_visualized += samples_visualized
        
        return loss

    def on_test_epoch_end(self) -> None:
        """Compute and log test metrics at the end of the test epoch."""
        ssim_score = self.test_ssim.compute()
        psnr_score = self.test_psnr.compute()
        fid_score = self.test_fid.compute()
        
        self.log(
            "test_ssim",
            ssim_score,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "test_psnr",
            psnr_score,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "test_fid",
            fid_score,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        
        # Reset metrics for next epoch
        self.test_ssim.reset()
        self.test_psnr.reset()
        self.test_fid.reset()
        
        # Reset visualization counter
        self._total_samples_visualized = 0
        
    def _log_visualizations(  # noqa: PLR0913
        self,
        trainer: Trainer,
        batch: Any,
        images: Tensor,
        mask: Tensor,
        reconstructed: Tensor,
        max_samples: int,
        artifact_prefix: str = "test",
        *,
        epoch_suffix: bool = False,
    ) -> int:
        """
        Log visualizations for pretraining.

        Args:
            trainer: Lightning trainer
            batch: Batch data containing image_name (optional)
            images: Input images tensor of shape (B, C, H, W)
            mask: Mask tensor of shape (B, num_patches)
            reconstructed: Reconstructed images tensor of shape (B, C, H, W)
            max_samples: Maximum number of samples to visualize
            artifact_prefix: Prefix for artifact path ("test" or "val")
            epoch_suffix: Whether to add epoch info to artifact filename

        Returns:
            Number of samples actually visualized
        """
        if trainer is None or trainer.logger is None:
            return 0

        try:
            logger.info("Logging pretrain visualizations")
            num_samples = min(max_samples, len(images))
            
            # Extract image_name from batch if available
            batch_image_name = None
            if isinstance(batch, dict) and "image_name" in batch:
                batch_image_name = batch["image_name"]
            
            for i in range(num_samples):
                image = images[i]
                mask_i = mask[i]
                reconstructed_i = reconstructed[i]
                
                # Get image name if available, otherwise use default
                if batch_image_name and i < len(batch_image_name):
                    image_name = batch_image_name[i]
                    base_path = f"{artifact_prefix}/{Path(image_name).stem}"
                else:
                    image_name = f"sample_{i}"
                    base_path = f"{artifact_prefix}/{image_name}"
                
                fig = visualize_pretrain(
                    image=image,
                    mask=mask_i,
                    reconstructed=reconstructed_i,
                    model=self.model,
                    sample_name=image_name,
                    patch_size=self.patch_size,
                    image_size=self.image_size[0],
                )
                
                if epoch_suffix and trainer is not None:
                    artifact_file = (
                        f"{base_path}/idx_{i}_epoch_{trainer.current_epoch}.png"
                    )
                else:
                    artifact_file = f"{base_path}/idx_{i}.png"
                
                trainer.logger.experiment.log_figure(
                    figure=fig,
                    artifact_file=artifact_file,
                    run_id=trainer.logger.run_id,
                )
                plt.close(fig)
                
            return num_samples
        except Exception:
            logger.exception("Error in pretrain visualization")
            return 0

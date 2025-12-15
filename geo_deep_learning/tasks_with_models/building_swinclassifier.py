"""Building Swin Classifier model."""

import logging
import math
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import kornia as krn
import torch
from kornia.augmentation import AugmentationSequential
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from geo_deep_learning.tools.schedulers.lr_scheduler import (
    MaeLRScheduler,
    MaeLRSchedulerFactory,
)
from geo_deep_learning.utils.models import swinclassifier_load_weights_from_checkpoint
from geo_deep_learning.utils.tensors import denormalization
from geo_deep_learning.models.classification.swin_building_classifier import (
    SwinBuildingClassifier,
)
from geo_deep_learning.tools.visualization import visualize_building_classification

warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed",
)

logger = logging.getLogger(__name__)


class BuildingSwinClassifier(LightningModule):
    """Building Classification using Swin Transformer."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        image_size: tuple[int, int],
        in_channels: int,
        patch_size: int,
        window_size: int,
        mlp_ratio: float,
        depths: list[int],
        embed_dim: int,
        num_heads: list[int],
        qkv_bias: bool,
        qk_scale: float | None,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        ape: bool,
        patch_norm: bool,
        use_checkpoint: bool,
        num_classes: int,
        bbox_mlp_hidden_dim: int = 128,
        bbox_mlp_output_dim: int = 256,
        classifier_hidden_dim: int = 512,
        classifier_dropout: float = 0.5,
        freeze_encoder: bool = False,
        loss: Callable,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        scheduler_config: dict[str, Any] | None = None,
        class_labels: list[str] | None = None,
        weights_from_checkpoint_path: str | None = None,
        max_samples: int = 6,
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        """Initialize the building classifier model."""
        super().__init__()
        self.save_hyperparameters()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.attn_drop_rate = attn_drop_rate
        self.ape = ape
        self.patch_norm = patch_norm
        self.use_checkpoint = use_checkpoint
        self.num_classes = num_classes
        self.bbox_mlp_hidden_dim = bbox_mlp_hidden_dim
        self.bbox_mlp_output_dim = bbox_mlp_output_dim
        self.classifier_hidden_dim = classifier_hidden_dim
        self.classifier_dropout = classifier_dropout
        self.freeze_encoder = freeze_encoder
        self.max_samples = max_samples
        
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config or {"interval": "epoch"}
        self.weights_from_checkpoint_path = weights_from_checkpoint_path

        # Classification metrics - grouped to reduce model summary clutter
        metrics = MetricCollection({
            'accuracy': MulticlassAccuracy(num_classes=num_classes),
            'precision': MulticlassPrecision(num_classes=num_classes, average='macro'),
            'recall': MulticlassRecall(num_classes=num_classes, average='macro'),
            'f1': MulticlassF1Score(num_classes=num_classes, average='macro'),
        })
        
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        self.labels = (
            [str(i) for i in range(num_classes)]
            if class_labels is None
            else class_labels
        )
        self._total_samples_visualized = 0

    def _apply_aug(self) -> AugmentationSequential:
        """Augmentation pipeline for classification."""
        random_resized_crop = krn.augmentation.RandomResizedCrop(
            size=self.image_size,
            scale=(0.8, 1.0),
            p=0.5,
            align_corners=False,
            keepdim=True,
        )
        return AugmentationSequential(
            krn.augmentation.RandomHorizontalFlip(p=0.5, keepdim=True),
            krn.augmentation.RandomVerticalFlip(p=0.5, keepdim=True),
            krn.augmentation.RandomRotation(degrees=15.0, p=0.5),
            random_resized_crop,
            krn.augmentation.RandomGaussianNoise(mean=0.0, std=0.1, p=0.2),
            data_keys=None,
            random_apply=1,
        )

    def configure_model(self) -> None:
        """Configure model."""
        # Create a config object for SwinBuildingClassifier
        from omegaconf import OmegaConf
        
        cfg = OmegaConf.create({
            'model': {
                'input_size': self.image_size,
                'in_chans': self.in_channels,
                'patch_size': self.patch_size,
                'window_size': self.window_size,
                'mlp_ratio': self.mlp_ratio,
                'qkv_bias': self.qkv_bias,
                'qk_scale': self.qk_scale,
                'drop_rate': self.drop_rate,
                'attn_drop_rate': self.attn_drop_rate,
                'ape': self.ape,
                'patch_norm': self.patch_norm,
                'use_checkpoint': self.use_checkpoint,
                'num_classes': self.num_classes,
                'bbox_mlp_hidden_dim': self.bbox_mlp_hidden_dim,
                'bbox_mlp_output_dim': self.bbox_mlp_output_dim,
                'classifier_hidden_dim': self.classifier_hidden_dim,
                'classifier_dropout': self.classifier_dropout,
            },
            'variant': {
                'embed_dim': self.embed_dim,
                'depths': self.depths,
                'num_heads': self.num_heads,
                'drop_path_rate': self.drop_path_rate,
            }
        })
        
        self.model = SwinBuildingClassifier(cfg)
        
        if self.weights_from_checkpoint_path:
            map_location = self.device
            load_parts = self.hparams.get("load_parts", ["rgb_encoder", "mask_encoder"])
            logger.info(
                "Loading weights from checkpoint: %s",
                self.weights_from_checkpoint_path,
            )
            swinclassifier_load_weights_from_checkpoint(
                self.model,
                self.weights_from_checkpoint_path,
                load_parts=load_parts,
                map_location=map_location,
                freeze_encoder=self.freeze_encoder
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

    def forward(
        self,
        rgb_image: Tensor,
        binary_mask: Tensor,
        bbox_list: list,
    ) -> Tensor:
        """Forward pass for building classification.
        
        Args:
            rgb_image: RGB image tensor of shape (B, 3, H, W)
            binary_mask: Binary mask tensor of shape (B, 1, H, W)
            bbox_list: List of bounding boxes for each sample
            
        Returns:
            Classification logits of shape (B, num_classes)
        """
        return self.model(rgb_image, binary_mask, bbox_list)

    def on_before_batch_transfer(
        self,
        batch: dict[str, Any],
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict[str, Any]:
        """
        On before batch transfer - apply augmentation and normalization.
        
        Data pipeline flow:
        1. Dataset loads images and normalizes to [0, 1]
        2. Here: Apply augmentation (if training) - transforms images and masks
        3. Here: Apply standardization using dataset's mean/std
        
        Note: Bounding boxes are NOT transformed by augmentation since they
        represent instance locations in the full image coordinate system.
        The model uses these bbox coordinates to encode spatial information.
        """
        if self.trainer.training:
            aug = self._apply_aug()
            # Apply augmentation to RGB image and mask together
            # Note: bbox is not augmented as it's used for spatial encoding
            transformed = aug({
                "image": batch["rgb_image"],
                "mask": batch["binary_mask"]
            })
            batch["rgb_image"] = transformed["image"]
            batch["binary_mask"] = transformed["mask"]
        
        return batch

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run training step."""
        rgb_image = batch["rgb_image"]
        binary_mask = batch["binary_mask"]
        bbox_list = batch["bbox"]
        labels = batch["label"].long()
        batch_size = rgb_image.shape[0]
        
        # Forward pass
        logits = self(rgb_image, binary_mask, bbox_list)
        loss = self.loss(logits, labels)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        self.train_metrics(preds, labels)

        # Log metrics
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

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run validation step."""
        rgb_image = batch["rgb_image"]
        binary_mask = batch["binary_mask"]
        bbox_list = batch["bbox"]
        labels = batch["label"].long()
        batch_size = rgb_image.shape[0]
        
        # Forward pass
        logits = self(rgb_image, binary_mask, bbox_list)
        loss = self.loss(logits, labels)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        self.val_metrics(preds, labels)

        # Log metrics
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
        self.log_dict(
            self.val_metrics,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )

        # Log visualizations (only on first validation batch of each epoch)
        if batch_idx == 0 and self.trainer.current_epoch % 5 == 0:  # Log every 5 epochs
            samples_to_visualize = min(3, len(rgb_image))  # Log up to 3 samples per validation
            self._log_visualizations(
                trainer=self.trainer,
                batch=batch,
                predictions=preds,
                max_samples=samples_to_visualize,
                artifact_prefix="val",
                epoch_suffix=True,
            )

        return preds

    def test_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Run test step."""
        rgb_image = batch["rgb_image"]
        binary_mask = batch["binary_mask"]
        bbox_list = batch["bbox"]
        labels = batch["label"].long()
        batch_size = rgb_image.shape[0]
        
        # Forward pass
        logits = self(rgb_image, binary_mask, bbox_list)
        loss = self.loss(logits, labels)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        self.test_metrics(preds, labels)

        # Log metrics
        self.log(
            "test_loss",
            loss,
            batch_size=batch_size,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            rank_zero_only=True,
        )
        self.log_dict(
            self.test_metrics,
            batch_size=batch_size,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            rank_zero_only=True,
        )

        # Log visualizations
        if self._total_samples_visualized < self.max_samples:
            remaining_samples = self.max_samples - self._total_samples_visualized
            samples_to_visualize = min(remaining_samples, len(rgb_image))
            samples_visualized = self._log_visualizations(
                trainer=self.trainer,
                batch=batch,
                predictions=preds,
                max_samples=samples_to_visualize,
                artifact_prefix="test",
                epoch_suffix=False,
            )
            self._total_samples_visualized += samples_visualized

    def _log_visualizations(  # noqa: PLR0913
        self,
        trainer: Trainer,
        batch: dict[str, Any],
        predictions: Tensor,
        max_samples: int,
        artifact_prefix: str = "val",
        *,
        epoch_suffix: bool = True,
    ) -> int:
        """
        Log visualizations for building classification.

        Args:
            trainer: Lightning trainer
            batch: Batch data containing rgb_image, binary_mask, bbox, label, image_name, mean, std
            predictions: Model predictions (class indices)
            max_samples: Maximum number of samples to visualize
            artifact_prefix: Prefix for artifact path ("test" or "val")
            epoch_suffix: Whether to add epoch info to artifact filename

        Returns:
            Number of samples actually visualized
        """
        if batch is None or predictions is None:
            return 0

        try:
            logger.info("Logging building classification visualizations")
            image_batch = batch["rgb_image"]
            mask_batch = batch["binary_mask"]
            bbox_list = batch["bbox"]
            labels_batch = batch["label"]
            batch_image_name = batch.get("image_name", [])
            mean_batch = batch.get("mean")
            std_batch = batch.get("std")
            num_samples = min(max_samples, len(image_batch))
            
            for i in range(num_samples):
                image = image_batch[i]
                mask = mask_batch[i]
                bbox = bbox_list[i]
                label = labels_batch[i].item()
                pred = predictions[i].item()
                
                # Get image name
                if batch_image_name and i < len(batch_image_name):
                    image_name = batch_image_name[i]
                else:
                    image_name = f"sample_{i}"
                
                # Denormalize image if mean/std are available
                if mean_batch is not None and std_batch is not None:
                    mean = mean_batch[i] if len(mean_batch.shape) > 1 else mean_batch
                    std = std_batch[i] if len(std_batch.shape) > 1 else std_batch
                    image = denormalization(image, mean=mean, std=std)
                    # Convert back to [0, 1] range for visualization
                    image = image.float() / 255.0
                else:
                    # Image should already be in [0, 1] range
                    image = image.clamp(0, 1)

                fig = visualize_building_classification(
                    image=image,
                    mask=mask,
                    bbox=bbox,
                    label=label,
                    prediction=pred,
                    sample_name=image_name,
                    class_labels=self.labels,
                )
                
                base_path = f"{artifact_prefix}/{Path(image_name).stem}"
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
        except Exception:
            logger.exception("Error in building classification visualization")
            return 0
        else:
            return num_samples

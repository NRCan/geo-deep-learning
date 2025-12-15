"""Segmentation SwinUnet model."""

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
from torchmetrics import JaccardIndex
from torchmetrics.wrappers import ClasswiseWrapper

from geo_deep_learning.tools.schedulers.lr_scheduler import (
    MaeLRScheduler,
    MaeLRSchedulerFactory,
)
from geo_deep_learning.utils.models import swinunet_load_weights_from_checkpoint
from geo_deep_learning.utils.tensors import denormalization
from models.segmentation.swin_unet import SwinUnetSegmentationModel
from tools.visualization import visualize_prediction

warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed",
)

logger = logging.getLogger(__name__)


class SegmentationSwinUnet(LightningModule):
    """Segmentation SegFormer model."""

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
        final_upsample: str,
        ignore_index: int = 255,
        weights: str | None,
        max_samples: int,
        num_classes: int,
        target_class: int,
        freeze_encoder: bool,
        loss: Callable,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        scheduler_config: dict[str, Any] | None = None,
        class_labels: list[str] | None = None,
        class_colors: list[str] | None = None,
        weights_from_checkpoint_path: str | None = None,
        **kwargs: object,  # noqa: ARG002
    ) -> None:

    
        """Initialize the model."""
        super().__init__()
        self.save_hyperparameters()
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
        self.final_upsample = final_upsample
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.image_size = image_size
        self.max_samples = max_samples
        self.target_class = target_class
        self.freeze_encoder = freeze_encoder
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config or {"interval": "epoch"}
        self.weights = weights
        self.weights_from_checkpoint_path = weights_from_checkpoint_path


        self.class_colors = class_colors
        self.threshold = 0.5

        num_classes = num_classes + 1 if num_classes == 1 else num_classes
        # Use -1 as ignore_index (standard value that works better with torchmetrics)
        self.iou_metric = JaccardIndex(
            num_classes=num_classes,
            task="multiclass",
            average="none",
            ignore_index=-1,
        )
        self.labels = (
            [str(i) for i in range(num_classes)]
            if class_labels is None
            else class_labels
        )
        self.iou_classwise_metric = ClasswiseWrapper(
            self.iou_metric,
            labels=self.labels,
        )
        self._total_samples_visualized = 0

    def _apply_aug(self) -> AugmentationSequential:
        """Augmentation pipeline."""
        random_resized_crop = krn.augmentation.RandomResizedCrop(
            size=self.image_size,
            scale=(0.5, 2.0),
            p=0.5,
            align_corners=False,
            keepdim=True,
        )
        return AugmentationSequential(
            krn.augmentation.RandomHorizontalFlip(p=0.2, keepdim=True),
            krn.augmentation.RandomVerticalFlip(p=0.2, keepdim=True),
            krn.augmentation.RandomRotation(degrees=20.0, p=0.5),
            random_resized_crop,
            krn.augmentation.RandomGaussianNoise(mean=0.0, std=1., p=0.2),
            data_keys=None,
            random_apply=1,
        )

    def configure_model(self) -> None:
        """Configure model."""
        self.model = SwinUnetSegmentationModel(
            image_size=self.image_size,
            in_channels=self.in_channels,
            patch_size=self.patch_size,
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            depths=self.depths,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            drop_rate=self.drop_rate,
            drop_path_rate=self.drop_path_rate,
            attn_drop_rate=self.attn_drop_rate,
            ape=self.ape,
            patch_norm=self.patch_norm,
            use_checkpoint=self.use_checkpoint,
            final_upsample=self.final_upsample,
            num_classes=self.num_classes
        )
        if self.weights_from_checkpoint_path:
            map_location = self.device
            load_parts = self.hparams.get("load_parts")
            logger.info(
                "Loading weights from checkpoint: %s",
                self.weights_from_checkpoint_path,
            )
            swinunet_load_weights_from_checkpoint(
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

    def forward(self, image: Tensor) -> Tensor:
        """Forward pass."""
        return self.model(image)

    def on_before_batch_transfer(
        self,
        batch: dict[str, Any],
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict[str, Any]:
        """On before batch transfer."""
        if self.trainer.training:
            aug = self._apply_aug()
            transformed = aug({"image": batch["image"], "mask": batch["mask"]})
            batch.update(transformed)
        return batch

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run training step."""
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        y = y.squeeze(1).long()
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
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        y = y.squeeze(1).long()
        y_hat = self(x)
        loss = self.loss(y_hat, y)
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
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Run test step."""
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        y = y.squeeze(1).long()
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        if self.num_classes == 1:
            y_hat = (y_hat.sigmoid().squeeze(1) > self.threshold).long()
        else:
            y_hat = y_hat.softmax(dim=1).argmax(dim=1)
            # Clamp predictions to valid class indices
            adjusted_num_classes = self.num_classes + 1 if self.num_classes == 1 else self.num_classes
            y_hat = torch.clamp(y_hat, min=0, max=adjusted_num_classes - 1)

        # Map 255 to -1 for ignore_index handling (both -1 and 255 will be ignored)
        y_metric = y.clone()
        y_metric[y_metric == 255] = -1

        metrics = self.iou_classwise_metric(y_hat, y_metric)
        self.iou_classwise_metric.reset()
        metrics["test_loss"] = loss

        if self._total_samples_visualized < self.max_samples:
            remaining_samples = self.max_samples - self._total_samples_visualized
            samples_to_visualize = min(remaining_samples, len(x))
            samples_visualized = self._log_visualizations(
                trainer=self.trainer,
                batch=batch,
                outputs=y_hat,
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

    def _log_visualizations(  # noqa: PLR0913
        self,
        trainer: Trainer,
        batch: dict[str, Any],
        outputs: Tensor,
        max_samples: int,
        artifact_prefix: str = "val",
        *,
        epoch_suffix: bool = True,
    ) -> None:
        """
        SwinUnet-specific log visualizations.

        Args:
            trainer: Lightning trainer
            batch: Batch data containing image, mask, image_name, mean, std
            outputs: Model predictions
            max_samples: Maximum number of samples to visualize
            artifact_prefix: Prefix for artifact path ("test" or "val")
            epoch_suffix: Whether to add epoch info to artifact filename

        Returns:
            Number of samples actually visualized

        """
        if batch is None or outputs is None:
            return 0

        try:
            logger.info("Logging visualizations")
            image_batch = batch["image"]
            mask_batch = batch["mask"].squeeze(1).long()
            batch_image_name = batch["image_name"]
            mean_batch = batch["mean"]
            std_batch = batch["std"]
            num_samples = min(max_samples, len(image_batch))
            for i in range(num_samples):
                image = image_batch[i]
                image_name = batch_image_name[i]
                mean = mean_batch[i]
                std = std_batch[i]
                image = denormalization(image, mean=mean, std=std)

                fig = visualize_prediction(
                    image=image,
                    mask=mask_batch[i],
                    prediction=outputs[i],
                    sample_name=image_name,
                    num_classes=self.num_classes,
                    class_colors=self.class_colors,
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
            logger.exception("Error in SwinUnet visualization")
        else:
            return num_samples

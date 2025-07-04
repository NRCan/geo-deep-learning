"""Segmentation UNetPlus model."""

import logging
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import segmentation_models_pytorch as smp
import torch
from lightning.pytorch import LightningDataModule, LightningModule
from tools.script_model import ScriptModel
from tools.utils import denormalization
from tools.visualization import visualize_prediction
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.segmentation import MeanIoU
from torchmetrics.wrappers import ClasswiseWrapper

# Ignore warning about default grid_sample and affine_grid behavior triggered by kornia
warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed",
)

logger = logging.getLogger(__name__)


class SegmentationUnetPlus(LightningModule):
    """Segmentation UNetPlus model."""

    def __init__(  # noqa: PLR0913
        self,
        encoder: str,
        in_channels: int,
        num_classes: int,
        max_samples: int,
        mean: list[float],
        std: list[float],
        data_type_max: float,
        loss: Callable,
        lr: float,
        weight_decay: float,
        weights: str | None = None,
        class_labels: list[str] | None = None,
        class_colors: list[str] | None = None,
        weights_from_checkpoint_path: str | None = None,
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.save_hyperparameters()
        self.max_samples = max_samples
        self.mean = mean
        self.std = std
        self.lr = lr
        self.weight_decay = weight_decay
        self.data_type_max = data_type_max
        self.class_colors = class_colors
        self.input_channels = in_channels
        self.num_classes = num_classes
        self.threshold = 0.5
        self.weights_from_checkpoint_path = weights_from_checkpoint_path
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder,
            in_channels=self.input_channels,
            encoder_weights=weights,
            classes=self.num_classes,
        )
        if weights_from_checkpoint_path:
            logger.info(
                "Loading weights from checkpoint: %s",
                weights_from_checkpoint_path,
            )
            checkpoint = torch.load(weights_from_checkpoint_path)
            self.load_state_dict(checkpoint["state_dict"])
        self.loss = loss
        num_classes = num_classes + 1 if num_classes == 1 else num_classes
        self.iou_metric = MeanIoU(
            num_classes=num_classes,
            per_class=True,
            input_format="index",
            include_background=True,
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

    def configure_optimizers(self) -> list[list[dict[str, Any]]]:
        """Configure optimizers."""
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=1e-3,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

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
        y = batch["mask"]
        batch_size = x.shape[0]
        # y = y.squeeze(1).long()
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
        batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run validation step."""
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
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
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y = y.squeeze(1).long()
        if self.num_classes == 1:
            y_hat = (y_hat.sigmoid().squeeze(1) > self.threshold).long()
        else:
            y_hat = y_hat.softmax(dim=1).argmax(dim=1)

        metrics = self.iou_classwise_metric(y_hat, y)
        metrics["test_loss"] = loss

        if self._total_samples_visualized < self.max_samples:
            remaining_samples = self.max_samples - self._total_samples_visualized
            num_samples = min(remaining_samples, len(x))
            for i in range(num_samples):
                image = x[i]
                image_name = batch["image_name"][i]
                image = denormalization(
                    image,
                    mean=self.mean,
                    std=self.std,
                    data_type_max=self.data_type_max,
                )
                fig = visualize_prediction(
                    image,
                    y[i],
                    y_hat[i],
                    image_name,
                    self.num_classes,
                    class_colors=self.class_colors,
                )
                artifact_file = f"test/{Path(image_name).stem}/idx_{i}.png"
                self.logger.experiment.log_figure(
                    figure=fig,
                    artifact_file=artifact_file,
                    run_id=self.logger.run_id,
                )
                self._total_samples_visualized += 1
                if self._total_samples_visualized >= self.max_samples:
                    break

        self.log_dict(
            metrics,
            batch_size=batch_size,
            prog_bar=False,
            logger=True,
            on_step=False,
            rank_zero_only=True,
        )

    def on_train_end(self) -> None:
        """Run on train end."""
        if self.trainer.is_global_zero and self.trainer.checkpoint_callback is not None:
            best_model_path = self.trainer.checkpoint_callback.best_model_path
            if best_model_path:
                logger.info("Best model path: %s", best_model_path)
                best_model_dir = Path(best_model_path).parent
                best_model_name = Path(best_model_path).stem
                best_model_export_path = str(
                    best_model_dir / f"{best_model_name}_scripted.pt",
                )
                self.export_model(
                    best_model_path,
                    best_model_export_path,
                    self.trainer.datamodule,
                )

    def export_model(
        self,
        checkpoint_path: str,
        export_path: str,
        datamodule: LightningDataModule,
    ) -> None:
        """Export model."""
        map_location = "cuda"
        if self.device.type == "cpu":
            map_location = "cpu"
        best_model = self.__class__.load_from_checkpoint(
            checkpoint_path,
            weights_from_checkpoint_path=None,
            map_location=map_location,
        )
        input_shape = (1, self.input_channels, *datamodule.patch_size)
        device = torch.device(map_location)
        script_model = ScriptModel(
            model=best_model.model,
            device=device,
            num_classes=self.num_classes,
            input_shape=input_shape,
            mean=self.mean,
            std=self.std,
            image_min=0,
            image_max=self.data_type_max,
            norm_min=0.0,
            norm_max=1.0,
            from_logits=True,
        )
        scripted_model = torch.jit.script(script_model)
        scripted_model.save(export_path)
        logger.info("Model exported to TorchScript")

import pytorch_lightning as pl
from torch import Tensor
from typing import Any, Callable, Dict, List
from hydra.utils import instantiate
from torchmetrics.segmentation import MeanIoU
from torchmetrics.wrappers import ClasswiseWrapper
from models.segformer import SegFormer

class SegmentationSegformer(pl.LightningModule):
    def __init__(self, 
                 encoder: str,
                 in_channels: int, 
                 num_classes: int,
                 loss: Callable,
                 class_labels: List[str] = None,
                 **kwargs: Any):
        super().__init__()
        self.save_hyperparameters()
        self.model = SegFormer(encoder, in_channels, num_classes)
        self.loss = loss
        self.metric = MeanIoU(num_classes=num_classes, per_class=True)
        self.labels = [str(i) for i in range(num_classes)] if class_labels is None else class_labels
        self.classwise_metric = ClasswiseWrapper(self.metric, labels=self.labels)

    def forward(self, image: Tensor) -> Tensor:
        return self.model(image)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        x = batch["image"]
        y = batch["label"]
        y_pred = self(x)
        loss: Tensor = self.loss(y_pred, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        val_ious = self.classwise_metric(y_hat, y)
        self.log_dict({"val_loss": loss, "val_iou_per_class": val_ious}, 
                      prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        test_ious = self.classwise_metric(y_hat, y)
        self.log_dict({"test_loss": loss, "test_iou_per_class": test_ious}, 
                      prog_bar=True, logger=True, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = instantiate(self.hparams["optimizer"], params=self.model.parameters())
        scheduler = instantiate(self.hparams["lr_scheduler"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss"
                }
        }
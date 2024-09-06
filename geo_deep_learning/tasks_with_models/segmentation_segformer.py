from torch import Tensor
from typing import Any, Callable, Dict, List
from lightning.pytorch import LightningModule
from torchmetrics.segmentation import MeanIoU
from torchmetrics.wrappers import ClasswiseWrapper
from models.segformer import SegFormer

class SegmentationSegformer(LightningModule):
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
        y = y.squeeze(1).long()
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_hat = y_hat.argmax(dim=1)
        self.log('train_loss', loss, 
                 prog_bar=True, logger=True, 
                 on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=0)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        print(f"batch_index: {batch_idx},  x: {x.shape}, y: {y.shape}")
        y = y.squeeze(1).long()
        y_hat = self(x)
        print(f"batch_index: {batch_idx},  y_hat: {y_hat.shape}, y: {y.shape}")
        loss = self.loss(y_hat, y)
        y_hat = y_hat.softmax(dim=1).argmax(dim=1)
        print(f"batch_index: {batch_idx},  y_hat: {y_hat.shape}, y: {y.shape}")
        val_metrics = self.classwise_metric(y_hat, y)
        val_metrics["val_loss"] = loss
        self.log_dict(val_metrics, 
                      prog_bar=True, logger=True, 
                      on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=0)
    
    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        y = y.squeeze(1).long()
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_hat = y_hat.argmax(dim=1)
        test_metrics = self.classwise_metric(y_hat, y)
        test_metrics["test_loss"] = loss
        self.log_dict(test_metrics, 
                      prog_bar=True, logger=True, 
                      on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=0)
    
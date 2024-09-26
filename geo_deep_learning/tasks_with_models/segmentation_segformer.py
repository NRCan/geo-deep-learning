import numpy as np
from torch import Tensor
from typing import Any, Callable, Dict, List
from lightning.pytorch import LightningModule
from torchmetrics.classification import MulticlassJaccardIndex
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
        self.save_hyperparameters(ignore=["loss"])
        self.model = SegFormer(encoder, in_channels, num_classes)
        self.loss = loss
        self.metric= MulticlassJaccardIndex(num_classes=num_classes, average=None, zero_division=np.nan)
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
                 on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        y = y.squeeze(1).long()
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_hat = y_hat.softmax(dim=1).argmax(dim=1)
        mean_iou = self.metric(y_hat, y)
        self.log('val_loss', loss,
                    prog_bar=True, logger=True, 
                    on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        # self.log_dict(val_metrics, 
        #               prog_bar=True, logger=True, 
        #               on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=0)
    
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
                      on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
    
import numpy as np
import matplotlib.pyplot as plt
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
        self.save_hyperparameters()
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
        self.log('val_loss', loss,
                    prog_bar=True, logger=True, 
                    on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        return y_hat
    
    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        y = y.squeeze(1).long()
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_hat = y_hat.softmax(dim=1).argmax(dim=1)
        test_metrics = self.classwise_metric(y_hat, y)
        test_metrics["loss"] = loss
        self.log_dict(test_metrics, 
                      prog_bar=True, logger=True, 
                      on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
    
    # def log_images(self, images, labels, predictions, stage):
    #     fig, axes = plt.subplots(len(images), 3, figsize=(15, 5 * len(images)))
    #     for i, (img, lbl, pred) in enumerate(zip(images, labels, predictions)):
    #         if len(images) == 1:
    #             ax_img, ax_lbl, ax_pred = axes
    #         else:
    #             ax_img, ax_lbl, ax_pred = axes[i]
    #         ax_img.imshow(img.permute(1, 2, 0).cpu().numpy())
    #         ax_img.set_title('Image')
    #         ax_lbl.imshow(lbl.cpu().numpy())
    #         ax_lbl.set_title('Label')
    #         ax_pred.imshow(pred.cpu().numpy())
    #         ax_pred.set_title('Prediction')
    #         for ax in [ax_img, ax_lbl, ax_pred]:
    #             ax.axis('off')
    #     plt.tight_layout()
    #     mlflow.log_figure(fig, f"{stage}_predictions.png")
    #     plt.close(fig)
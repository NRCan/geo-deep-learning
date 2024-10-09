import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Any, Callable, Dict, List
from lightning.pytorch import LightningModule, LightningDataModule
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.wrappers import ClasswiseWrapper
from models.segformer import SegFormer
from tools.script_model import script_model

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
    
    def on_train_end(self):
        if self.trainer.is_global_zero and self.trainer.checkpoint_callback is not None:
            best_model_path = self.trainer.checkpoint_callback.best_model_path
            if best_model_path:
                print(f"Best model path: {best_model_path}")
                best_model_dir = Path(best_model_path).parent
                best_model_name = Path(best_model_path).stem
                best_model_export_path = str(best_model_dir / f"{best_model_name}_scripted.pt")
                self.export_model(best_model_export_path, self.trainer.datamodule)
                
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, **kwargs):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model = cls(**checkpoint['hyper_parameters'])
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    def export_model(self, checkpoint_path: str, export_path: str, datamodule: LightningDataModule):
        best_model = self.load_from_checkpoint(checkpoint_path)
        best_model.eval()
        
        scrpted_model = script_model(best_model, datamodule)
        dummy_input = torch.randn(1, self.hparams.in_channels, *datamodule.patch_size)        
        traced_model = torch.jit.trace(scrpted_model, dummy_input)
        torch.jit.save(traced_model, export_path)
        print(f"Model exported to TorchScript {export_path}")
        
    
    
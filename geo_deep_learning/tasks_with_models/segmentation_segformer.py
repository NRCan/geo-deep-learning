import warnings
# Ignore warning about default grid_sample and affine_grid behavior triggered by kornia
warnings.filterwarnings("ignore", message="Default grid_sample and affine_grid behavior has changed")
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Any, Callable, Dict, List, Optional
from lightning.pytorch import LightningModule, LightningDataModule
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from torchmetrics.segmentation import MeanIoU
from torchmetrics.wrappers import ClasswiseWrapper
from models.segformer import SegFormer
from tools.script_model import ScriptModel
from tools.utils import denormalization
from tools.visualization import visualize_prediction

class SegmentationSegformer(LightningModule):
    def __init__(self, 
                 encoder: str,
                 in_channels: int,
                 num_classes: int,
                 max_samples: int,
                 mean: List[float],
                 std: List[float],
                 data_type_max: float,
                 loss: Callable,
                 optimizer: OptimizerCallable = torch.optim.Adam,
                 scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
                 scheduler_config: Optional[Dict[str, Any]] = {"interval": "epoch"},
                 freeze_encoder: bool = False,
                 weights: str = None,
                 class_labels: List[str] = None,
                 class_colors: List[str] = None,
                 weights_from_checkpoint_path: Optional[str] = None,
                 **kwargs: Any):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config
        self.max_samples = max_samples
        self.mean = mean
        self.std = std
        self.data_type_max = data_type_max
        self.class_colors = class_colors
        self.input_channels = in_channels
        self.num_classes = num_classes
        self.model = SegFormer(encoder, in_channels, weights, freeze_encoder, self.num_classes)
        if weights_from_checkpoint_path:
            print(f"Loading weights from checkpoint: {weights_from_checkpoint_path}")
            checkpoint = torch.load(weights_from_checkpoint_path)
            self.load_state_dict(checkpoint['state_dict'])
        self.loss = loss
        num_classes = num_classes + 1 if num_classes == 1 else num_classes
        self.iou_metric = MeanIoU(num_classes=num_classes,
                                  per_class=True,
                                  input_format="index",
                                  include_background=True
                                 )
        self.labels = [str(i) for i in range(num_classes)] if class_labels is None else class_labels
        self.iou_classwise_metric = ClasswiseWrapper(self.iou_metric, labels=self.labels)
        self._total_samples_visualized = 0
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)
        return [optimizer], [{'scheduler': scheduler, **self.scheduler_config}]
    
    def forward(self, image: Tensor) -> Tensor:
        return self.model(image)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        y = y.squeeze(1).long()
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        
        self.log('train_loss', loss, 
                 batch_size=batch_size,
                 prog_bar=True, logger=True, 
                 on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        y = y.squeeze(1).long()
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss,
                 batch_size=batch_size,
                 prog_bar=True, logger=True, 
                 on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        if self.num_classes == 1:
            y_hat = (y_hat.sigmoid().squeeze(1) > 0.5).long()
        else:
            y_hat = y_hat.softmax(dim=1).argmax(dim=1)
            
        return y_hat
    
    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        y = y.squeeze(1).long()
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        
        if self.num_classes == 1:
            y_hat = (y_hat.sigmoid().squeeze(1) > 0.5).long()
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
                image = denormalization(image, mean=self.mean, std=self.std, data_type_max=self.data_type_max)
                fig = visualize_prediction(image,
                                            y[i],
                                            y_hat[i],
                                            image_name,
                                            self.num_classes,
                                            class_colors=self.class_colors)
                artifact_file = f"test/{Path(image_name).stem}/idx_{i}.png"
                self.logger.experiment.log_figure(figure=fig,
                                                  artifact_file=artifact_file,
                                                  run_id=self.logger.run_id)
                self._total_samples_visualized += 1
                if self._total_samples_visualized >= self.max_samples:
                    break
        
        self.log_dict(metrics, 
                      batch_size=batch_size,
                      prog_bar=False, logger=True, 
                      on_step=False, rank_zero_only=True)
    
    def on_train_end(self):
        if self.trainer.is_global_zero and self.trainer.checkpoint_callback is not None:
            best_model_path = self.trainer.checkpoint_callback.best_model_path
            if best_model_path:
                print(f"Best model path: {best_model_path}")
                best_model_dir = Path(best_model_path).parent
                best_model_name = Path(best_model_path).stem
                best_model_export_path = str(best_model_dir / f"{best_model_name}_scripted.pt")
                self.export_model(best_model_path, best_model_export_path, self.trainer.datamodule)
    
    def export_model(self, checkpoint_path: str, export_path: str, datamodule: LightningDataModule):
        map_location = "cuda"
        if self.device.type == "cpu":
            map_location = "cpu"
        best_model = self.__class__.load_from_checkpoint(checkpoint_path,
                                                         weights_from_checkpoint_path=None,
                                                         map_location=map_location)
        input_shape = (1, self.input_channels, *datamodule.patch_size)
        device = torch.device(map_location)
        script_model = ScriptModel(model=best_model.model,
                                   device=device,
                                   num_classes=self.num_classes,
                                   input_shape=input_shape, mean=self.mean, std=self.std,
                                   image_min=0, image_max=self.data_type_max, norm_min=0.0, norm_max=1.0,
                                   from_logits=True)
        scripted_model = torch.jit.script(script_model)
        scripted_model.save(export_path)
        print(f"Model exported to TorchScript")
        
    
    
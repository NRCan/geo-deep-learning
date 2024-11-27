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
from torchmetrics.segmentation import MeanIoU
from torchmetrics.wrappers import ClasswiseWrapper
from models.dofa.dofa_seg import DOFASeg
from tools.utils import denormalization
from tools.script_model import script_model
from tools.visualization import visualize_prediction

class SegmentationDOFA(LightningModule):
    def __init__(self, 
                 encoder: str,
                 pretrained: bool,
                 image_size: tuple[int, int],
                 in_channels: int,
                 num_classes: int,
                 max_samples: int,
                 mean: List[float],
                 std: List[float],
                 data_type_max: float,
                 loss: Callable,
                 class_labels: List[str] = None,
                 class_colors: List[str] = None,
                 weights_from_checkpoint_path: Optional[str] = None,
                 **kwargs: Any):
        super().__init__()
        self.save_hyperparameters()
        self.class_colors = class_colors
        self.max_samples = max_samples
        self.mean = mean
        self.std = std
        self.data_type_max = data_type_max
        self.num_classes = num_classes
        self.model = DOFASeg(encoder, pretrained, image_size, self.num_classes)
        if weights_from_checkpoint_path:
            print(f"Loading weights from checkpoint: {weights_from_checkpoint_path}")
            checkpoint = torch.load(weights_from_checkpoint_path)
            self.load_state_dict(checkpoint['state_dict'])
        self.loss = loss
        self.iou_metric = MeanIoU(num_classes=num_classes,
                                  per_class=True,
                                  input_format="index",
                                  include_background=True
                                 )
        self.labels = [str(i) for i in range(num_classes)] if class_labels is None else class_labels
        self.iou_classwise_metric = ClasswiseWrapper(self.iou_metric, labels=self.labels)
        self._total_samples_visualized = 0

    def forward(self, image: Tensor) -> Tensor:
        return self.model(image)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        x = batch["image"]
        y = batch["mask"]
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
        y = batch["mask"]
        y = y.squeeze(1).long()
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_hat = y_hat.softmax(dim=1).argmax(dim=1)
        
        if batch_idx == 0: 
            print(f"Epoch {self.current_epoch} - Validation Loss: {loss.item()}")
        
        self.log('val_loss', loss,
                    prog_bar=True, logger=True, 
                    on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        return y_hat
    
    def test_step(self, batch, batch_idx):
        x = batch["image"]
        image_names = batch["image_name"]
        y = batch["mask"]
        y = y.squeeze(1).long()
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_hat = y_hat.softmax(dim=1).argmax(dim=1)
        metrics = self.iou_classwise_metric(y_hat, y)
        metrics["test_loss"] = loss
        
        if self._total_samples_visualized < self.max_samples:
            remaining_samples = self.max_samples - self._total_samples_visualized
            num_samples = min(remaining_samples, len(x))
            for i in range(num_samples):
                image = x[i]
                image_name = image_names[i]
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
        input_channels = self.hparams["in_channels"]
        map_location = "cuda"
        if self.device.type == "cpu":
            map_location = "cpu"
        best_model = self.__class__.load_from_checkpoint(checkpoint_path, map_location=map_location)
        best_model.eval()
        
        scrpted_model = script_model(best_model.model, datamodule, self.num_classes, from_logits=True)
        patch_size = datamodule.patch_size
        dummy_input = torch.rand(1, input_channels, *patch_size, device=torch.device(map_location))       
        traced_model = torch.jit.trace(scrpted_model, dummy_input)
        torch.jit.save(traced_model, export_path)
        print(f"Model exported to TorchScript")
    
    
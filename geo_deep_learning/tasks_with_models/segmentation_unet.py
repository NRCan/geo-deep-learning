import warnings
# Ignore warning about default grid_sample and affine_grid behavior triggered by kornia
warnings.filterwarnings("ignore", message="Default grid_sample and affine_grid behavior has changed")
import numpy as np
import segmentation_models_pytorch as smp
import torch
from pathlib import Path
from torch import Tensor
from typing import Any, Callable, Dict, List, Optional
from lightning.pytorch import LightningModule, LightningDataModule
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from torchmetrics.segmentation import MeanIoU
from torchmetrics.classification import Accuracy, AveragePrecision, Recall, F1Score, JaccardIndex
from torchmetrics.wrappers import ClasswiseWrapper
from tools.script_model import script_model
from tools.utils import denormalization
from tools.visualization import visualize_prediction
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from tqdm import tqdm
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss
from lightning.pytorch.callbacks import ModelCheckpoint


class SegmentationUnet(LightningModule):
    def __init__(self, 
                 encoder: str,
                 in_channels: int,
                 num_classes: int,
                 max_samples: int,
                 mean: List[float],
                 std: List[float],
                 data_type_max: float,
                 lr: float,
                 weight_decay: float,
                 class_weight: float = 1,
                 weights: str = None,
                 class_labels: List[str] = None,
                 class_colors: List[str] = None,
                 optimizer: OptimizerCallable = torch.optim.Adam,
                 scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
                 scheduler_config: Optional[Dict[str, Any]] = {"interval": "epoch"},
                 weights_from_checkpoint_path: Optional[str] = None,
                 **kwargs: Any):
        super().__init__()
        self.save_hyperparameters()
        self.max_samples = max_samples
        self.mean = mean
        self.std = std
        self.lr = lr
        self.weight_decay = weight_decay
        self.data_type_max = data_type_max
        self.class_colors = class_colors
        self.num_classes = num_classes
        self.weights_from_checkpoint_path = weights_from_checkpoint_path
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config

        self.model = smp.Unet(encoder_name=encoder, in_channels=in_channels, encoder_weights=weights, classes=self.num_classes)
        if weights_from_checkpoint_path:
            checkpoint = torch.load(weights_from_checkpoint_path)
            del checkpoint['state_dict']['loss.pos_weight']
            self.load_state_dict(checkpoint['state_dict'])
        
        self.loss = SoftBCEWithLogitsLoss(pos_weight=torch.tensor(5.949), ignore_index=255)
        torch.set_float32_matmul_precision('high')
        num_classes = num_classes + 1 if num_classes == 1 else num_classes
        self.iou_metric = MeanIoU(num_classes=num_classes,
                                  per_class=True,
                                  input_format="index",
                                  include_background=True,
                                 )
        self.precision_metric = AveragePrecision(ignore_index=255, num_classes=num_classes, task="multiclass", average="none")
        self.accuracy_metric = Accuracy(ignore_index=255, num_classes=num_classes, task="multiclass", average="none")
        self.recall_metric = Recall(ignore_index=255, num_classes=num_classes, task="multiclass", average="none")
        self.f1score_metric = F1Score(ignore_index=255, num_classes=num_classes, task="multiclass", average="none")
        self.jaccard_metric = JaccardIndex(ignore_index=255, num_classes=num_classes, task="multiclass", average="none")

        self.labels = [str(i) for i in range(num_classes)] if class_labels is None else class_labels
        
        self.iou_classwise_metric = ClasswiseWrapper(self.iou_metric, labels=self.labels)
        self.accuracy_classwise_metric = ClasswiseWrapper(self.accuracy_metric, labels=self.labels)
        self.precision_classwise_metric = ClasswiseWrapper(self.precision_metric, labels=self.labels)
        self.recall_classwise_metric = ClasswiseWrapper(self.recall_metric, labels=self.labels)
        self.f1score_classwise_metric = ClasswiseWrapper(self.f1score_metric, labels=self.labels)
        self.jaccard_classwise_metric = ClasswiseWrapper(self.jaccard_metric, labels=self.labels)
        self._total_samples_visualized = 0
    
            
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        # scheduler = self.scheduler(optimizer)
        scheduler = OneCycleLR(optimizer, max_lr=1e-3, pct_start=0.3, total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [{'scheduler': scheduler, **self.scheduler_config}]
    
    def forward(self, image: Tensor) -> Tensor:
        return self.model(image)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        #y = y.squeeze(1).long() 
        y_hat = self(x)
        #loss = self.loss(y_hat, y)

        y_onehot = torch.nn.functional.one_hot(y.squeeze(1).long(), num_classes=2).permute(0, 3, 1, 2).float()
        loss = self.loss(y_hat, y_onehot)
        
        self.log('trn_loss', loss, 
                 batch_size=batch_size,
                 prog_bar=True, logger=True, 
                 on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        y_hat = self(x)

        y_onehot = torch.nn.functional.one_hot(y.squeeze(1).long(), num_classes=2).permute(0, 3, 1, 2).float()
        if self.num_classes == 1:
            y_hat_prob = y_hat.sigmoid().squeeze(1)
            y_hat_class = (y_hat_prob > 0.5).long()
        else:
            y_hat_prob = y_hat.softmax(dim=1)
            y_hat_class = y_hat_prob.argmax(dim=1)

        loss = self.loss(y_hat, y_onehot)

        #y = y.squeeze(1).long() 
        #loss = self.loss(y_hat, y)

        self.log('val_loss', loss,
                 batch_size=batch_size,
                 prog_bar=True, logger=True, 
                 on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        
        
        # if self.num_classes == 1:
        #     y_hat = (y_hat.sigmoid().squeeze(1) > 0.5).long()
        # else:
        #     y_hat = y_hat.softmax(dim=1).argmax(dim=1)
        # y = y.squeeze(1).long() 
        # metrics = self.iou_classwise_metric(y_hat, y)
        
        # metrics = {f"val_{k}": v for k, v in metrics.items()}

        # self.log_dict(metrics, 
        #               batch_size=batch_size,
        #               prog_bar=False, logger=True, 
        #               on_step=False, on_epoch=True, rank_zero_only=True, sync_dist=True)
        # return y_hat

        y = y.squeeze(1).long() 
        metrics = self.iou_classwise_metric(y_hat_class, y)
        metrics.update(self.jaccard_classwise_metric(y_hat_class, y))
        metrics.update(self.accuracy_classwise_metric(y_hat_class.float(), y))
        metrics.update(self.f1score_classwise_metric(y_hat_class, y))

        metrics.update(self.recall_classwise_metric(y_hat_prob, y))
        metrics.update(self.precision_classwise_metric(y_hat_prob, y))
        
        metrics = {f"val_{k}": v for k, v in metrics.items()}

        self.log_dict(metrics, 
                    batch_size=batch_size,
                    prog_bar=False, logger=True, 
                    on_step=False, on_epoch=True, rank_zero_only=True, sync_dist=True)
        
        # reset metrics to free memory
        self.iou_metric.reset()
        self.accuracy_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1score_metric.reset()
        self.jaccard_metric.reset()

        # cleanup
        del metrics

        return y_hat_class
    
    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        y_hat = self(x)

        #y_hat = y_hat.squeeze(1).long()
        # loss = self.loss(y_hat, y)
        y_onehot = torch.nn.functional.one_hot(y.squeeze(1).long(), num_classes=2).permute(0, 3, 1, 2).float()
        if self.num_classes == 1:
            y_hat_prob = y_hat.sigmoid().squeeze(1)
            y_hat_class = (y_hat_prob > 0.5).long()
        else:
            y_hat_prob = y_hat.softmax(dim=1)
            y_hat_class = y_hat_prob.argmax(dim=1)

        loss = self.loss(y_hat, y_onehot)

        # if self.num_classes == 1:
        #     y_hat = (y_hat.sigmoid().squeeze(1) > 0.5).long()
        # else:
        #     y_hat = y_hat.softmax(dim=1).argmax(dim=1)
        # y = y.squeeze(1).long() 
        # metrics = self.iou_classwise_metric(y_hat, y)
        # metrics["tst_loss"] = loss

        y = y.squeeze(1).long() 
        metrics = self.iou_classwise_metric(y_hat_class, y)
        metrics.update(self.jaccard_classwise_metric(y_hat_class, y))
        metrics.update(self.accuracy_classwise_metric(y_hat_class.float(), y))
        metrics.update(self.f1score_classwise_metric(y_hat_class, y))

        metrics.update(self.recall_classwise_metric(y_hat_prob, y))
        metrics.update(self.precision_classwise_metric(y_hat_prob, y))
        metrics = {f"tst_{k}": v for k, v in metrics.items()}

        metrics["tst_loss"] = loss
        
        if self._total_samples_visualized < self.max_samples:
            remaining_samples = self.max_samples - self._total_samples_visualized
            num_samples = min(remaining_samples, len(x))
            for i in range(num_samples):
                if i < 10:
                    idx = i
                else:
                    idx = np.random.randint(0, len(x))
                    
                image = x[idx]
                image_name = batch["image_name"][idx]
                image = denormalization(image, mean=self.mean, std=self.std, data_type_max=self.data_type_max)
                fig = visualize_prediction(image,
                                            y[idx],
                                            y_hat_class[idx],
                                            image_name,
                                            self.num_classes,
                                            class_colors=self.class_colors)
                artifact_file = f"test/{Path(image_name).stem}/idx_{idx}.png"
                self.logger.experiment.log_figure(figure=fig,
                                                  artifact_file=artifact_file,
                                                  run_id=self.logger.run_id)
                self._total_samples_visualized += 1
                if self._total_samples_visualized >= self.max_samples:
                    break

        self.log_dict(metrics, 
                    batch_size=batch_size,
                    prog_bar=False, logger=True, 
                    on_step=False, rank_zero_only=True, sync_dist=True)
        
        # reset metrics to free memory
        self.iou_metric.reset()
        self.accuracy_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1score_metric.reset()
        self.jaccard_metric.reset()

        # cleanup
        del metrics
    
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
        best_model = self.__class__.load_from_checkpoint(checkpoint_path,
                                                         weights_from_checkpoint_path=None,
                                                         map_location=map_location)
        best_model.eval()
        
        scrpted_model = script_model(best_model.model, datamodule, self.num_classes, from_logits=True)
        patch_size = datamodule.patch_size
        dummy_input = torch.rand(1, input_channels, *patch_size, device=torch.device(map_location))       
        traced_model = torch.jit.trace(scrpted_model, dummy_input)
        torch.jit.save(traced_model, export_path)
        print(f"Model exported to TorchScript")
        
    
    
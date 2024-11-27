import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import ListedColormap
from lightning.pytorch.callbacks import ModelCheckpoint
from tools.visualization import visualize_prediction
from tools.utils import denormalization

class SegmentationCallback(ModelCheckpoint):
    
    def __init__(self, 
                 max_samples: int = 3,
                 mean = None, 
                 std = None,
                 num_classes = None,
                 data_type_max = 255,
                 class_colors = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_samples = max_samples
        self.mean = mean
        self.std = std
        self.num_classes = num_classes
        self.data_type_max = data_type_max
        self.class_colors = class_colors
        self.current_batch = None
        self.current_outputs = None
        self._setup_colormap()
    
    def on_validation_batch_end(self,
                                trainer,
                                pl_module,
                                outputs,
                                batch,
                                batch_idx,
                                dataloader_idx=0):
        if trainer.is_global_zero:
            self.current_batch = batch
            self.current_outputs = outputs
    
    def _save_checkpoint(self, trainer, filepath):
        print(f"Saving checkpoint at epoch {trainer.current_epoch} with val_loss: {self.best_model_score}")
        super()._save_checkpoint(trainer, filepath)
        if trainer.is_global_zero:
            self._log_visualizations(trainer, None)
        
    def _setup_colormap(self):
        if self.class_colors is None:
            # Default colormap for unknown number of classes
            self.cmap = plt.get_cmap('tab20')
        else:
            self.cmap = ListedColormap(self.class_colors)
    
    def _log_visualizations(self, trainer, batch_idx):
        if self.current_batch is not None and self.current_outputs is not None:
            image_batch = self.current_batch["image"]
            mask_batch = self.current_batch["mask"]
            batch_image_name = self.current_batch["image_name"]
            batch_mask_name = self.current_batch["mask_name"]
            batch_size = image_batch.shape[0]
            N = min(self.max_samples, batch_size)
            num_classes = mask_batch.max().item() + 1 if self.num_classes is None else self.num_classes
            try:
                for i in range(N):
                    image = image_batch[i]
                    mask = mask_batch[i]
                    image_name = batch_image_name[i]
                    output = self.current_outputs[i]
                    image = denormalization(image, self.mean, self.std, self.data_type_max)
                    fig = visualize_prediction(image, mask, output, 
                                            sample_name=image_name,
                                            num_classes=num_classes,
                                            class_colors=self.class_colors)
                    artifact_file = f"val/{Path(image_name).stem}/idx_{i}_epoch_{trainer.current_epoch}.png"
                    trainer.logger.experiment.log_figure(figure=fig,
                                                        artifact_file = artifact_file,
                                                        run_id=trainer.logger.run_id)
            except Exception as e:
                print(f"Error in logging visualizations: {e}")
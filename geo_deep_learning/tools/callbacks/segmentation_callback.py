import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from lightning.pytorch.callbacks import ModelCheckpoint

class SegmentationCallback(ModelCheckpoint):
    
    def __init__(self, max_samples: int = 3, class_colors = None,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_samples = max_samples
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
        self.current_batch = batch
        self.current_outputs = outputs
    
    
    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        self._log_visualizations(trainer)
        
    def _setup_colormap(self):
        if self.class_colors is None:
            # Default colormap for unknown number of classes
            self.cmap = plt.get_cmap('tab20')
        else:
            self.cmap = ListedColormap(self.class_colors)
    
    def _log_visualizations(self, trainer):
        if self.current_batch is not None and self.current_outputs is not None:
            image_batch = self.current_batch["image"]
            label_batch = self.current_batch["label"]
            batch_size = image_batch.shape[0]
            N = min(self.max_samples, batch_size)
            num_classes = label_batch.max().item() + 1 if self.class_colors is None else len(self.class_colors)
            
            fig, axes = plt.subplots(N, 3, figsize=(15, 5 * N))
            axes = axes.reshape(N, 3) if N > 1 else axes.reshape(1, 3)
            
            
            for i in range(N):
                image = image_batch[i]
                label = label_batch[i]
                output = self.current_outputs[i]
                image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                label = label.squeeze(0).long().cpu().numpy()
                output = output.cpu().numpy()   
                ax_image, ax_label, ax_output = axes[i]
                
                ax_image.imshow(image)
                ax_image.set_title('Image')
                ax_image.axis("off")
                
                ax_label.imshow(label, cmap=self.cmap, vmin=0, vmax=num_classes-1)
                ax_label.set_title('Label')
                ax_label.axis("off")
                
                ax_output.imshow(output, cmap=self.cmap, vmin=0, vmax=num_classes-1)
                ax_output.set_title('Output')
                ax_output.axis("off")
            
            plt.tight_layout()
            artifact_file = f"val/predictions_epoch_{trainer.current_epoch}.png"
            trainer.logger.experiment.log_figure(figure=fig, 
                                                 artifact_file = artifact_file,
                                                 run_id=trainer.logger.run_id)
            plt.close(fig)
            
            self.current_batch = None
            self.current_outputs = None
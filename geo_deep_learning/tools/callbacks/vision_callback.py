
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import Callback

class SegmentationVisualizationCallback(Callback):
    
    def __init__(self, 
                 max_samples: int = 5):
        self.max_samples = max_samples
        self.samples_logged = 0
    
    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self.samples_logged = 0
    
    def on_validation_batch_end(self, 
                                trainer, 
                                pl_module, 
                                outputs, 
                                batch, 
                                batch_idx, 
                                dataloader_idx=0):
        
        if self.samples_logged < self.max_samples:
        
            image_batch = batch["image"]
            label_batch = batch["label"]
            batch_size = image_batch.shape[0]
            N = min(self.max_samples - self.samples_logged, batch_size)
            fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
            axes = axes.reshape(N, 3) if N > 1 else axes.reshape(1, 3)
            for i in range(N):
                image = image_batch[i]
                label = label_batch[i]
                output = outputs[i]
                image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                label = label.squeeze(0).long().cpu().numpy()
                output = output.cpu().numpy()   
                ax_image, ax_label, ax_output = axes[i]
                
                ax_image.imshow(image)
                ax_image.set_title('Image')
                ax_label.imshow(label)
                ax_label.set_title('Label')
                ax_output.imshow(output)
                ax_output.set_title('Output')
                for ax in [ax_image, ax_label, ax_output]:
                    ax.axis("off")
            
            plt.tight_layout()
            artifact_file = f"val/predictions_{self.samples_logged}.png"
            trainer.logger.experiment.log_figure(figure=fig, 
                                                 artifact_file = artifact_file,
                                                 run_id=trainer.logger.run_id)
            plt.close(fig)
            self.samples_logged += N
        
    
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from tools.visualization import visualize_prediction
from tools.utils import denormalization
from pathlib import Path

class VisualizationCallback(Callback):
    def __init__(self, max_samples=3, mean=None, std=None, num_classes=None, 
                 data_type_max=255, class_colors=None):
        super().__init__()
        self.max_samples = max_samples
        self.mean = mean
        self.std = std
        self.num_classes = num_classes + 1 if num_classes == 1 else num_classes
        self.data_type_max = data_type_max
        self.class_colors = class_colors
        self.current_batch = None
        self.current_outputs = None

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if trainer.is_global_zero:
            self.current_batch = batch
            self.current_outputs = outputs

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Check if we saved a checkpoint this batch
        if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback.best_model_score is not None:
            current_score = trainer.callback_metrics.get('val_loss')
            if current_score == trainer.checkpoint_callback.best_model_score:
                self._log_visualizations(trainer)

    def _log_visualizations(self, trainer):
        if self.current_batch is None or self.current_outputs is None:
            return
            
        try:
            image_batch = self.current_batch["image"]
            mask_batch = self.current_batch["mask"]
            batch_image_name = self.current_batch["image_name"]
            
            N = min(self.max_samples, len(image_batch))
            for i in range(N):
                image = image_batch[i]
                image_name = batch_image_name[i]
                image = denormalization(image, self.mean, self.std, self.data_type_max)
                fig = visualize_prediction(
                    image, mask_batch[i], self.current_outputs[i],
                    image_name, self.num_classes, class_colors=self.class_colors
                )
                # artifact_file = f"val/{Path(image_name).stem}/epoch_{trainer.current_epoch}.png"
                artifact_file = f"val/{Path(image_name).stem}/idx_{i}_epoch_{trainer.current_epoch}.png"
                trainer.logger.experiment.log_figure(
                    figure=fig,
                    artifact_file=artifact_file,
                    run_id=trainer.logger.run_id
                )
        except Exception as e:
            print(f"Error in visualization: {e}")
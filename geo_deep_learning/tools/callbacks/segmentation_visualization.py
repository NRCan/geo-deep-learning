"""Visualization callback for segmentation models."""

import logging

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

logger = logging.getLogger(__name__)


class VisualizationCallback(Callback):
    """Visualization callback on best model save."""

    def __init__(self, max_samples: int = 6) -> None:
        """
        Initialize callback.

        Args:
            max_samples: Maximum number of samples to visualize

        """
        self.max_samples = max_samples
        self.current_batch = None
        self.current_outputs = None
        self.last_best_score = None

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
        outputs: object,
        batch: object,
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Store the last validation batch for visualization."""
        if trainer.is_global_zero:
            self.current_batch = batch
            self.current_outputs = outputs

    @rank_zero_only
    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Trigger visualization when best model is saved."""
        if (
            hasattr(trainer, "checkpoint_callback")
            and trainer.checkpoint_callback.best_model_score is not None
        ):
            current_best_score = trainer.checkpoint_callback.best_model_score
            if self.last_best_score != current_best_score:
                self.last_best_score = current_best_score
                self._log_visualizations(trainer, pl_module)

    def _log_visualizations(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Delegate visualization to model-specific implementation."""
        if not hasattr(pl_module, "_log_visualizations"):
            logger.warning(
                "Model %s does not implement _log_visualizations",
                type(pl_module).__name__,
            )
            return
        try:
            if self.current_batch is None or self.current_outputs is None:
                logger.warning("No batch or outputs to visualize")
                return
            pl_module._log_visualizations(  # noqa: SLF001
                trainer,
                self.current_batch,
                self.current_outputs,
                self.max_samples,
            )
        except Exception:
            logger.exception("Error during visualization logging")

from lightning.pytorch.callbacks import Callback
from lightning.pytorch import Trainer, LightningModule

class OverrideEpochStepCallback(Callback):
    def __init__(self, override = True) -> None:
        super().__init__()
        self.override = override

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.override:
            self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.override:
            self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.override:
            self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(self, trainer: Trainer, pl_module: LightningModule):
        pl_module.log("step", int(trainer.current_epoch), sync_dist=True)
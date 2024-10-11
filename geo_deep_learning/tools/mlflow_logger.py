from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import MLFlowLogger

class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, MLFlowLogger):
            config_filepath = self.config.config[0]
            trainer.logger.experiment.log_artifact(local_path=config_filepath, 
                                                   artifact_path="config",
                                                   run_id=trainer.logger.run_id)

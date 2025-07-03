"""MLFlow logger."""

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import MLFlowLogger


class LoggerSaveConfigCallback(SaveConfigCallback):
    """Save config callback for MLFlow logger."""

    def save_config(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
        stage: str,  # noqa: ARG002
    ) -> None:
        """Save config."""
        if isinstance(trainer.logger, MLFlowLogger):
            config_filepath = self.config.config[0]
            trainer.logger.experiment.log_artifact(
                local_path=config_filepath,
                artifact_path="config",
                run_id=trainer.logger.run_id,
            )

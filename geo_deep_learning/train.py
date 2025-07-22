"""Train model with Lightning CLI."""

import logging
from typing import Any

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.cli import ArgsType, LightningCLI
from lightning.pytorch.loggers import MLFlowLogger

from configs import logging_config  # noqa: F401
from geo_deep_learning.tools.mlflow_logger import LoggerSaveConfigCallback

logger = logging.getLogger(__name__)


class TestMLFlowLogger(MLFlowLogger):
    """Custom MLFlowLogger that prevents hyperparameter logging during test."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize TestMLFlowLogger."""
        super().__init__(*args, **kwargs)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Override to prevent hyperparameter logging during test."""


class GeoDeepLearningCLI(LightningCLI):
    """Custom LightningCLI."""

    def after_fit(self) -> None:
        """Log test metrics."""
        if self.trainer.is_global_zero:
            test_dataloader = self.datamodule.test_dataloader()
            if test_dataloader is None:
                logger.warning("No test dataloader found.")
                return
            best_model_path = self.trainer.checkpoint_callback.best_model_path
            test_logger = TestMLFlowLogger(
                experiment_name=self.trainer.logger._experiment_name,  # noqa: SLF001
                run_name=self.trainer.logger._run_name,  # noqa: SLF001
                run_id=self.trainer.logger.run_id,
                save_dir=self.trainer.logger.save_dir,
            )

            test_trainer = Trainer(
                devices=1,
                accelerator="auto",
                strategy="auto",
                logger=test_logger,
            )
            best_model = self.model.__class__.load_from_checkpoint(
                best_model_path,
                weights_from_checkpoint_path=None,
                strict=True,
            )
            test_trainer.test(
                model=best_model,
                dataloaders=test_dataloader,
            )
            self.trainer.logger.log_hyperparams({"best_model_path": best_model_path})
            logger.info("Test metrics logged successfully to all loggers.")
        self.trainer.strategy.barrier()


def main(args: ArgsType = None) -> None:
    """Run the main training pipeline."""
    seed_everything(42, workers=True)
    cli = GeoDeepLearningCLI(
        save_config_callback=LoggerSaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        auto_configure_optimizers=False,
        args=args,
    )
    if cli.trainer.is_global_zero:
        logger.info("Done!")


if __name__ == "__main__":
    main()

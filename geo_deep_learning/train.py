"""Train model with Lightning CLI."""

import logging
from typing import Any

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.cli import ArgsType, LightningCLI
from lightning.pytorch.loggers import MLFlowLogger

from geo_deep_learning.config import logging_config  # noqa: F401
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
            
            # === DIAGNOSTIC: Log checkpoint information ===
            best_model_path = self.trainer.checkpoint_callback.best_model_path
            best_model_score = self.trainer.checkpoint_callback.best_model_score
            
            logger.info("=" * 80)
            logger.info("CHECKPOINT LOADING DIAGNOSTICS")
            logger.info("=" * 80)
            logger.info("Best model checkpoint path: %s", best_model_path)
            logger.info("Best model score (val_loss): %s", best_model_score)
            logger.info("Checkpoint callback monitor: %s", self.trainer.checkpoint_callback.monitor)
            logger.info("Checkpoint callback mode: %s", self.trainer.checkpoint_callback.mode)
            
            # Verify checkpoint exists
            from pathlib import Path
            if not Path(best_model_path).exists():
                logger.error("❌ CRITICAL: Best checkpoint file does not exist: %s", best_model_path)
                return
            else:
                logger.info("✓ Checkpoint file exists")
                
            # Load checkpoint and inspect
            import torch as checkpoint_torch
            checkpoint_data = checkpoint_torch.load(best_model_path, map_location="cpu")
            logger.info("Checkpoint epoch: %s", checkpoint_data.get("epoch", "N/A"))
            logger.info("Checkpoint global_step: %s", checkpoint_data.get("global_step", "N/A"))
            logger.info("Checkpoint keys: %s", list(checkpoint_data.keys()))
            
            # Check if state_dict has model weights
            if "state_dict" in checkpoint_data:
                state_dict_keys = list(checkpoint_data["state_dict"].keys())[:5]
                logger.info("Sample state_dict keys: %s", state_dict_keys)
            logger.info("=" * 80)
            
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
            
            # === DIAGNOSTIC: Log model loading ===
            logger.info("Loading best model from checkpoint...")
            best_model = self.model.__class__.load_from_checkpoint(
                best_model_path,
                weights_from_checkpoint_path=None,
                strict=True,
            )
            logger.info("✓ Model loaded successfully from checkpoint")
            logger.info("Model class: %s", best_model.__class__.__name__)
            logger.info("Model device: %s", next(best_model.parameters()).device)
            
            # === DIAGNOSTIC: Run test ===
            logger.info("Starting test with loaded model...")
            test_trainer.test(
                model=best_model,
                dataloaders=test_dataloader,
            )
            self.trainer.logger.log_hyperparams({
                "best_model_path": best_model_path,
                "best_model_score": float(best_model_score) if best_model_score is not None else None,
            })
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

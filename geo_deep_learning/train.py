from tools.mlflow_logger import LoggerSaveConfigCallback
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.cli import ArgsType, LightningCLI

class TestMLFlowLogger(MLFlowLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def log_hyperparams(self, params):
        # Override to prevent hyperparameter logging during test
        pass
class GeoDeepLearningCLI(LightningCLI):
    def before_fit(self):
        self.datamodule.prepare_data()
        self.datamodule.setup("fit")
        self.log_dataset_sizes()
    
    def after_fit(self):
        if self.trainer.is_global_zero:
            best_model_path = self.trainer.checkpoint_callback.best_model_path
            
            test_logger = TestMLFlowLogger(
                experiment_name=self.trainer.logger._experiment_name,
                run_name=self.trainer.logger._run_name,
                run_id=self.trainer.logger.run_id,
                save_dir=self.trainer.logger.save_dir
            )
            
            test_trainer = Trainer(devices=1, 
                                   accelerator="auto", 
                                   strategy="auto",
                                   logger=test_logger,
                                   )
            best_model = self.model.__class__.load_from_checkpoint(best_model_path,
                                                                   weights_from_checkpoint_path=None,
                                                                   strict=True)
            test_trainer.test(model=best_model, 
                              dataloaders=self.datamodule.test_dataloader())
            self.trainer.logger.log_hyperparams({"best_model_path": best_model_path})
            print("Test metrics logged successfully to all loggers.")
        self.trainer.strategy.barrier()
    
    def log_dataset_sizes(self):
        if self.trainer.is_global_zero:
            train_size = len(self.datamodule.train_dataloader().dataset)
            val_size = len(self.datamodule.val_dataloader().dataset)
            test_size = len(self.datamodule.test_dataloader().dataset)
            
            
            metrics = {
                "num_training_samples": train_size,
                "num_validation_samples": val_size,
                "num_test_samples": test_size
            }
            
            self.trainer.logger.log_metrics(metrics)
            
            print(f"Number of training samples: {train_size}")
            print(f"Number of validation samples: {val_size}")
            print(f"Number of test samples: {test_size}")


def main(args: ArgsType = None) -> None:
    cli = GeoDeepLearningCLI(save_config_callback=LoggerSaveConfigCallback,
                             save_config_kwargs={"overwrite": True},
                             parser_kwargs={"parser_mode": "omegaconf"},
                             auto_configure_optimizers=False,
                             args=args)
    if cli.trainer.is_global_zero:
        print("Done!")
    

if __name__ == "__main__":
    main()
    
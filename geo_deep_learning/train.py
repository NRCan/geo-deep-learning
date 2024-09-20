import torch
from lightning.pytorch import Trainer
from lightning.pytorch.cli import ArgsType, LightningCLI


class GeoDeepLearningCLI(LightningCLI):
    def before_fit(self):
        self.datamodule.prepare_data()
        self.datamodule.setup("fit")
        self.print_dataset_sizes()
    
    def after_fit(self):
        if self.trainer.is_global_zero:
            best_model_path = self.trainer.checkpoint_callback.best_model_path
            test_trainer = Trainer(devices=1, 
                                   accelerator="auto", 
                                   strategy="auto")
            best_model = self.model.__class__.load_from_checkpoint(best_model_path)
            test_trainer.test(model=best_model, dataloaders=self.datamodule.test_dataloader())
        self.trainer.strategy.barrier()
    
    def print_dataset_sizes(self):
        if self.trainer.is_global_zero:
            train_size = len(self.datamodule.train_dataloader().dataset)
            val_size = len(self.datamodule.val_dataloader().dataset)
            test_size = len(self.datamodule.test_dataloader().dataset)

            print(f"Number of training samples: {train_size}")
            print(f"Number of validation samples: {val_size}")
            print(f"Number of test samples: {test_size}")


def main(args: ArgsType = None) -> None:
    cli = GeoDeepLearningCLI(save_config_kwargs={"overwrite": True}, 
                             args=args)
    if cli.trainer.is_global_zero:
        print("Done!")
    

if __name__ == "__main__":
    main()
    
from lightning.pytorch.cli import ArgsType, LightningCLI
import datasets, datamodules, tasks_with_models

def main():
    LightningCLI(
        datasets=[datasets],
        datamodules=[datamodules],
        models=[tasks_with_models],
        task=tasks_with_models.SegmentationSegformer,
        args_type=ArgsType,
    )
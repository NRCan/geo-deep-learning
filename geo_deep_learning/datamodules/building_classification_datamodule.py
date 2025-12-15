"""Building Classification DataModule."""

from typing import Any

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from geo_deep_learning.datasets.building_classification_dataset import (
    BuildingClassificationDataset,
    BuildingClassificationOversamplingDataset,
    collate_fn_building_classification,
)


class BuildingClassificationDataModule(LightningDataModule):
    """Building Classification DataModule with support for instance-level classification."""

    def __init__(  # noqa: PLR0913
        self,
        batch_size: int = 16,
        num_workers: int = 8,
        data_type_max: int = 255,
        patch_size: tuple[int, int] = (512, 512),
        mean: list[float] | None = None,
        std: list[float] | None = None,
        csv_root_folder: str = "",
        patches_root_folder: str = "",
        target_class: int | None = None,
        min_instance_area: int = 100,
        use_oversampling: bool = False,
        oversampling_strategy: str = "balance",
    ) -> None:
        """
        Initialize Building Classification DataModule.
        
        Args:
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            data_type_max: Maximum value for data type (for normalization)
            patch_size: Size of image patches
            mean: Mean values for normalization
            std: Standard deviation values for normalization
            csv_root_folder: Root folder containing CSV files
            patches_root_folder: Root folder containing image patches
            target_class: If specified, only extract instances of this class (1-4)
            min_instance_area: Minimum area threshold for building instances
            use_oversampling: Whether to use oversampling for training data
            oversampling_strategy: Strategy for oversampling ('balance', 'max', 'min')
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.data_type_max = data_type_max
        self.csv_root_folder = csv_root_folder
        self.patches_root_folder = patches_root_folder
        self.target_class = target_class
        self.min_instance_area = min_instance_area
        self.use_oversampling = use_oversampling
        self.oversampling_strategy = oversampling_strategy
        
        self.norm_stats = {
            "mean": mean or [0.0, 0.0, 0.0],
            "std": std or [1.0, 1.0, 1.0],
        }

    def setup(self, stage: str | None = None) -> None:
        """Create datasets for training, validation, and testing."""
        if self.use_oversampling:
            self.train_dataset = BuildingClassificationOversamplingDataset(
                csv_root_folder=self.csv_root_folder,
                    patches_root_folder=self.patches_root_folder,
                    split="trn",
                    norm_stats=self.norm_stats,
                    target_class=self.target_class,
                    min_instance_area=self.min_instance_area,
                    oversampling_strategy=self.oversampling_strategy,
                )
        else:
            self.train_dataset = BuildingClassificationDataset(
                csv_root_folder=self.csv_root_folder,
                    patches_root_folder=self.patches_root_folder,
                    split="trn",
                    norm_stats=self.norm_stats,
                    target_class=self.target_class,
                    min_instance_area=self.min_instance_area,
                )
            
        # Validation dataset - no oversampling
        self.val_dataset = BuildingClassificationDataset(
                csv_root_folder=self.csv_root_folder,
                patches_root_folder=self.patches_root_folder,
                split="val",
                norm_stats=self.norm_stats,
                target_class=self.target_class,
                min_instance_area=self.min_instance_area,
            )
        
       
        self.test_dataset = BuildingClassificationDataset(
                csv_root_folder=self.csv_root_folder,
                patches_root_folder=self.patches_root_folder,
                split="tst",
                norm_stats=self.norm_stats,
                target_class=self.target_class,
                min_instance_area=self.min_instance_area,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Dataloader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
            shuffle=True,
            collate_fn=collate_fn_building_classification,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Dataloader for validation."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
            shuffle=False,
            collate_fn=collate_fn_building_classification,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Dataloader for testing."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
            shuffle=False,
            collate_fn=collate_fn_building_classification,
        )


if __name__ == "__main__":
    csv_root_folder = ""
    patches_root_folder = csv_root_folder
    datamodule = BuildingClassificationDataModule(
        csv_root_folder=csv_root_folder,
        patches_root_folder=patches_root_folder,
    )
    datamodule.setup()
    print(f"Train dataset size: {len(datamodule.train_dataset)}")
    print(f"Val dataset size: {len(datamodule.val_dataset)}")

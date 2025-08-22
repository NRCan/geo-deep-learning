"""CSVDataModule."""

from typing import Any

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from geo_deep_learning.datasets.csv_dataset import CSVDataset


class CSVDataModule(LightningDataModule):
    """CSV DataModule."""

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
    ) -> None:
        """Initialize CSVDataModule."""
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.data_type_max = data_type_max
        self.csv_root_folder = csv_root_folder
        self.patches_root_folder = patches_root_folder
        self.norm_stats = {
            "mean": mean or [0.0, 0.0, 0.0],
            "std": std or [1.0, 1.0, 1.0],
        }

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Create dataset."""
        self.train_dataset = CSVDataset(
            split="trn",
            norm_stats=self.norm_stats,
            csv_root_folder=self.csv_root_folder,
            patches_root_folder=self.patches_root_folder,
        )
        self.val_dataset = CSVDataset(
            split="val",
            norm_stats=self.norm_stats,
            csv_root_folder=self.csv_root_folder,
            patches_root_folder=self.patches_root_folder,
        )
        self.test_dataset = CSVDataset(
            split="tst",
            norm_stats=self.norm_stats,
            csv_root_folder=self.csv_root_folder,
            patches_root_folder=self.patches_root_folder,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Dataloader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Dataloader for validation."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Dataloader for testing."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            shuffle=False,
        )


if __name__ == "__main__":
    csv_root_folder = ""
    patches_root_folder = csv_root_folder
    dataset = CSVDataModule(csv_root_folder, patches_root_folder)
    # print(f"mean:{dataset.mean}, std:{dataset.std}")

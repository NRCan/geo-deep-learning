"""CSVDataModule."""

from typing import Any

import kornia as krn
import torch
from kornia.augmentation import AugmentationSequential
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from geo_deep_learning.datasets.csv_dataset import CSVDataset
from geo_deep_learning.tools.utils import normalization, standardization


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
        band_indices: list[int] | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize CSVDataModule."""
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.data_type_max = data_type_max
        self.mean = mean or [0.0, 0.0, 0.0]
        self.std = std or [1.0, 1.0, 1.0]
        self.band_indices = band_indices
        self.kwargs = kwargs

        if self.band_indices is not None:
            self.mean = [self.mean[i] for i in self.band_indices]
            self.std = [self.std[i] for i in self.band_indices]

        self.normalize = krn.augmentation.Normalize(
            mean=self.mean,
            std=self.std,
            p=1,
            keepdim=True,
        )
        random_resized_crop_zoom_in = krn.augmentation.RandomResizedCrop(
            size=self.patch_size,
            scale=(1.0, 2.0),
            p=0.5,
            align_corners=False,
            keepdim=True,
        )
        random_resized_crop_zoom_out = krn.augmentation.RandomResizedCrop(
            size=self.patch_size,
            scale=(0.5, 1.0),
            p=0.5,
            align_corners=False,
            keepdim=True,
        )

        self.transform = AugmentationSequential(
            krn.augmentation.RandomHorizontalFlip(p=0.5, keepdim=True),
            krn.augmentation.RandomVerticalFlip(p=0.5, keepdim=True),
            krn.augmentation.RandomRotation90(
                times=(1, 3),
                p=0.5,
                align_corners=True,
                keepdim=True,
            ),
            random_resized_crop_zoom_in,
            random_resized_crop_zoom_out,
            data_keys=None,
            random_apply=1,
        )

    def prepare_data(self) -> None:
        """Prepare data."""
        # download, enhance, tile, etc...

    def preprocess(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Preprocess sample."""
        sample["image"] = self._manage_bands(sample["image"])
        sample["image"] /= self.data_type_max
        sample["image"] = self.normalize(sample["image"])

        return sample

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Create dataset."""
        train_transform = Compose([self.transform, self.preprocess])
        test_transform = Compose([self.preprocess])

        self.train_dataset = CSVDataset(
            split="trn",
            transforms=train_transform,
            **self.kwargs,
        )
        self.val_dataset = CSVDataset(
            split="val",
            transforms=test_transform,
            **self.kwargs,
        )
        self.test_dataset = CSVDataset(
            split="tst",
            transforms=test_transform,
            **self.kwargs,
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

    def _manage_bands(self, image: torch.Tensor) -> torch.Tensor:
        """Select specific bands from the image tensor based on band indices."""
        if self.band_indices is not None:
            bands = image.size(0)
            if max(self.band_indices) >= bands:
                msg = (
                    f"Band index {max(self.band_indices)} "
                    f"is out of range for image with {bands} bands"
                )
                raise ValueError(
                    msg,
                )
            band_indices = torch.LongTensor(self.band_indices).to(image.device)
            return torch.index_select(image, dim=0, index=band_indices)
        return image

    def _model_script_preprocess(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Preprocess sample by normalizing and standardizing the image."""
        device = sample["image"].device
        mean_ = torch.tensor(self.mean, device=device).reshape(len(self.mean), 1)
        std_ = torch.tensor(self.std, device=device).reshape(len(self.std), 1)

        sample["image"] = normalization(sample["image"], 0, self.data_type_max)
        sample["image"] = standardization(sample["image"], mean_, std_)
        return sample


if __name__ == "__main__":
    csv_root_folder = ""
    patches_root_folder = csv_root_folder
    dataset = CSVDataModule(csv_root_folder, patches_root_folder)
    # print(f"mean:{dataset.mean}, std:{dataset.std}")

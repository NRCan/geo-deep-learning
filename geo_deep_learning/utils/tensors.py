"""Utility functions related to tensors."""

import logging

import torch

logger = logging.getLogger(__name__)


def normalization(
    input_tensor: torch.Tensor,
    image_min: int = 0,
    image_max: int = 255,
    norm_min: float = 0.0,
    norm_max: float = 1.0,
) -> torch.Tensor:
    """Normalize the input tensor."""
    input_shape = input_tensor.shape
    input_tensor = (norm_max - norm_min) * (input_tensor - image_min) / (
        image_max - image_min
    ) + norm_min
    return input_tensor.reshape(input_shape)


def standardization(
    input_tensor: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Standardize the input tensor."""
    input_shape = input_tensor.shape
    batch_size, channels = input_tensor.shape[:2]
    input_tensor = input_tensor.reshape(batch_size, channels, -1)
    input_tensor = (input_tensor - mean) / std
    return input_tensor.reshape(input_shape)


def denormalization(
    image: torch.Tensor,
    mean: torch.Tensor | float | None = None,
    std: torch.Tensor | float | None = None,
    data_type_max: int = 255,
) -> torch.Tensor:
    """Denormalize the input tensor."""
    if mean is not None and std is not None:
        if not torch.is_tensor(mean):
            mean = torch.tensor(mean, device=image.device)
        if not torch.is_tensor(std):
            std = torch.tensor(std, device=image.device)

        mean = mean.reshape(-1, 1, 1)
        std = std.reshape(-1, 1, 1)

        image = image * std + mean

    return (image * data_type_max).clamp(0, data_type_max).to(torch.uint8)


def manage_bands(
    image: torch.Tensor,
    band_indices: list[int] | None = None,
) -> torch.Tensor:
    """Select specific bands from the image tensor based on band indices."""
    if band_indices is not None:
        bands = image.size(0)
        if max(band_indices) >= bands:
            msg = (
                f"Band index {max(band_indices)} "
                f"is out of range for image with {bands} bands"
            )
            raise ValueError(
                msg,
            )
        band_indices = torch.LongTensor(band_indices).to(image.device)
        return torch.index_select(image, dim=0, index=band_indices)
    return image

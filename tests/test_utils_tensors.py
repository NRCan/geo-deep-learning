"""Unit tests for utils/tensors.py functions."""

import pytest
import torch

from geo_deep_learning.utils.tensors import (
    denormalization,
    manage_bands,
    normalization,
    standardization,
)


def test_normalization_simple_range() -> None:
    """Test normalization from 0-255 to 0-1 range."""
    tensor = torch.tensor([[0.0, 127.5, 255.0]])
    normed = normalization(
        tensor,
        image_min=0,
        image_max=255,
        norm_min=0.0,
        norm_max=1.0,
    )
    expected = torch.tensor([[0.0, 0.5, 1.0]])
    assert torch.allclose(normed, expected, atol=1e-6)


def test_normalization_custom_range() -> None:
    """Test normalization to a custom output range."""
    tensor = torch.tensor([0.0, 255.0])
    normed = normalization(
        tensor,
        image_min=0,
        image_max=255,
        norm_min=-1.0,
        norm_max=1.0,
    )
    expected = torch.tensor([-1.0, 1.0])
    assert torch.allclose(normed, expected, atol=1e-6)


def test_standardization_basic() -> None:
    """Test z-score standardization on a small tensor."""
    # shape: (batch, channels, height, width)
    tensor = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # 1x1x2x2
    mean = torch.tensor([2.5])
    std = torch.tensor([1.118034])  # sqrt(var=1.25)
    standardized = standardization(tensor, mean, std)
    expected = (tensor - mean.view(1, 1, 1, 1)) / std.view(1, 1, 1, 1)
    assert torch.allclose(standardized, expected, atol=1e-6)


def test_denormalization_with_mean_std() -> None:
    """Test denormalization with given mean and std."""
    img = torch.tensor([[[(0.5), (0.8)]]])  # shape 1x1x2
    mean = torch.tensor([0.2])
    std = torch.tensor([0.4])
    denorm = denormalization(img, mean, std, data_type_max=255)

    # Compute manually:
    expected = ((img * std) + mean) * 255
    expected = expected.clamp(0, 255).to(torch.uint8)
    assert torch.equal(denorm, expected)


def test_denormalization_without_mean_std() -> None:
    """Test denormalization when mean/std are None."""
    img = torch.tensor([[[(0.5), (1.0)]]])
    result = denormalization(img, mean=None, std=None, data_type_max=255)
    expected = (img * 255).clamp(0, 255).to(torch.uint8)
    assert torch.equal(result, expected)


def test_manage_bands_select_subset() -> None:
    """Test selecting specific bands by index."""
    img = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)  # 3 bands
    subset = manage_bands(img, [0, 2])
    expected = torch.stack((img[0], img[2]))
    assert torch.equal(subset, expected)


def test_manage_bands_out_of_range() -> None:
    """Test that invalid band indices raise ValueError."""
    img = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    with pytest.raises(ValueError, match="Band index 2 is out of range"):
        manage_bands(img, [0, 2])


def test_manage_bands_none_returns_same() -> None:
    """Test that None for band_indices returns the original tensor."""
    img = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
    result = manage_bands(img, None)
    assert torch.equal(result, img)

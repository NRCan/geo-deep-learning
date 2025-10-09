"""Unit tests for utils/models.py functions."""

from pathlib import Path

import pytest
import torch

from geo_deep_learning.utils.models import load_weights_from_checkpoint


class DummyModel(torch.nn.Module):
    """Simple test model with named submodules for partial loading tests."""

    def __init__(self) -> None:
        """Initialize dummy encoder and decoder layers."""
        super().__init__()
        self.encoder = torch.nn.Linear(10, 5)
        self.decoder = torch.nn.Linear(5, 2)


@pytest.fixture
def dummy_checkpoint(tmp_path: Path) -> Path:
    """Create a dummy checkpoint file for testing."""
    model = DummyModel()
    state_dict = model.state_dict()

    # Save a fake checkpoint containing a state_dict key
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save({"state_dict": state_dict}, checkpoint_path)
    return checkpoint_path


def test_full_checkpoint_load(dummy_checkpoint: Path) -> None:
    """Test full model weight loading."""
    model = DummyModel()
    result = load_weights_from_checkpoint(model, str(dummy_checkpoint))

    # Full load returns None (not a tuple)
    assert result is None

    # Parameters should now match the checkpoint
    ckpt = torch.load(dummy_checkpoint)["state_dict"]
    for name, param in model.state_dict().items():
        assert torch.allclose(param, ckpt[name])


def test_partial_load_as_list(dummy_checkpoint: Path) -> None:
    """Test selective part loading using a list of module prefixes."""
    model = DummyModel()
    result = load_weights_from_checkpoint(
        model,
        str(dummy_checkpoint),
        load_parts=["encoder"],
    )

    # Partial load returns a named tuple-like object
    assert hasattr(result, "missing_keys")
    assert hasattr(result, "unexpected_keys")

    # Encoder weights loaded, decoder weights should be missing
    missing = result.missing_keys
    assert any("decoder" in k for k in missing)
    assert not any("encoder" in k for k in missing)


def test_partial_load_as_string(dummy_checkpoint: Path) -> None:
    """Test selective part loading when `load_parts` is a string."""
    model = DummyModel()
    result = load_weights_from_checkpoint(
        model,
        str(dummy_checkpoint),
        load_parts="decoder",
    )

    assert hasattr(result, "missing_keys")
    assert hasattr(result, "unexpected_keys")

    # Decoder loaded successfully
    missing = result.missing_keys
    assert any("encoder" in k for k in missing)
    assert not any("decoder" in k for k in missing)


def test_missing_state_dict_key(tmp_path: Path) -> None:
    """Test checkpoint without 'state_dict' key (raw state_dict only)."""
    model = DummyModel()
    state_dict = model.state_dict()
    ckpt_path = tmp_path / "raw_state_dict.pth"
    torch.save(state_dict, ckpt_path)

    result = load_weights_from_checkpoint(model, str(ckpt_path))
    assert result is None


def test_invalid_checkpoint_path(tmp_path: Path) -> None:
    """Test that invalid checkpoint path raises FileNotFoundError."""
    model = DummyModel()
    missing_path = tmp_path / "nonexistent.pth"

    with pytest.raises(FileNotFoundError):
        load_weights_from_checkpoint(model, str(missing_path))

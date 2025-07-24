"""Utility functions."""

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


def load_weights_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    load_parts: str | list[str] | None = None,
    map_location: torch.device | None = None,
) -> tuple[list[str], list[str]] | None:
    """
    Load weights from a checkpoint into a model.

    Args:
        model: The model to load weights into
        checkpoint_path: Path to the checkpoint file
        load_parts: List of model parts to load (e.g., ["encoder", "neck"])
        map_location: Optional device mapping for loading the checkpoint

    Returns:
        Tuple of (missing_keys, unexpected_keys) if selective loading,
        or None if full loading

    """
    logger.info("Loading weights from checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
    if load_parts is not None:
        if isinstance(load_parts, str):
            load_parts = [load_parts]

        filtered_state_dict = {}
        loaded_parts_keys = {part: [] for part in load_parts}
        for k, v in state_dict.items():
            for part in load_parts:
                if k.startswith(f"{part}."):
                    filtered_state_dict[k] = v
                    loaded_parts_keys[part].append(k)

        result = model.load_state_dict(filtered_state_dict, strict=False)
        logger.info("Loaded weights for parts: %s", load_parts)
        for part in load_parts:
            num_keys = len(loaded_parts_keys[part])
            if num_keys > 0:
                logger.info("  - %s: %s parameters loaded", part, num_keys)
                examples = loaded_parts_keys[part][:3]
                if examples:
                    logger.info("    Examples: %s", ", ".join(examples))
            else:
                logger.info(
                    "  - %s: NO PARAMETERS FOUND - check if this part exists",
                    part,
                )

        logger.info("Missing keys: %s", len(result.missing_keys))
        logger.info("Unexpected keys: %s", len(result.unexpected_keys))
        return result

    model.load_state_dict(state_dict)
    return None

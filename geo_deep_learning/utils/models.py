"""Utility functions related to models."""

import logging
import re
import torch
from typing import Any, Iterable
logger = logging.getLogger(__name__)


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

def _supports_weights_only() -> bool:
    return "weights_only" in torch.load.__code__.co_varnames  # type: ignore[attr-defined]


def _load_checkpoint_file(
    checkpoint_path: str,
    *,
    map_location: torch.device | str | None,
) -> dict[str, Any] | list | torch.Tensor:
    # _register_safe_globals()
    load_kwargs = {"map_location": map_location or "cpu"}

    if _supports_weights_only():
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=True, **load_kwargs)
            logger.info(
                "Loaded checkpoint with weights_only=True from %s",
                checkpoint_path,
            )
            return checkpoint
        except Exception as exc:
            logger.warning(
                "Failed to load %s with weights_only=True: %s. Falling back to weights_only=False.",
                checkpoint_path,
                exc,
            )

    checkpoint = torch.load(checkpoint_path, weights_only=False, **load_kwargs)
    logger.info(
        "Loaded checkpoint with weights_only=False from %s",
        checkpoint_path,
    )
    return checkpoint


def _remove_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key[len(prefix) :]: value for key, value in state_dict.items()}


def _normalise_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state_dict = _remove_prefix(state_dict, "model.")
    state_dict = _remove_prefix(state_dict, "module.")
    return state_dict


def _process_swin_unet_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename MAE-style keys to match the Swin UNet implementation."""
    processed: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        # Replace "backbone.backbone" with "encoder"
        new_key = key.replace("backbone.backbone.", "")
        # Replace "features" with "layers"
        new_key = new_key.replace("features", "layers")
        # Add ".blocks" between two numbers
        new_key = re.sub(r"layers.(\d+).(\d+)", r"layers.\1.blocks.\2", new_key)
        # Handle MLP layers
        new_key = re.sub(r"layers.(\d+).blocks.(\d+).mlp.0.weight", r"layers.\1.blocks.\2.mlp.fc1.weight", new_key)
        new_key = re.sub(r"layers.(\d+).blocks.(\d+).mlp.3.weight", r"layers.\1.blocks.\2.mlp.fc2.weight", new_key)
        new_key = re.sub(r"layers.(\d+).blocks.(\d+).mlp.0.bias", r"layers.\1.blocks.\2.mlp.fc1.bias", new_key)
        new_key = re.sub(r"layers.(\d+).blocks.(\d+).mlp.3.bias", r"layers.\1.blocks.\2.mlp.fc2.bias", new_key)
        processed[new_key] = value
    return processed


def _filter_state_dict_for_model(
    state_dict: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    filtered: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered[key] = value
    return filtered


def _filter_state_dict_by_parts(
    state_dict: dict[str, torch.Tensor],
    load_parts: Iterable[str],
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    filtered_state_dict: dict[str, torch.Tensor] = {}
    loaded_parts_keys: dict[str, list[str]] = {part: [] for part in load_parts}

    for key, value in state_dict.items():
        for part in load_parts:
            prefix = f"{part}."
            if key.startswith(prefix):
                filtered_state_dict[key] = value
                loaded_parts_keys[part].append(key)

    return filtered_state_dict, loaded_parts_keys

def freeze_encoder(model: torch.nn.Module) -> None:
    """Freeze encoder layers for finetuning"""
    modules_to_freeze = [
        model.swin_unet.patch_embed,
        model.swin_unet.layers,
        model.swin_unet.norm,
    ]
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False

    model.swin_unet.patch_embed.eval()
    model.swin_unet.layers.eval()
    model.swin_unet.norm.eval()

    num_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"Rank {get_rank()}: Frozen {num_frozen}/{num_total} parameters ({100 * num_frozen / num_total:.2f}%)")


def swinunet_load_weights_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    load_parts: str | list[str] | None = None,
    map_location: torch.device | str | None = None,
    freeze_encoder: bool = False,
) -> tuple[list[str], list[str]] | None:
    """
    Custom loader tailored for Swin UNet checkpoints (including MAE-derived weights).
    """
    logger.info("Loading SwinUNet weights from checkpoint: %s", checkpoint_path)
    checkpoint = _load_checkpoint_file(checkpoint_path, map_location=map_location)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    else:
        state_dict = checkpoint

    state_dict = _normalise_state_dict(state_dict)
    state_dict = _process_swin_unet_keys(state_dict)
    model_state = model.encoder.state_dict()
    state_dict = _filter_state_dict_for_model(state_dict, model_state)
    
    if load_parts is not None:
        if isinstance(load_parts, str):
            load_parts = [load_parts]
        
        state_dict, loaded_parts_keys = _filter_state_dict_by_parts(state_dict, load_parts)
        result = model.load_state_dict(state_dict, strict=False)
        freeze_encoder(model)
        logger.info("Loaded weights for parts: %s", list(load_parts))
        for part in load_parts:
            keys = loaded_parts_keys.get(part, [])
            if keys:
                logger.info(
                    "  - %s: loaded %s parameters (examples: %s)",
                    part,
                    len(keys),
                    ", ".join(keys[:3]),
                )
            else:
                logger.info("  - %s: no parameters matched", part)

        logger.info("Missing keys: %s", len(result.missing_keys))
        logger.info("Unexpected keys: %s", len(result.unexpected_keys))
        return list(result.missing_keys), list(result.unexpected_keys)

    result = model.encoder.load_state_dict(state_dict, strict=False)
    if isinstance(result, tuple):
        missing_keys, unexpected_keys = result
        if missing_keys or unexpected_keys:
            logger.info("Missing keys: %s", missing_keys)
            logger.info("Unexpected keys: %s", unexpected_keys)
        else:
            logger.info("Checkpoint loaded without missing or unexpected keys.")
    else:
        logger.info("Checkpoint loaded successfully.")

    if freeze_encoder:
        swinunet_freeze_encoder(model)

    return None


def swinunet_freeze_encoder(model: torch.nn.Module) -> None:
    """Freeze encoder layers for finetuning scenarios."""
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        logger.warning("Encoder module not found on model; skipping freeze.")
        return

    modules_to_freeze = [
        getattr(encoder, "patch_embed", None),
        getattr(encoder, "layers", None),
        getattr(encoder, "norm", None),
    ]
    modules_to_freeze = [module for module in modules_to_freeze if module is not None]

    if not modules_to_freeze:
        logger.warning("No encoder submodules found to freeze.")
        return

    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = 0

    for module in modules_to_freeze:
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_params += param.numel()
        module.eval()

    if frozen_params > 0:
        percent = (frozen_params / total_params) * 100
        logger.info(
            "Frozen encoder parameters: %s/%s (%.2f%%).",
            frozen_params,
            total_params,
            percent,
        )
    else:
        logger.info("Encoder was already frozen; no parameters updated.")


def swinclassifier_load_weights_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    load_parts: str | list[str] | None = None,
    map_location: torch.device | str | None = None,
    freeze_encoder: bool = False,
) -> tuple[list[str], list[str]] | None:
    """
    Custom loader for Swin Building Classifier checkpoints (including MAE-derived weights).
    
    This function loads pretrained weights into the RGB and mask encoders of a 
    SwinBuildingClassifier model.
    
    Args:
        model: The SwinBuildingClassifier model
        checkpoint_path: Path to the checkpoint file
        load_parts: Parts to load (e.g., ["rgb_encoder", "mask_encoder"])
        map_location: Device mapping for loading
        freeze_encoder: Whether to freeze the encoder layers after loading
        
    Returns:
        Tuple of (missing_keys, unexpected_keys) or None
    """
    logger.info("Loading Swin classifier weights from checkpoint: %s", checkpoint_path)
    checkpoint = _load_checkpoint_file(checkpoint_path, map_location=map_location)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    else:
        state_dict = checkpoint

    state_dict = _normalise_state_dict(state_dict)
    state_dict = _process_swin_unet_keys(state_dict)
    
    # Load into both RGB and mask encoders
    if load_parts is None:
        load_parts = ["rgb_encoder", "mask_encoder"]
    elif isinstance(load_parts, str):
        load_parts = [load_parts]
    
    all_missing_keys = []
    all_unexpected_keys = []
    
    for part in load_parts:
        encoder = getattr(model, part, None)
        if encoder is None:
            logger.warning("Part %s not found on model; skipping.", part)
            continue
            
        encoder_state = encoder.state_dict()
        filtered_state = _filter_state_dict_for_model(state_dict, encoder_state)
        
        if filtered_state:
            result = encoder.load_state_dict(filtered_state, strict=False)
            logger.info(
                "Loaded %s parameters into %s (examples: %s)",
                len(filtered_state),
                part,
                ", ".join(list(filtered_state.keys())[:3]),
            )
            all_missing_keys.extend(result.missing_keys)
            all_unexpected_keys.extend(result.unexpected_keys)
        else:
            logger.warning("No matching parameters found for %s", part)
    
    if all_missing_keys:
        logger.info("Total missing keys: %s", len(all_missing_keys))
    if all_unexpected_keys:
        logger.info("Total unexpected keys: %s", len(all_unexpected_keys))
    
    if freeze_encoder:
        swin_freeze_encoders(model, load_parts)
    
    return all_missing_keys, all_unexpected_keys


def swin_freeze_encoders(model: torch.nn.Module, encoder_names: list[str]) -> None:
    """Freeze encoder layers in the Swin Building Classifier."""
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = 0
    
    for encoder_name in encoder_names:
        encoder = getattr(model, encoder_name, None)
        if encoder is None:
            logger.warning("Encoder %s not found; skipping freeze.", encoder_name)
            continue
        
        for param in encoder.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_params += param.numel()
        encoder.eval()
        logger.info("Froze encoder: %s", encoder_name)
    
    if frozen_params > 0:
        percent = (frozen_params / total_params) * 100
        logger.info(
            "Frozen encoder parameters: %s/%s (%.2f%%).",
            frozen_params,
            total_params,
            percent,
        )
    else:
        logger.info("Encoders were already frozen; no parameters updated.")


def swinmae_load_weights_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    load_parts: str | list[str] | None = None,
    map_location: torch.device | str | None = None,
) -> tuple[list[str], list[str]] | None:
    """
    Custom loader tailored for Swin MAE checkpoints.
    
    Args:
        model: The Swin MAE model
        checkpoint_path: Path to the checkpoint file
        load_parts: Parts to load (e.g., ["encoder", "decoder", "neck"])
        map_location: Device mapping for loading
        
    Returns:
        Tuple of (missing_keys, unexpected_keys) or None
    """
    logger.info("Loading Swin MAE weights from checkpoint: %s", checkpoint_path)
    checkpoint = _load_checkpoint_file(checkpoint_path, map_location=map_location)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    else:
        state_dict = checkpoint

    state_dict = _normalise_state_dict(state_dict)
    
    # Load into encoder, decoder, and neck
    if load_parts is None:
        load_parts = ["encoder", "decoder", "neck"]
    elif isinstance(load_parts, str):
        load_parts = [load_parts]
    
    all_missing_keys = []
    all_unexpected_keys = []
    
    for part in load_parts:
        part_module = getattr(model, part, None)
        if part_module is None:
            logger.warning("Part %s not found on model; skipping.", part)
            continue
        
        # Transform checkpoint keys to match model keys based on part
        transformed_state = dict(state_dict)
        
        if part == "decoder":
            # Checkpoint has 'layers_up', model expects 'layers'
            transformed_state = {}
            for key, value in state_dict.items():
                if 'layers_up' in key:
                    new_key = key.replace('layers_up', 'layers')
                    transformed_state[new_key] = value
                elif 'norm' in key:
                    new_key = key.replace('norm_up', 'norm')
                    transformed_state[new_key] = value
                elif 'decoder_pred' in key:
                    new_key = key.replace('decoder_pred', 'projection')
                    transformed_state[new_key] = value

        elif part == "neck":
            # Checkpoint has 'first_patch_expanding', model expects 'patch_expanding'
            transformed_state = {}
            for key, value in state_dict.items():
                if 'first_patch_expanding' in key:
                    new_key = key.replace('first_patch_expanding', 'patch_expanding')
                    transformed_state[new_key] = value
        
        # Filter to only include keys that match the model structure
        part_state = part_module.state_dict()
        filtered_state = _filter_state_dict_for_model(transformed_state, part_state)
        
        if filtered_state:
            result = part_module.load_state_dict(filtered_state, strict=False)
            logger.info(
                "Loaded %s parameters into %s (examples: %s)",
                len(filtered_state),
                part,
                ", ".join(list(filtered_state.keys())[:3]),
            )
            all_missing_keys.extend(result.missing_keys)
            all_unexpected_keys.extend(result.unexpected_keys)
        else:
            logger.warning("No matching parameters found for %s", part)
    
    if all_missing_keys:
        logger.info("Total missing keys: %s", len(all_missing_keys))
        logger.info("Missing keys examples: %s", all_missing_keys[:5])
    if all_unexpected_keys:
        logger.info("Total unexpected keys: %s", len(all_unexpected_keys))
        logger.info("Unexpected keys examples: %s", all_unexpected_keys[:5])

    return all_missing_keys, all_unexpected_keys

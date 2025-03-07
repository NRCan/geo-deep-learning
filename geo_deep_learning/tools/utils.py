import torch
from typing import Optional, Union, List
def normalization(input_tensor: torch.Tensor, 
                  image_min: int = 0, 
                  image_max: int = 255, 
                  norm_min: float = 0.0, 
                  norm_max: float = 1.0):
    input_shape = input_tensor.shape
    input_tensor = (norm_max - norm_min) * (input_tensor - image_min) / (image_max - image_min) + norm_min
    input_tensor = input_tensor.reshape(input_shape)
    return input_tensor

def standardization(input_tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    input_shape = input_tensor.shape
    B, C = input_tensor.shape[:2]
    input_tensor = input_tensor.reshape(B, C, -1)
    input_tensor = (input_tensor - mean) / std
    input_tensor = input_tensor.reshape(input_shape)
    return input_tensor

def denormalization(image: torch.Tensor, 
                    mean: torch.Tensor | float | None = None, 
                    std: torch.Tensor | float | None = None, 
                    data_type_max: int = 255):
    if mean is not None and std is not None:
        if not torch.is_tensor(mean):
            mean = torch.tensor(mean, device=image.device)
        if not torch.is_tensor(std):
            std = torch.tensor(std, device=image.device)
            
        mean = mean.reshape(-1, 1, 1)
        std = std.reshape(-1, 1, 1)
        
        image = image * std + mean
    
    image = (image * data_type_max).clamp(0, data_type_max).to(torch.uint8)
    
    return image

def load_weights_from_checkpoint(model, 
                                 checkpoint_path, 
                                 load_parts: Optional[Union[str, List[str]]] = None, 
                                 map_location: Optional[torch.device] = None):
    """
    Load weights from a checkpoint into a model.
    
    Args:
        model: The model to load weights into
        checkpoint_path: Path to the checkpoint file
        load_parts: List of model parts to load (e.g., ["encoder", "neck"]) or None to load all
        map_location: Optional device mapping for loading the checkpoint
        
    Returns:
        Tuple of (missing_keys, unexpected_keys) if selective loading,
        or None if full loading
    """
    print(f"Loading weights from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    if load_parts is not None:
        if isinstance(load_parts, str):
            load_parts = [load_parts]
        
        # Filter the state dict based on the specified parts to load
        filtered_state_dict = {}
        loaded_parts_keys = {part: [] for part in load_parts}
        for k, v in state_dict.items():
            # Check if any of the specified parts match this key
            for part in load_parts:
                if f"model.{part}" in k:
                    filtered_state_dict[k] = v
                    loaded_parts_keys[part].append(k)
        
        # Load the filtered state dict
        result = model.load_state_dict(filtered_state_dict, strict=False)
        print(f"\nLoaded weights for parts: {load_parts}")
        for part in load_parts:
            num_keys = len(loaded_parts_keys[part])
            if num_keys > 0:
                print(f"  - {part}: {num_keys} parameters loaded")
                # Print a few examples of loaded keys (up to 3)
                examples = loaded_parts_keys[part][:3]
                if examples:
                    print(f"    Examples: {', '.join(examples)}")
            else:
                print(f"  - {part}: NO PARAMETERS FOUND - check if this part exists in the checkpoint")
        
        print(f"\nMissing keys: {len(result.missing_keys)}")
        print(f"Unexpected keys: {len(result.unexpected_keys)}")
        return result
    else:
        # Load the full state dict
        model.load_state_dict(state_dict)
        return None
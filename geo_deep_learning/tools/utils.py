import torch

def normalization(input_tensor, min_val, max_val):
    input_shape = input_tensor.shape
    input_tensor = (max_val - min_val) * (input_tensor - min_val) / (max_val - min_val) + min_val
    input_tensor = input_tensor.reshape(input_shape)
    return input_tensor

def standardization(input_tensor, mean, std):
    input_shape = input_tensor.shape
    B, C = input_tensor.shape[:2]
    input_tensor = input_tensor.reshape(B, C, -1)
    input_tensor = (input_tensor - mean) / std
    input_tensor = input_tensor.reshape(input_shape)
    return input_tensor

def denormalization(image, mean, std, data_type_max):
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
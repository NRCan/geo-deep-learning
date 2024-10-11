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
    
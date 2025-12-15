"""
MLP Encoder for Bounding Box Features
Encodes geometric bounding box information using multi-layer perceptron
"""

import torch
import torch.nn as nn


class BBoxMLPEncoder(nn.Module):
    """
    MLP to encode bounding box information
    
    Takes geometric bounding box features and encodes them into
    a higher-dimensional representation suitable for fusion with
    visual features.
    
    Input features: [x1, y1, x2, y2, width, height, area, center_x, center_y]
    """

    def __init__(self, input_dim=9, hidden_dim=128, output_dim=256):
        """
        Args:
            input_dim: Dimension of input bbox features (default: 9)
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, bbox_features):
        """
        Encode bounding box features
        
        Args:
            bbox_features: (B, input_dim) tensor containing normalized bbox features
                          [x1, y1, x2, y2, width, height, area, center_x, center_y]
        
        Returns:
            encoded_features: (B, output_dim) encoded bbox representation
        """
        return self.mlp(bbox_features)
    
    @staticmethod
    def extract_bbox_features(bbox, input_size):
        """
        Extract and normalize features from bounding box coordinates
        
        Args:
            bbox: (x1, y1, x2, y2) tuple, list, or tensor
            input_size: Image size for normalization
        
        Returns:
            bbox_features: (9,) tensor with normalized features
                          [x1, y1, x2, y2, width, height, area, center_x, center_y]
        """
        # Handle different bbox formats
        if isinstance(bbox, (list, tuple)):
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
            else:
                raise ValueError(f"Expected bbox with 4 values, got {len(bbox)}: {bbox}")
        elif isinstance(bbox, torch.Tensor):
            if bbox.numel() == 4:
                x1, y1, x2, y2 = bbox.tolist()
            else:
                raise ValueError(f"Expected bbox tensor with 4 values, got {bbox.numel()}: {bbox}")
        else:
            raise ValueError(f"Unexpected bbox type: {type(bbox)}, value: {bbox}")
        
        # Calculate geometric features
        width = x2 - x1
        height = y2 - y1
        area = width * height
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        # Normalize coordinates by image size
        x1_norm = x1 / input_size[0]
        y1_norm = y1 / input_size[0]
        x2_norm = x2 / input_size[0]
        y2_norm = y2 / input_size[0]
        width_norm = width / input_size[0]
        height_norm = height / input_size[1]
        area_norm = area / (input_size[0] * input_size[1])
        center_x_norm = center_x / input_size[0]
        center_y_norm = center_y / input_size[0]
        
        return torch.tensor(
            [x1_norm, y1_norm, x2_norm, y2_norm, width_norm, 
             height_norm, area_norm, center_x_norm, center_y_norm],
            dtype=torch.float32
        )


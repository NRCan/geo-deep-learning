"""
Classification Head
Multi-layer perceptron head for classification tasks
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class ClassificationHead(nn.Module):
    """
    Classification head with configurable hidden layers
    
    Takes concatenated features and produces class logits through
    a multi-layer perceptron with dropout regularization.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim=512,
        dropout=0.5
    ):
        """
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights for Linear layers"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input features of shape (B, input_dim)
        
        Returns:
            logits: Class logits of shape (B, num_classes)
        """
        return self.head(x)


class MultiScaleClassificationHead(nn.Module):
    """
    Multi-scale classification head with deeper architecture
    
    Suitable for more complex classification tasks with multiple
    hidden layers and batch normalization.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dims=[512, 256],
        dropout=0.5
    ):
        """
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.head = nn.Sequential(*layers)
       
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input features of shape (B, input_dim)
        
        Returns:
            logits: Class logits of shape (B, num_classes)
        """
        return self.head(x)


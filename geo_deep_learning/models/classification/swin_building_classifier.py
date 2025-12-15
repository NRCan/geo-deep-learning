"""
Swin Building Classifier
Complete model integrating Swin encoder, MLP encoder, and classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from timm.models.layers import trunc_normal_

from geo_deep_learning.models.encoders.swin_encoder import SwinEncoder
from geo_deep_learning.models.encoders.MLP_encoder import BBoxMLPEncoder
from geo_deep_learning.models.heads.classification_head import ClassificationHead


class SwinBuildingClassifier(nn.Module):
    """
    Building Classifier for instance-level classification
    
    This model:
    1. Takes RGB image and binary mask as SEPARATE inputs
    2. Processes RGB and mask independently through separate encoders
    3. Fuses RGB and mask features
    4. Encodes bbox information using MLP
    5. Combines fused image features + bbox features for classification
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.model.num_classes
        self.input_size = cfg.model.input_size
        
        # Separate Swin Transformer Encoder for RGB (3 channels)
        self.rgb_encoder = SwinEncoder(
            input_size=cfg.model.input_size,
            patch_size=cfg.model.patch_size,
            in_chans=cfg.model.in_chans,  # RGB only
            embed_dim=cfg.variant.embed_dim,
            depths=cfg.variant.depths,
            num_heads=cfg.variant.num_heads,
            window_size=cfg.model.window_size,
            mlp_ratio=cfg.model.mlp_ratio,
            qkv_bias=cfg.model.qkv_bias,
            qk_scale=cfg.model.qk_scale,
            drop_rate=cfg.model.drop_rate,
            attn_drop_rate=cfg.model.attn_drop_rate,
            drop_path_rate=cfg.variant.drop_path_rate,
            ape=cfg.model.ape,
            patch_norm=cfg.model.patch_norm,
            use_checkpoint=cfg.model.use_checkpoint
        )
        
        # Separate encoder for binary mask (1 channel)
        # Use smaller dimension for mask since it's simpler
        self.mask_encoder = SwinEncoder(
            input_size=cfg.model.input_size,
            patch_size=cfg.model.patch_size,
            in_chans=1,  # Binary mask only
            embed_dim=cfg.variant.embed_dim // 2,  # Smaller dimension for mask
            depths=cfg.variant.depths,
            num_heads=[max(1, h // 2) for h in cfg.variant.num_heads],  # Fewer heads for mask
            window_size=cfg.model.window_size,
            mlp_ratio=cfg.model.mlp_ratio,
            qkv_bias=cfg.model.qkv_bias,
            qk_scale=cfg.model.qk_scale,
            drop_rate=cfg.model.drop_rate,
            attn_drop_rate=cfg.model.attn_drop_rate,
            drop_path_rate=cfg.variant.drop_path_rate,
            ape=cfg.model.ape,
            patch_norm=cfg.model.patch_norm,
            use_checkpoint=cfg.model.use_checkpoint
        )
        
        # Get the feature dimensions
        self.rgb_feature_dim = cfg.variant.embed_dim * (2 ** (len(cfg.variant.depths) - 1))
        self.mask_feature_dim = (cfg.variant.embed_dim // 2) * (2 ** (len(cfg.variant.depths) - 1))
        
        # Fusion layer to combine RGB and mask features
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.rgb_feature_dim + self.mask_feature_dim, self.rgb_feature_dim),
            nn.LayerNorm(self.rgb_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.model.drop_rate)
        )
        
        # BBox MLP Encoder
        self.bbox_encoder = BBoxMLPEncoder(
            input_dim=9,  # [x1, y1, x2, y2, width, height, area, center_x, center_y]
            hidden_dim=cfg.model.get('bbox_mlp_hidden_dim', 128),
            output_dim=cfg.model.get('bbox_mlp_output_dim', 256)
        )
        
        # Combined feature dimension: fused image features + bbox features
        self.total_feature_dim = self.rgb_feature_dim + self.bbox_encoder.mlp[-2].out_features
        
        # Classification Head
        self.classifier = ClassificationHead(
            input_dim=self.total_feature_dim,
            num_classes=self.num_classes,
            hidden_dim=cfg.model.get('classifier_hidden_dim', 512),
            dropout=cfg.model.get('classifier_dropout', 0.5)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize model weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, rgb_image, binary_mask, bbox_list):
        """
        Forward pass for building classification with separate RGB and mask processing
        
        Args:
            rgb_image (torch.Tensor): RGB image tensor of shape (B, 3, H, W)
            binary_mask (torch.Tensor): Binary mask tensor of shape (B, 1, H, W)
            bbox_list (list): List of bbox tuples for each sample in batch
                             Each tuple contains (x1, y1, x2, y2)
        
        Returns:
            torch.Tensor: Classification logits of shape (B, num_classes)
        """
        batch_size = rgb_image.size(0)
        
        # Process RGB image through RGB encoder
        rgb_features, rgb_downsample = self.rgb_encoder(rgb_image)
        
        # Process binary mask through mask encoder
        mask_features, mask_downsample = self.mask_encoder(binary_mask)
        
        # Process RGB features
        B_rgb, N_rgb, C_rgb = rgb_features.shape
        H_rgb = W_rgb = int(N_rgb ** 0.5)
        rgb_features_spatial = rgb_features.transpose(1, 2).reshape(B_rgb, C_rgb, H_rgb, W_rgb)
        
        # Process mask features
        B_mask, N_mask, C_mask = mask_features.shape
        H_mask = W_mask = int(N_mask ** 0.5)
        mask_features_spatial = mask_features.transpose(1, 2).reshape(B_mask, C_mask, H_mask, W_mask)
        
        # Global average pooling for both
        rgb_features_pooled = F.adaptive_avg_pool2d(rgb_features_spatial, (1, 1)).flatten(1)
        mask_features_pooled = F.adaptive_avg_pool2d(mask_features_spatial, (1, 1)).flatten(1)
        
        # Concatenate RGB and mask features
        combined_image_features = torch.cat([rgb_features_pooled, mask_features_pooled], dim=1)
        
        # Fuse RGB and mask features
        fused_features = self.fusion_layer(combined_image_features)  # (B, rgb_feature_dim)
        
        # Process each sample in the batch
        all_logits = []
        
        for i in range(batch_size):
            # Get bbox features for this sample
            bbox_features = BBoxMLPEncoder.extract_bbox_features(
                bbox_list[i], 
                self.input_size
            )
            bbox_features = bbox_features.unsqueeze(0).to(rgb_image.device)  # (1, 9)
            
            # Encode bbox features
            encoded_bbox = self.bbox_encoder(bbox_features)  # (1, bbox_output_dim)
            
            # Get fused image features for this sample
            sample_fused_features = fused_features[i:i+1]  # (1, rgb_feature_dim)
            
            # Combine fused image features with bbox features
            combined_features = torch.cat([sample_fused_features, encoded_bbox], dim=1)
            
            # Classify
            logits = self.classifier(combined_features)  # (1, num_classes)
            all_logits.append(logits)
        
        # Stack all logits
        final_logits = torch.cat(all_logits, dim=0)  # (B, num_classes)
        
        return final_logits


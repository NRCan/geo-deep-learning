import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x
    

class Decoder(nn.Module):
    """
    Decoder for SegFormer
    """
    def __init__(self, encoder="mit_b2",
                 in_channels=[64, 128, 320, 512],
                 feature_strides=[4, 8, 16, 32],
                 embedding_dim=768,
                 num_classes=1, dropout_ratio=0.1):
        super(Decoder, self).__init__()
        if encoder == "mit_b0":
            in_channels = [32, 64, 160, 256]
        if encoder in ["mit_b0", "mit_b1"]:
            embedding_dim = 256
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        self.num_classes = num_classes
        self.in_channels = in_channels
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout2d(dropout_ratio)

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, x):
        c1, c2, c3, c4 = x
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3]).contiguous()
        _c4 = F.interpolate(input=_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3]).contiguous()
        _c3 = F.interpolate(input=_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3]).contiguous()
        _c2 = F.interpolate(input=_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3]).contiguous()
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
import torch
import torch.nn as nn

class conv_block(nn.Module):
    '''
    Block for convolutional layer of U-Net at the encoder end.
    Args:
        ch_in : number of input channels
        ch_out : number of output channels
    Returns:
        feature map of the given input
    '''
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    '''
    Block for deconvolutional layer of U-Net at the decoder end
    Args:
        ch_in : number of input channels
        ch_out : number of output channels
    Returns:
        feature map of the given input
    '''
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    '''
    Recurrent convolution block for RU-Net and R2U-Net
    Args:
        ch_out : number of output channels
        t: the number of recurrent convolution blocks to be used
    Returns:
        feature map of the given input
    '''
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1

class RRCNN_block(nn.Module):
    '''
    Recurrent Residual convolution block for R2U-Net
    Args:
        ch_in  : number of input channels
        ch_out : number of output channels
        t    : the number of recurrent residual convolution blocks to be used
    Returns:
        feature map of the given input
    '''
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1  # residual learning

class RCNN_block(nn.Module):
    '''
    Recurrent convolution block for RU-Net
    Args:
        ch_in  : number of input channels
        ch_out : number of output channels
        t    : the number of recurrent residual convolution blocks to be used
    Returns:
        feature map of the given input
    '''
    def __init__(self, ch_in, ch_out, t=2):
        super(RCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x = self.RCNN(x)
        return x

class ResCNN_block(nn.Module):
    '''
    Residual convolution block 
    Args:
        ch_in  : number of input channels
        ch_out : number of output channels
    Returns:
        feature map of the given input
    '''
    def __init__(self, ch_in, ch_out):
        super(ResCNN_block, self).__init__()
        self.Conv = conv_block(ch_in, ch_out)
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv_1x1(x)
        x = self.Conv(x)
        return x + x1

class U_Net(nn.Module):
    '''
    U-Net Network.
    Implements traditional U-Net with a compressive encoder and an expanding decoder
    Args:
        img_ch: Input image channels
        output_ch: Number of channels expected in the output
    Returns:
        Feature map of input (batch_size, output_ch=1, h, w)
    '''
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class R2U_Net(nn.Module):
    '''
    R2U-Net Network.
    Implements U-Net with a RRCNN block.
    Args:
        img_ch: Input image channels
        output_ch: Number of channels expected in the output
        t: number of recurrent blocks expected
    Returns:
        Feature map of input (batch_size, output_ch=1, h, w)
    '''
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)
        
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class RecU_Net(nn.Module):
    '''
    RU-Net Network.
    Implements U-Net with a RCNN block.
    Args:
        img_ch: Input image channels
        output_ch: Number of channels expected in the output
        t: number of recurrent blocks expected
    Returns:
        Feature map of input (batch_size, output_ch=1, h, w)    
    '''
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(RecU_Net, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RCNN1 = RCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RCNN2 = RCNN_block(ch_in=64, ch_out=128, t=t)
        
        self.RCNN3 = RCNN_block(ch_in=128, ch_out=256, t=t)
        
        self.RCNN4 = RCNN_block(ch_in=256, ch_out=512, t=t)
        
        self.RCNN5 = RCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RCNN5 = RCNN_block(ch_in=1024, ch_out=512, t=t)
        
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RCNN4 = RCNN_block(ch_in=512, ch_out=256, t=t)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RCNN3 = RCNN_block(ch_in=256, ch_out=128, t=t)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RCNN2 = RCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class ResU_Net(nn.Module):
    '''
    Residual U-Net Network.
    Implements U-Net with a ResCNN block.
    Args:
        img_ch: Input image channels
        output_ch: Number of channels expected in the output
    Returns:
        Feature map of size (batch_size, output_ch, h, w)    
    '''
    def __init__(self, img_ch=3, output_ch=1):
        super(ResU_Net, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.ResCNN1 = ResCNN_block(ch_in=img_ch, ch_out=64)

        self.ResCNN2 = ResCNN_block(ch_in=64, ch_out=128)
        
        self.ResCNN3 = ResCNN_block(ch_in=128, ch_out=256)
        
        self.ResCNN4 = ResCNN_block(ch_in=256, ch_out=512)
        
        self.ResCNN5 = ResCNN_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_ResCNN5 = ResCNN_block(ch_in=1024, ch_out=512)
        
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_ResCNN4 = ResCNN_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_ResCNN3 = ResCNN_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_ResCNN2 = ResCNN_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.ResCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.ResCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.ResCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.ResCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.ResCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_ResCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_ResCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_ResCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_ResCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
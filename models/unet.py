import torch
from utils import utils
from torch import nn


class EncodingBlock(nn.Module):
    """Convolutional batch norm block with relu activation (main block used in the encoding steps)"""

    def __init__(self, in_size, out_size, kernel_size=3, padding=0, stride=1, dilation=1, batch_norm=True,
                 dropout=False, prob=0.5):
        super().__init__()

        if batch_norm:
            # reflection padding for same size output as input (reflection padding has shown better results than zero padding)
            layers = [nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                      nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                                dilation=dilation),
                      nn.PReLU(),
                      nn.BatchNorm2d(out_size),
                      nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                      nn.Conv2d(out_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                                dilation=dilation),
                      nn.PReLU(),
                      nn.BatchNorm2d(out_size),
                      ]
        else:
            layers = [nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                      nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                                dilation=dilation),
                      nn.PReLU(),
                      nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                      nn.Conv2d(out_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                                dilation=dilation),
                      nn.PReLU(), ]

        if dropout:
            layers.append(nn.Dropout(p=prob))

        self.EncodingBlock = nn.Sequential(*layers)

    def forward(self, input_data):
        output = self.EncodingBlock(input_data)
        return output


class DecodingBlock(nn.Module):
    """Module in the decoding section of the UNet"""

    def __init__(self, in_size, out_size, batch_norm=False, upsampling=True):
        super().__init__()
        if upsampling:
            self.up = nn.Sequential(utils.Interpolate(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))
        else:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)

        self.conv = EncodingBlock(in_size, out_size, batch_norm=batch_norm)

    def forward(self, input1, input2):
        output2 = self.up(input2)
        output1 = nn.functional.interpolate(input1, output2.size()[2:], mode='bilinear', align_corners=True)
        return self.conv(torch.cat([output1, output2], 1))


class UNet(nn.Module):
    """Main UNet architecture"""

    def __init__(self, num_classes, number_of_bands, dropout=False, prob=0.5):
        super().__init__()

        self.conv1 = EncodingBlock(number_of_bands, 64, dropout=dropout, prob=prob)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = EncodingBlock(64, 128, dropout=dropout, prob=prob)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = EncodingBlock(128, 256, dropout=dropout, prob=prob)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = EncodingBlock(256, 512, dropout=dropout, prob=prob)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = EncodingBlock(512, 1024, dropout=dropout, prob=prob)

        self.decode4 = DecodingBlock(1024, 512)
        self.decode3 = DecodingBlock(512, 256)
        self.decode2 = DecodingBlock(256, 128)
        self.decode1 = DecodingBlock(128, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, input_data):
        conv1 = self.conv1(input_data)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        decode4 = self.decode4(conv4, center)
        decode3 = self.decode3(conv3, decode4)
        decode2 = self.decode2(conv2, decode3)
        decode1 = self.decode1(conv1, decode2)

        final = nn.functional.interpolate(self.final(decode1), input_data.size()[2:], mode='bilinear', align_corners=True)

        return final


class UNetSmall(nn.Module):
    """Main UNet architecture"""

    def __init__(self, num_classes, number_of_bands, dropout=False, prob=0.5):
        super().__init__()

        self.conv1 = EncodingBlock(number_of_bands, 32, dropout=dropout, prob=prob)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = EncodingBlock(32, 64, dropout=dropout, prob=prob)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = EncodingBlock(64, 128, dropout=dropout, prob=prob)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.center = EncodingBlock(128, 256, dropout=dropout, prob=prob)

        self.decode3 = DecodingBlock(256, 128)
        self.decode2 = DecodingBlock(128, 64)
        self.decode1 = DecodingBlock(64, 32)

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, input_data):
        conv1 = self.conv1(input_data)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        center = self.center(maxpool3)

        decode3 = self.decode3(conv3, center)
        decode2 = self.decode2(conv2, decode3)
        decode1 = self.decode1(conv1, decode2)

        final = nn.functional.interpolate(self.final(decode1), input_data.size()[2:], mode='bilinear', align_corners=True)

        return final

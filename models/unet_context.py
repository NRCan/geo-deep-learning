import torch.nn as nn
import segmentation_models_pytorch as smp

def dilated_conv(n_convs, in_channels, out_channels, dilation):
    layers = []
    for i in range(n_convs):
        layers.append(nn.ZeroPad2d(dilation))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, dilation=dilation))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class contextModuleS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.ctx1_s = dilated_conv(n_convs=2, in_channels=in_channels, out_channels=out_channels, dilation=1)
        self.ctx2_s = dilated_conv(n_convs=1, in_channels=in_channels, out_channels=out_channels, dilation=2)
        self.ctx3_s = dilated_conv(n_convs=1, in_channels=in_channels, out_channels=out_channels, dilation=4)
        self.ctx4_s = dilated_conv(n_convs=1, in_channels=in_channels, out_channels=out_channels, dilation=8)
        self.ctx5_s = dilated_conv(n_convs=1, in_channels=in_channels, out_channels=out_channels, dilation=16)
        self.ctx7_s = dilated_conv(n_convs=1, in_channels=in_channels, out_channels=out_channels, dilation=1)
        self.conv_s = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        ctx1_s = self.ctx1_s(x)
        ctx2_s = self.ctx2_s(ctx1_s)
        ctx3_s = self.ctx3_s(ctx2_s)
        ctx4_s = self.ctx4_s(ctx3_s)
        ctx5_s = self.ctx5_s(ctx4_s)
        ctx7_s = self.ctx7_s(ctx5_s)
        conv_s = self.conv_s(ctx7_s)

        return conv_s


class contextModuleL(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.ctx0_l = dilated_conv(n_convs=1, in_channels=in_ch, out_channels=out_ch, dilation=1)
        self.ctx1_l = dilated_conv(n_convs=1, in_channels=in_ch, out_channels=out_ch * 2, dilation=1)
        self.ctx2_l = dilated_conv(n_convs=1, in_channels=in_ch * 2, out_channels=out_ch * 4, dilation=2)
        self.ctx3_l = dilated_conv(n_convs=1, in_channels=in_ch * 4, out_channels=out_ch * 8, dilation=4)
        self.ctx4_l = dilated_conv(n_convs=1, in_channels=in_ch * 8, out_channels=out_ch * 16, dilation=8)
        self.ctx5_l = dilated_conv(n_convs=1, in_channels=in_ch * 16, out_channels=out_ch * 32, dilation=16)
        self.ctx7_l = dilated_conv(n_convs=1, in_channels=in_ch * 32, out_channels=out_ch * 32, dilation=1)
        self.conv_l = nn.Conv2d(in_ch * 32, out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        ctx0_l = self.ctx0_l(x)
        ctx1_l = self.ctx1_l(ctx0_l)
        ctx2_l = self.ctx2_l(ctx1_l)
        ctx3_l = self.ctx3_l(ctx2_l)
        ctx4_l = self.ctx4_l(ctx3_l)
        ctx5_l = self.ctx5_l(ctx4_l)
        ctx7_l = self.ctx7_l(ctx5_l)
        conv_l = self.conv_l(ctx7_l)

        return conv_l


class ContextualUnet(nn.Module):

    def __init__(self, num_bands, num_channels):
        super().__init__()

        model = smp.Unet(encoder_name="resnext50_32x4d",
                         encoder_weights="imagenet",
                         encoder_depth=5,
                         in_channels=num_bands,
                         classes=num_channels,
                         activation=None)
        # self.e_ch = model.encoder.out_channels[-1]
        self.d_ch = model.decoder.blocks[-1].conv2[0].out_channels
        self.model = model
        self.ctx_s = contextModuleS(num_bands, num_bands)
        self.ctx_l = contextModuleL(self.d_ch, self.d_ch)

    def forward(self, x):

        s_ctx = self.ctx_s(x)
        num_features = self.model.encoder(s_ctx)
        # num_features[-1] = s_ctx
        decoder_output = self.model.decoder(*num_features)
        l_ctx = self.ctx_l(decoder_output)
        masks = self.model.segmentation_head(l_ctx)

        return masks
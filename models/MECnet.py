from torch import nn
import torch.nn.functional as F
import torch


def conv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def conv1x1_bn_relu(in_features, out_features):
    return nn.Sequential(
        nn.Conv2d(in_features, out_features, kernel_size=1),
        nn.BatchNorm2d(out_features),
        nn.ReLU(inplace=True)
    )


def upsample_layer(in_channels, out_channels, scale_factor=2):
    return nn.Sequential(
        nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))


def upsample_layer4x4(in_channels, out_channels):
    return nn.Sequential(
        nn.Upsample(scale_factor=4, mode='bilinear'),
        nn.Conv2d(in_channels, out_channels, kernel_size=1))


class MultiFieAtConv2d(nn.Module):
    def __init__(self, channels):
        super(MultiFieAtConv2d, self).__init__()
        self.short_field_conv = ShortFieldConv2d(in_channels=channels, out_channels=channels)
        self.mid_field_conv = MidLongFieldConv2d(channels=channels)
        self.global_field_conv = GlobalFieldConv2d(in_channels=channels, channels=channels)
        self.merge_conv = nn.Sequential(
            nn.Conv2d(channels*4, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        s = self.short_field_conv(x)
        m = self.mid_field_conv(x)
        g = self.global_field_conv(x)
        merge = self.merge_conv(torch.cat((x, s, m, g), dim=1))
        return merge

# class SAM(nn.Module):
#     """ spatial attention module"""
#     def __init__(self, in_dim):
#         super(SAM, self).__init__()
#         self.chanel_in = in_dim
#
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X (HxW) X (HxW)
#         """
#         m_batchsize, C, height, width = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
#
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, height, width)
#
#         out = self.gamma*out + x
#         return out

class ShortFieldConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShortFieldConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.conv(x)
        w = self.bn(w)
        w = self.sigmoid(w)
        x = w*x + x
        return x


class MidLongFieldConv2d(nn.Module):
    '''
    Three scales are used to extract medium features
    '''
    def __init__(self, channels):
        super(MidLongFieldConv2d, self).__init__()
        channels_mid = max(channels//4, 32)
        self.conv7x7 = nn.Conv2d(channels, channels_mid, kernel_size=(7, 7), stride=4, padding=3)
        self.bn1 = nn.BatchNorm2d(channels_mid)

        self.conv3x3 = conv3x3_bn_relu(channels_mid, channels_mid)
        self.conv_upsample_1 = upsample_layer4x4(channels_mid, channels)
        self.bn_upsample_1 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x1 = self.conv7x7(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.conv3x3(x1)

        merge = self.relu(self.bn_upsample_1(self.conv_upsample_1(x1)))
        # w = self.sigmoid(self.bn_upsample_1(self.conv_upsample_1(x1)))
        # x = w * x + x
        return merge


class GlobalFieldConv2d(nn.Module):
    def __init__(self, in_channels, channels, reduction_factor=4):
        super(GlobalFieldConv2d, self).__init__()
        inter_channels = max(in_channels // reduction_factor, 32)
        self.fc1 = nn.Conv2d(in_channels, inter_channels, 1)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(inter_channels, channels, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, x.shape[-1])
        w = self.fc1(w)
        w = self.bn1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.bn2(w)
        w = self.sigmoid(w)
        x = w * x + x
        return x


class rSoftMax(nn.Module):
    def __init__(self, radix, groups):
        super().__init__()
        self.radix = radix
        self.groups = groups

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.groups, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

class GlobalHighLowMerge(nn.Module):
    '''
    conv1x1: 通道降维、高低语义粗融合
    global-inf: 全局信息引导通道维度的精细融合
    conv3x3: 局部信息引导空间维度的精细融合
    '''
    def __init__(self, channels, reduction_factor=4):
        super(GlobalHighLowMerge, self).__init__()
        inter_channels = max(channels//reduction_factor, 32)
        self.conv1x1 = nn.Conv2d(channels*2, channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Conv2d(channels, inter_channels, 1)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels, 1)

        self.conv3x3bn_relu = conv3x3_bn_relu(channels, channels)

    def forward(self, high_f, low_f):
        x = torch.cat((high_f, low_f), dim=1)
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.relu(x)

        w = F.adaptive_avg_pool2d(x, x.shape[-1])
        w = self.fc1(w)
        w = self.bn1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = torch.sigmoid(w)
        # w = self.relu(w)  # 测试一下

        x = x * w + x
        x = self.conv3x3bn_relu(x)
        return x


class MSP3(nn.Module):

    def __init__(self, num_classes=1, inter_channels=32, channels=3, feature_visualization=False):
        super(MSP3, self).__init__()
        self.conv1x1 = nn.Conv2d(5, inter_channels, 1) # channels*num_classes
        self.bn1x1 = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv3x3 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1)
        self.bn3x3 = nn.BatchNorm2d(inter_channels)

        self.conv5x5 = nn.Conv2d(inter_channels, inter_channels, kernel_size=5, stride=1, padding=2)
        self.bn5x5 = nn.BatchNorm2d(inter_channels)

        self.conv7x7 = nn.Conv2d(inter_channels, inter_channels, kernel_size=7, stride=1, padding=3)
        self.bn7x7 = nn.BatchNorm2d(inter_channels)

        self.conv_merge = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1)
        self.bn_merge = nn.BatchNorm2d(inter_channels)

        self.conv = nn.Conv2d(inter_channels, num_classes, kernel_size=1, bias=False)
        self.feature_visualization = feature_visualization

    def forward(self, x_tuple):
        x = torch.cat(x_tuple, dim=1)
        x = self.conv1x1(x)
        x = self.bn1x1(x)
        x = self.relu(x)

        w3x3 = self.conv3x3(x)
        w3x3 = self.bn3x3(w3x3)
        # w3x3 = self.relu(w3x3)
        w3 = torch.sigmoid(w3x3)

        w5x5 = self.conv5x5(w3x3)
        w5x5 = self.bn5x5(w5x5)
        # w5x5 = self.relu(w5x5)
        w5 = torch.sigmoid(w5x5)

        w7x7 = self.conv7x7(w5x5)
        w7x7 = self.bn7x7(w7x7)
        # w7x7 = self.relu(w7x7)
        w7 = torch.sigmoid(w7x7)

        x = x * w7 + x * w5 + x * w3 + x
        # x = x * w7x7 + x
        x = self.conv_merge(x)
        x = self.bn_merge(x)
        x = self.relu(x)
        x_ = x.clone()
        x = self.conv(x)
        if self.feature_visualization:
            x_visualize = F.interpolate(x_, size=(1024, 1024), mode='bilinear', align_corners=False)
            return x, x_visualize
        else:
            return x


def inspect_output_layer(in_features, class_num):
    return nn.Conv2d(in_features, class_num, kernel_size=1, stride=1, bias=False)


class MFConvNet(nn.Module):
    def __init__(self, channels=3, class_num=1):
        super(MFConvNet, self).__init__()
        channel_list = [64, 128, 256, 512, 512]
        self.class_num = class_num
        self.layer1 = nn.Sequential(
            conv3x3_bn_relu(channels, 64),
            MultiFieAtConv2d(channel_list[0])
            # MFCModule(channel_list[0], num_layer_id=0)
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(64, 128),
            MultiFieAtConv2d(channel_list[1]),
            # MFCModule(channel_list[1], num_layer_id=1),
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(128, 256),
            MultiFieAtConv2d(channel_list[2]),
            # MFCModule(channel_list[2], num_layer_id=2),
            conv3x3_bn_relu(256, 256),
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(256, 512),
            MultiFieAtConv2d(channel_list[3]),
            # MFCModule(channel_list[3], num_layer_id=3),
            conv3x3_bn_relu(512, 512))

        self.layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(512, 512),
            MultiFieAtConv2d(channel_list[4]),
            # MFCModule(channel_list[4], num_layer_id=4),
            conv3x3_bn_relu(512, 512))
        self._initialize_weights()

    def _initialize_weights(self):
        count = 0
        for m in self.modules():
            count += 1
            print(count, m)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        e5 = self.layer5(e4)
        return e1, e2, e3, e4, e5


# class VGGNet(nn.Module):
#     def __init__(self, channels=3, class_num=1):
#         super(VGGNet, self).__init__()
#         self.class_num = class_num
#         self.layer1 = nn.Sequential(
#             conv3x3_bn_relu(channels, 64),
#             conv3x3_bn_relu(64, 64),
#         )
#         self.layer2 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             conv3x3_bn_relu(64, 128),
#             conv3x3_bn_relu(128, 128),
#         )
#         self.layer3 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             conv3x3_bn_relu(128, 256),
#             conv3x3_bn_relu(256, 256),
#             conv3x3_bn_relu(256, 256),
#         )
#         self.layer4 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             conv3x3_bn_relu(256, 512),
#             conv3x3_bn_relu(512, 512),
#             conv3x3_bn_relu(512, 512),
#         )
#         self.layer5 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             conv3x3_bn_relu(512, 512),
#             conv3x3_bn_relu(512, 512),
#             conv3x3_bn_relu(512, 512),
#         )
#         self._initialize_weights()
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         e1 = self.layer1(x)
#         e2 = self.layer2(e1)
#         e3 = self.layer3(e2)
#         e4 = self.layer4(e3)
#         e5 = self.layer5(e4)
#         return e1, e2, e3, e4, e5


class DecoderNet(nn.Module):
    def __init__(self, num_classes=1):
        super(DecoderNet, self).__init__()
        self.class_num = num_classes
        channel_list = [64, 128, 256, 512, 512]
        self.out5 = inspect_output_layer(channel_list[4], class_num=self.class_num)

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv1x1_bn_relu(channel_list[4], channel_list[3])
        )
        self.decoder4 = GlobalHighLowMerge(channel_list[3])
        self.out4 = inspect_output_layer(channel_list[3], class_num=self.class_num)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv1x1_bn_relu(channel_list[3], channel_list[2])
        )
        self.decoder3 = GlobalHighLowMerge(channel_list[2])
        self.out3 = inspect_output_layer(channel_list[2], class_num=self.class_num)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv1x1_bn_relu(channel_list[2], channel_list[1]))
        self.decoder2 = GlobalHighLowMerge(channel_list[1])
        self.out2 = inspect_output_layer(channel_list[1], class_num=self.class_num)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv1x1_bn_relu(channel_list[1], channel_list[0]))
        self.decoder1 = GlobalHighLowMerge(channel_list[0])
        self.out1 = inspect_output_layer(channel_list[0], class_num=self.class_num)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, e1, e2, e3, e4, e5):
        o5 = nn.Upsample(scale_factor=16, mode='bilinear')(self.out5(e5))

        up4 = self.up4(e5)
        d4 = self.decoder4(up4, e4)
        o4 = nn.Upsample(scale_factor=8, mode='bilinear')(self.out4(d4))

        up3 = self.up3(d4)
        d3 = self.decoder3(up3, e3)
        o3 = nn.Upsample(scale_factor=4, mode='bilinear')(self.out3(d3))

        up2 = self.up2(d3)
        d2 = self.decoder2(up2, e2)
        o2 = nn.Upsample(scale_factor=2, mode='bilinear')(self.out2(d2))

        up1 = self.up1(d2)
        d1 = self.decoder1(up1, e1)
        o1 = self.out1(d1)
        return o1, o2, o3, o4, o5


class DecUNet(nn.Module):
    def __init__(self, num_classes=1):
        super(DecUNet, self).__init__()
        self.class_num = num_classes
        channel_list = [64, 128, 256, 512, 512]
        self.out5 = inspect_output_layer(channel_list[4], class_num=self.class_num)

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv1x1_bn_relu(channel_list[4], channel_list[3])
        )
        self.decoder4 = nn.Sequential(
            conv3x3_bn_relu(channel_list[3]*2, channel_list[3]),
            conv3x3_bn_relu(channel_list[3], channel_list[3]),
        )
        self.out4 = inspect_output_layer(channel_list[3], class_num=self.class_num)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv1x1_bn_relu(channel_list[3], channel_list[2])
        )
        self.decoder3 = nn.Sequential(
            conv3x3_bn_relu(channel_list[2]*2, channel_list[2]),
            conv3x3_bn_relu(channel_list[2], channel_list[2]),
        )
        self.out3 = inspect_output_layer(channel_list[2], class_num=self.class_num)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv1x1_bn_relu(channel_list[2], channel_list[1]))
        self.decoder2 = nn.Sequential(
            conv3x3_bn_relu(channel_list[1]*2, channel_list[1]),
            conv3x3_bn_relu(channel_list[1], channel_list[1]),
        )
        self.out2 = inspect_output_layer(channel_list[1], class_num=self.class_num)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv1x1_bn_relu(channel_list[1], channel_list[0]))
        self.decoder1 = nn.Sequential(
            conv3x3_bn_relu(channel_list[0]*2, channel_list[0]),
            conv3x3_bn_relu(channel_list[0], channel_list[0]),
        )
        self.out1 = inspect_output_layer(channel_list[0], class_num=self.class_num)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, e1, e2, e3, e4, e5):
        o5 = nn.Upsample(scale_factor=16, mode='bilinear')(self.out5(e5))

        up4 = self.up4(e5)
        d4 = self.decoder4(torch.cat((up4, e4), dim=1))
        o4 = nn.Upsample(scale_factor=8, mode='bilinear')(self.out4(d4))

        up3 = self.up3(d4)
        d3 = self.decoder3(torch.cat((up3, e3), dim=1))
        o3 = nn.Upsample(scale_factor=4, mode='bilinear')(self.out3(d3))

        up2 = self.up2(d3)
        d2 = self.decoder2(torch.cat((up2, e2), dim=1))
        o2 = nn.Upsample(scale_factor=2, mode='bilinear')(self.out2(d2))

        up1 = self.up1(d2)
        d1 = self.decoder1(torch.cat((up1, e1), dim=1))
        o1 = self.out1(d1)
        return o1, o2, o3, o4, o5


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


class DecFCN(nn.Module):
    def __init__(self, num_classes=1, visualization=False, norm_layer=nn.BatchNorm2d):
        super(DecFCN, self).__init__()
        self.head = _FCNHead(512, num_classes, norm_layer)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        self.visualization = visualization

    def forward(self, x, e3, e4, e5):
        score_fr = self.head(e5)  # 16
        score_pool4 = self.score_pool4(e4)  # 8
        score_pool3 = self.score_pool3(e3)  # 4

        upscore2 = F.interpolate(score_fr, score_pool4.size()[2:], mode='bilinear', align_corners=True)
        fuse_pool4 = upscore2 + score_pool4

        upscore_pool4 = F.interpolate(fuse_pool4, score_pool3.size()[2:], mode='bilinear', align_corners=True)
        fuse_pool3 = upscore_pool4 + score_pool3
        out = F.interpolate(fuse_pool3, x.size()[2:], mode='bilinear', align_corners=True)
        if self.visualization:
            x_visualize = F.interpolate(fuse_pool3, size=(1024, 1024), mode='bilinear', align_corners=False)
            return out, x_visualize
        else:
            return out


class MergeDecoder(nn.Module):
    def __init__(self, num_classes=1, channels=3, visualization=False):
        super(MergeDecoder, self).__init__()
        self.class_num = num_classes
        self.visualization = visualization
        self.merge_feat = MSP3(num_classes=self.class_num, inter_channels=32, channels=channels,
                               feature_visualization=self.visualization)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, o_tuple):
        if self.visualization:
            x, x_visualize = self.merge_feat(o_tuple)
            return x, x_visualize
        else:
            x = self.merge_feat(o_tuple)
            return x


# class FCN(nn.Module):
#     def __init__(self, visualization=False):
#         super(FCN, self).__init__()
#         self.visualization = visualization
#         self.encoder = VGGNet()
#         self.decoder = DecFCN(visualization=visualization)
#
#     def forward(self, x):
#         e1, e2, e3, e4, e5 = self.encoder(x)
#
#         if self.visualization:
#             o1, x_visualize = self.decoder(x, e3, e4, e5)
#             return o1, x_visualize
#         else:
#             o1 = self.decoder(x, e3, e4, e5)
#             return o1
#
#
# class FMNet(nn.Module):
#     '''FCN + MEC-module'''
#
#     def __init__(self, visualization=False):
#         super(FMNet, self).__init__()
#         self.encoder = MFConvNet()
#         self.decoder = DecFCN(visualization=visualization)
#         self.visualization = visualization
#
#     def forward(self, x):
#         e1, e2, e3, e4, e5 = self.encoder(x)
#         if self.visualization:
#             o1, x_visualize = self.decoder(x, e3, e4, e5)
#             return o1, x_visualize
#         else:
#             o1 = self.decoder(x, e3, e4, e5)
#             return o1
#
#
# class MMNet(nn.Module):
#     '''MEC-module + DSFF-module'''
#
#     def __init__(self, visualization=False):
#         super(MMNet, self).__init__()
#         self.encoder = MFConvNet()
#         self.decoder = DecUNet()
#         self.msp = MergeDecoder(visualization=visualization)
#         self.visualization = visualization
#
#     def forward(self, x):
#         e1, e2, e3, e4, e5 = self.encoder(x)
#         o1, o2, o3, o4, o5 = self.decoder(e1, e2, e3, e4, e5)
#         if self.visualization:
#             res, visual_feat_map = self.msp((o1, o2, o3))
#             return res, visual_feat_map
#         else:
#             res = self.msp((o1, o2, o3))
#             return res
#
#
# class MGMNet(nn.Module):
#     '''MEC-module + MPF-module + DSFF-module'''
#
#     def __init__(self):
#         super(MGMNet, self).__init__()
#         self.encoder = MFConvNet()
#         self.decoder = DecoderNet()
#         self.msp = MergeDecoder()
#
#     def forward(self, x):
#         e1, e2, e3, e4, e5 = self.encoder(x)
#         o1, o2, o3, o4, o5 = self.decoder(e1, e2, e3, e4, e5)
#         res = self.msp(o1, o2, o3)
#         return res


class MECNet(nn.Module):
    '''
    MEC-module + MPF-module + DSFF-module + Multi-Constraint
    '''

    def __init__(self, visualization=False):
        super(MECNet, self).__init__()
        self.encoder = MFConvNet()
        self.decoder = DecoderNet()
        self.visualization = visualization
        self.msp = MergeDecoder(num_classes=1, channels=3, visualization=self.visualization)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        o1, o2, o3, o4, o5 = self.decoder(e1, e2, e3, e4, e5)
        if self.visualization:
            res, visual_feat_map = self.msp((o1, o2, o3, o4, o5))
            return (res, o1, o2, o3, o4, o5), visual_feat_map
        else:
            res = self.msp((o1, o2, o3, o4, o5))
            return res, o1, o2, o3, o4, o5


if __name__ == '__main__':
    from thop import profile
    import torchsummary
    band_num = 3
    class_num = 1
    model = MECNet()
    input = torch.randn(1, 3, 512, 512)
    # flops, params = profile(model, inputs=(input,))
    # model.cuda()
    torchsummary.summary(model, (3,512,512))
    # print('flops(G): %.3f' % (flops / 1e+9))
    # print('params(M): %.3f' % (params / 1e+6))

'''
legend:
    MFConvNet = MFConvNet
    DecoderNet = DecoderNet
    
'''
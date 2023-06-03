# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp

import numpy as np
import torch
import torch.nn as nn


from torchvision.models import resnet50,resnet34,resnet18

#deeplabv3
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, dilations = [1, 6, 12, 18]):
        super(ASPP, self).__init__()
        
        self.aspp_conv=nn.ModuleList()
        for dilation in dilations:
            if dilation==1:
                self.aspp_conv.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )
                )        
            else:
                self.aspp_conv.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                            stride=1, padding=dilation, dilation=dilation),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )
                )
         
        self.aspp_pooling=nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.contact=nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations)+1), out_channels, kernel_size=1,stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.dropout=nn.Dropout(0.5)
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        xs = []
            
        for conv in self.aspp_conv:
            temp=conv(x)
            xs.append(temp)
            
        temp=self.aspp_pooling(x)#pooling image
        temp = F.interpolate(temp, size=input_shape, mode='bilinear', align_corners=False)

        xs.append(temp)

        out = torch.cat(xs, dim=1)#contact
        out=self.contact(out)
 
        out=self.dropout(out)
        return out


class DeepLabV3_resnet50_stride16(nn.Module):
    def __init__(self, in_channels=3, out_channels=21):
        super(DeepLabV3_resnet50_stride16, self).__init__()
        # Backbone
        resnet = resnet50(pretrained=1)
        #用空洞卷积改变下采样，把放缩必减小
        resnet.layer4[0].conv2.stride=1
        resnet.layer4[0].conv2.padding=(1,1)
        resnet.layer4[0].conv2.dilation=(1,1)
        resnet.layer4[0].downsample[0].stride=1
        for i in range(1,3):
            # resnet.layer4[i].conv2.stride=1
            resnet.layer4[i].conv2.padding=(i*2,i*2)
            resnet.layer4[i].conv2.dilation=(i*2,i*2)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        # DeepLabHead
        self.DeepLabHead = nn.Sequential(
            ASPP(2048, 256),
            nn.Conv2d(256, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )
    def forward(self, x):
        input_shape = x.shape[-2:]
        # Apply backbone
        x = self.backbone(x)
        # Apply DeepLabHead
        x = self.DeepLabHead(x)
        # Upsample output to match input spatial dimensions
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

    
class DeepLabV3_resnet50_stride8(nn.Module):
    def __init__(self, in_channels=3, out_channels=21):
        super(DeepLabV3_resnet50_stride8, self).__init__()
        # Backbone
        resnet = resnet50(pretrained=1)
        
        resnet.layer3[0].conv2.stride=1
        resnet.layer3[0].downsample[0].stride=1
        # for i in range(1,6):
            
            
        resnet.layer4[0].conv2.stride=1
        resnet.layer4[0].conv2.padding=(1,1)
        resnet.layer4[0].conv2.dilation=(1,1)
        resnet.layer4[0].downsample[0].stride=1
        for i in range(1,3):
            # resnet.layer4[i].conv2.stride=1
            resnet.layer4[i].conv2.padding=(i*2,i*2)
            resnet.layer4[i].conv2.dilation=(i*2,i*2)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # DeepLabHead
        self.DeepLabHead = nn.Sequential(
            ASPP(2048, 256),
            nn.Conv2d(256, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )
    def forward(self, x):
        input_shape = x.shape[-2:]
        # Apply backbone
        x = self.backbone(x)
        # Apply DeepLabHead
        x = self.DeepLabHead(x)
        # Upsample output to match input spatial dimensions
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

class DeepLabV3_resnet34_stride16(nn.Module):
    def __init__(self, in_channels=3, out_channels=21):
        super(DeepLabV3_resnet34_stride16, self).__init__()
        # Backbone
        resnet=resnet34(pretrained=1)
        resnet.layer4[0].conv1.stride=1
        resnet.layer4[0].conv1.padding=(1,1)
        resnet.layer4[0].conv1.dilation=(1,1)
        resnet.layer4[0].downsample[0].stride=1
        for i in range(1,3):
            # resnet.layer4[i].conv2.stride=1
            resnet.layer4[i].conv1.padding=(i*2,i*2)
            resnet.layer4[i].conv1.dilation=(i*2,i*2)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        # DeepLabHead
        self.DeepLabHead = nn.Sequential(
            ASPP(512, 256),
            nn.Conv2d(256, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )
    def forward(self, x):
        input_shape = x.shape[-2:]
        # Apply backbone
        x = self.backbone(x)
        # Apply DeepLabHead
        x = self.DeepLabHead(x)
        # Upsample output to match input spatial dimensions
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x
    
class DeepLabV3_resnet18_stride16(nn.Module):
    def __init__(self, in_channels=3, out_channels=21):
        super(DeepLabV3_resnet18_stride16, self).__init__()
        # Backbone
        resnet=resnet18(pretrained=1)
        resnet.layer4[0].conv1.stride=1
        resnet.layer4[0].conv1.padding=(1,1)
        resnet.layer4[0].conv1.dilation=(1,1)
        resnet.layer4[0].downsample[0].stride=1
        
        # resnet.layer4[i].conv2.stride=1
        resnet.layer4[1].conv1.padding=(2,2)
        resnet.layer4[1].conv1.dilation=(2,2)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        # DeepLabHead
        self.DeepLabHead = nn.Sequential(
            ASPP(512, 256),
            nn.Conv2d(256, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )
    def forward(self, x):
        input_shape = x.shape[-2:]
        # Apply backbone
        x = self.backbone(x)
        # Apply DeepLabHead
        x = self.DeepLabHead(x)
        # Upsample output to match input spatial dimensions
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

# UNet的一大层，包含了两层小的卷积
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# 定义输入进来的第一层
class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

# 定义encoder中的向下传播，包括一个maxpool和一大层    
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

# 定义decoder中的向上传播
class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()
        # 定义了self.up的方法
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2) # // 除以的结果向下取整

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):  # x2是左侧的输出，x1是上一大层来的输出
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1) # 将两个tensor拼接在一起 dim=1：在通道数（C）上进行拼接
        x = self.conv(x)
        return x

# 定义最终的输出
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class Unet(nn.Module):
    def __init__(self, in_channels, classes): # in_channels 图片的通道数，1为灰度图，3为彩色图
        super(Unet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = classes

        self.inc = InConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN32s(nn.Module):

    def __init__(self, n_class=21):
        super(FCN32s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32,
                                          bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)

        h = self.upscore(h)
        h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

        return h

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())

class Net_seg(nn.Module):
    def __init__(self, classes):
        super(Net_seg, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1),  #
                nn.ReLU(),
                nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, 1, 1),  #
                nn.ReLU(),
                nn.MaxPool2d(2)  #
        )
        self.conv3 = nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1),  #
                nn.ReLU(),
                nn.MaxPool2d(2)  #
        )

        self.head = nn.Sequential(
                nn.Conv2d(64, classes, 3, 1, 1)
        )

    def forward(self, x):
        size = x.size()[2:]
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.head(out)
        # print(out.shape)
        out = F.interpolate(out, size, mode='bilinear', align_corners=True)
        return out
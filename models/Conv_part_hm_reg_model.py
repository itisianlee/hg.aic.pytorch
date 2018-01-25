# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from config import opt
from .new_ResNet import resnet152
from .hourglass import Hourglass, Bottleneck
from collections import OrderedDict

__all__ = ['Part_detection_subnet_model', 'Regression_subnet']


class Part_detection_subnet_model(nn.Module):
    def __init__(self, opt=None):
        super(Part_detection_subnet_model, self).__init__()
        self.part_resnet = resnet152(pretrained=True)

        for param in self.part_resnet.parameters():
            param.requires_grad = False
        self.conv_b6 = nn.Conv2d(2048, 16, kernel_size=1, stride=1)
        self.bn_b6 = nn.BatchNorm2d(16)
        self.relu_b6 = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(16, 14, kernel_size=4, stride=4)
        self.upsample = nn.Upsample(size=(256, 256))

    def forward(self, x):
        x = self.part_resnet(x)
        x = self.conv_b6(x)
        x = self.deconv(x)
        x = self.upsample(x)
        x = F.sigmoid(x)

        return x


class Regression_subnet(nn.Module):
    """Regression subnet"""

    def __init__(self, opt=None, block=Bottleneck, num_blocks=3, depth=4, bias=True):
        super(Regression_subnet, self).__init__()
        self.inplanes = 64
        self.num_feats = 128
        self.detection_subnet = load_detection_subnet()

        self.conv1 = nn.Conv2d(17, 64, kernel_size=7, stride=2, padding=3, bias=bias)  # out:[batch_size,64,128,128]
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        # 使用一个简单的网络结构,降采样->上采样
        # self.down1 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3, bias=bias),  # out:[bs, 128, 64, 64]
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True)
        # )
        # self.down2 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=bias),  # out:[bs, 256, 64, 64]
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True)
        # )
        # self.up1 = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True)
        # )
        # self.up2 = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=bias),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv3 = nn.Conv2d(128, 14, kernel_size=1, stride=1, bias=bias)

        ### -----------------------------------
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # out:[bs,64,64,64]
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, self.num_feats * 2,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(self.num_feats * 2),
        )
        self.d2 = nn.Sequential(
            block(self.inplanes, self.num_feats, downsample=downsample),
            block(self.num_feats * 2, self.num_feats),
            block(self.num_feats * 2, self.num_feats)
        )  # out:[bs,256,64,64]

        self.hg = Hourglass(block, num_blocks, self.num_feats, depth)

        self.d61 = nn.Conv2d(256, 512, kernel_size=1, stride=1)
        self.bn_d61 = nn.BatchNorm2d(512)
        self.relu_d61 = nn.ReLU(inplace=True)

        # self.d62 = nn.Conv2d(512, 512, kernel_size=1, stride=1)
        # self.bn_d62 = nn.BatchNorm2d(512)
        # self.relu_d62 = nn.ReLU(inplace=True)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.d7 = nn.Conv2d(512, 14, kernel_size=1, stride=1)  # out:[bs,14,256,256]

        # self.d5 = nn.ConvTranspose2d(14, 14, kernel_size=4, stride=4)

        ### -----------------------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                init.xavier_uniform(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d) and m.weight.requires_grad:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        detec = self.detection_subnet(x)
        # print("Detection size:", detec.size())
        # a = detec.cpu().data.numpy()[0][0]
        # import numpy as np
        # print("Detec Data:", np.max(a))
        # print("Input size:", x.size())
        # print("Input Data:", x.cpu().data.numpy()[0][0][0][:10])
        x = torch.cat((detec, x), 1)
        # print("Cat size:", x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.d2(x)
        x = self.hg(x)

        x = self.d61(x)
        x = self.bn_d61(x)
        x = self.relu_d61(x)
        x = self.up1(x)
        x = self.up2(x)

        # x = self.d62(x)
        # x = self.bn_d62(x)
        # x = self.relu_d62(x)

        x = self.d7(x)
        # --------
        # x = self.down1(x)
        # x = self.down2(x)
        # x = self.up1(x)
        # x = self.up2(x)
        # x = self.conv2(x)
        # x = self.conv3(x)

        return F.sigmoid(x)


def load_detection_subnet():
    model = Part_detection_subnet_model()
    state_dict = torch.load(opt.checkpoints + opt.model[0] + '.pkl',
                            map_location=lambda storage, loc: storage.cuda(0))
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False

    return model


if __name__ == '__main__':
    # reg_model = Regression_subnet()
    # x = torch.autograd.Variable(torch.arange(0, 17 * 256 * 256).view(1, 17, 256, 256)).float()
    # o = reg_model(x)
    # print (o.size())
    model = load_detection_subnet()
    print(model)

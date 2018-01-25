# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from config import opt
from .new_ResNet import resnet101
from .hourglass import Hourglass, Bottleneck

__all__ = ['Part_detection_subnet101', 'Regression_subnet101']


class Part_detection_subnet101(nn.Module):
    def __init__(self):
        super(Part_detection_subnet101, self).__init__()
        self.part_resnet = resnet101(pretrained=True)
        self.add_extras = Add_extras()
        # for param in self.part_resnet.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x = self.part_resnet(x)  # [16, 2048, 16, 16]
        return self.add_extras(x)


class Add_extras(nn.Module):
    def __init__(self):
        super(Add_extras, self).__init__()
        self.b6 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.b7 = nn.Sequential(
            # nn.Upsample(size=(128, 128), mode='bilinear'),
            nn.ConvTranspose2d(256, 14, kernel_size=4, stride=4),
            nn.Upsample(size=(256, 256))
        )

    def forward(self, x):
        x = self.b6(x)
        x = self.b7(x)
        return F.sigmoid(x)


class Regression_subnet101(nn.Module):
    """Regression subnet"""

    def __init__(self, block=Bottleneck, num_blocks=3, depth=4, bias=True):
        super(Regression_subnet101, self).__init__()
        self.inplanes = 64
        self.num_feats = 128
        self.detection_subnet = load_detection_subnet()
        self.d1 = nn.Sequential(
            nn.Conv2d(17, 64, kernel_size=7, stride=2, padding=3, bias=bias),  # out:[batch_size,64,128,128]
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # out:[bs,64,64,64]
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, self.num_feats * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.num_feats * 2),
        )
        self.d2 = nn.Sequential(
            block(self.inplanes, self.num_feats, downsample=downsample),
            block(self.num_feats * 2, self.num_feats),
            block(self.num_feats * 2, self.num_feats)
        )  # out:[bs,256,64,64]
        self.hg = Hourglass(block, num_blocks, self.num_feats, depth)

        self.d6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.d7 = nn.Conv2d(512, 14, kernel_size=1, stride=1)  # out:[bs,14,256,256]

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                init.xavier_uniform(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d) and m.weight.requires_grad:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        detec = self.detection_subnet(x)
        x = torch.cat((detec, x), 1)
        x = self.d1(x)
        x = self.maxpool1(x)
        x = self.d2(x)
        x = self.hg(x)
        x = self.d6(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.d7(x)
        return F.sigmoid(x)


def load_detection_subnet():
    model = Part_detection_subnet101()
    state_dict = torch.load(opt.checkpoints + opt.model[2] + '.model',
                            map_location=lambda storage, loc: storage.cuda(1))
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

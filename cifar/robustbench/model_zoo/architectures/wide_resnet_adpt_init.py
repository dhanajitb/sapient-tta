import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


# groupwise and pointwise group convolutional size
gp = 8
pt = 16
###############################################


class ConvAdapt(nn.Module):
    def __init__(self, in_channels, out_channels, p):
        super(ConvAdapt, self).__init__()
        # Groupwise Convolution
        self.gwc = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            groups=int(p / gp),
            bias=True,
        )
        # print("b4 init self.gwc.weight",self.gwc.weight)
        # Adding Init here
        # self.gwc.weight.data.fill_(0.0)
        # # print("aft init self.gwc.weight",self.gwc.weight)
        # self.gwc.bias.data.fill_(0.0)
        assert in_channels == out_channels

        # Pointwise Convolution
        # self.pwc = nn.Conv2d(
        #     in_channels, out_channels, kernel_size=1, groups=int(p / pt), bias=True
        # )

        self.pwc = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, groups=int(1), bias=True    # need to formulate init for groups > 1
        )

        # # Adding Init here
        # self.pwc.weight.data=torch.eye(in_channels, requires_grad=True).view(in_channels,in_channels,1,1)
        # self.pwc.bias.data.fill_(0.0)

        # print("doing init")
        # assert False

    def forward(self, x):
        # print("x.shape:", x.shape)
        # print("self.gwc(x).shape:", self.gwc(x).shape)
        # print("self.pwc:",self.pwc)
        # print("self.pwc(x).shape:", self.pwc(x).shape)
        return self.gwc(x) + self.pwc(x)

# class ConvAdaptPwc(nn.Module):
#     def __init__(self, in_channels, out_channels, p):
#         super(ConvAdaptPwc, self).__init__()
#         # Groupwise Convolution
#         self.gwc = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             groups=int(p / gp),
#             bias=True,
#         )
#         assert in_channels == out_channels
#         # Pointwise Convolution
#         self.pwc = nn.Conv2d(
#             in_channels, out_channels, kernel_size=1, groups=int(p / pt), bias=True
#         )
#         self.pwc.weight.data=torch.eye(in_channels, requires_grad=False).view(in_channels,in_channels,1,1)

#     def forward(self, x):
#         # return self.gwc(x) + self.pwc(x)
#         return self.pwc(x)


class BasicBlockAdpt(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlockAdpt, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.lhc1 = ConvAdapt(out_planes, out_planes, out_planes)  # adapter
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.lhc2 = ConvAdapt(out_planes, out_planes, out_planes)  # adapter
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            # and nn.Sequential(
            and nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                # ),
                # ConvAdapt(
                #     self.expansion * out_planes,
                #     self.expansion * out_planes,
                #     int(self.expansion * out_planes),
                # ),
            )
            or None
        )

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.lhc1(self.conv1(out if self.equalInOut else x))))  # if lhc1
        # out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))  # if no lhc1
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.lhc2(self.conv2(out))
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNetAdpt(nn.Module):
    """Based on code from https://github.com/yaodongyu/TRADES"""

    def __init__(
        self,
        depth=28,
        num_classes=10,
        widen_factor=10,
        sub_block1=False,
        dropRate=0.0,
        bias_last=True,
    ):
        super(WideResNetAdpt, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block_1 = BasicBlockAdpt
        block_2 = BasicBlockAdpt
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block_1, 1, dropRate)
        if sub_block1:
            # 1st sub-block
            self.sub_block1 = NetworkBlock(
                n, nChannels[0], nChannels[1], block_2, 1, dropRate
            )
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block_2, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block_1, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes, bias=bias_last)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and not m.bias is None:
                m.bias.data.zero_()

    def forward(self, x, return_features=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if return_features:
            return self.fc(out), out
        else:
            return self.fc(out)

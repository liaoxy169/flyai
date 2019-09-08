import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from vgg import vgg16

class FCN32s(nn.Module):
    """There are some difference from original fcn"""

    def __init__(self, nclass, backbone='vgg16', aux=False, pretrained_base=True,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super(FCN32s, self).__init__()
        self.aux = aux
        if backbone == 'vgg16':
            self.pretrained = vgg16(pretrained=pretrained_base).features
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.head = _FCNHead(512, nclass, norm_layer)
        if aux:
            self.auxlayer = _FCNHead(512, nclass, norm_layer)

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])

    def forward(self, x):
        size = x.size()[2:]
        pool5 = self.pretrained(x)

        outputs = []
        out = self.head(pool5)
        out = F.interpolate(out, size, mode='bilinear', align_corners=True)
        outputs.append(out)

        if self.aux:
            auxout = self.auxlayer(pool5)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)

        #return tuple(outputs)
        return outputs[0]

class FCN16s(nn.Module):
    def __init__(self, nclass, backbone='vgg16', aux=False, pretrained_base=True, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FCN16s, self).__init__()
        self.aux = aux
        if backbone == 'vgg16':
            self.pretrained = vgg16(pretrained=pretrained_base).features
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.pool4 = nn.Sequential(*self.pretrained[:24])
        self.pool5 = nn.Sequential(*self.pretrained[24:])
        self.head = _FCNHead(512, nclass, norm_layer)
        self.score_pool4 = nn.Conv2d(512, nclass, 1)
        if aux:
            self.auxlayer = _FCNHead(512, nclass, norm_layer)

        self.__setattr__('exclusive', ['head', 'score_pool4', 'auxlayer'] if aux else ['head', 'score_pool4'])

    def forward(self, x):
        pool4 = self.pool4(x)
        pool5 = self.pool5(pool4)

        outputs = []
        score_fr = self.head(pool5)

        score_pool4 = self.score_pool4(pool4)

        upscore2 = F.interpolate(score_fr, score_pool4.size()[2:], mode='bilinear', align_corners=True)
        fuse_pool4 = upscore2 + score_pool4

        out = F.interpolate(fuse_pool4, x.size()[2:], mode='bilinear', align_corners=True)
        outputs.append(out)

        if self.aux:
            auxout = self.auxlayer(pool5)
            auxout = F.interpolate(auxout, x.size()[2:], mode='bilinear', align_corners=True)
            outputs.append(auxout)

        #return tuple(outputs)
        return outputs[0]

class FCN8s(nn.Module):
    def __init__(self, nclass, backbone='vgg16', aux=False, pretrained_base=True, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FCN8s, self).__init__()
        self.aux = aux
        if backbone == 'vgg16':
            self.pretrained = vgg16(pretrained=pretrained_base).features
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.pool3 = nn.Sequential(*self.pretrained[:17])
        self.pool4 = nn.Sequential(*self.pretrained[17:24])
        self.pool5 = nn.Sequential(*self.pretrained[24:])
        self.head = _FCNHead(512, nclass, norm_layer)
        self.score_pool3 = nn.Conv2d(256, nclass, 1)
        self.score_pool4 = nn.Conv2d(512, nclass, 1)
        if aux:
            self.auxlayer = _FCNHead(512, nclass, norm_layer)

        self.__setattr__('exclusive',
                         ['head', 'score_pool3', 'score_pool4', 'auxlayer'] if aux else ['head', 'score_pool3',
                                                                                         'score_pool4'])

    def forward(self, x):
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)

        outputs = []
        score_fr = self.head(pool5)

        score_pool4 = self.score_pool4(pool4)
        score_pool3 = self.score_pool3(pool3)

        upscore2 = F.interpolate(score_fr, score_pool4.size()[2:], mode='bilinear', align_corners=True)
        fuse_pool4 = upscore2 + score_pool4

        upscore_pool4 = F.interpolate(fuse_pool4, score_pool3.size()[2:], mode='bilinear', align_corners=True)
        fuse_pool3 = upscore_pool4 + score_pool3

        out = F.interpolate(fuse_pool3, x.size()[2:], mode='bilinear', align_corners=True)
        outputs.append(out)

        if self.aux:
            auxout = self.auxlayer(pool5)
            auxout = F.interpolate(auxout, x.size()[2:], mode='bilinear', align_corners=True)
            outputs.append(auxout)

        #return tuple(outputs)
        return outputs[0]

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


if __name__ == '__main__':
    import torch.optim as optim
    batch_size, n_class, h, w = 1, 2, 667, 1000
    fcn_model = FCN16s(n_class).to('cuda')
    criterion = nn.BCELoss()
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-3, momentum=0.9)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    input=input.to('cuda')
    y = torch.autograd.Variable(torch.randn(batch_size, n_class, h, w), requires_grad=False).to('cuda')
    for iter in range(10):
        optimizer.zero_grad()
        output = fcn_model(input)
        output = torch.sigmoid(output)
        loss = criterion(output, y)
        loss.backward()
        print("iter{}, loss {}".format(iter, loss.data))
        optimizer.step()


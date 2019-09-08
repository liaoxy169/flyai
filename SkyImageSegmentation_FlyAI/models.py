
from torch import nn
up_kwargs = {'mode': 'bilinear', 'align_corners': True}
class BaseNet(nn.Module):
    def __init__(self, nclass, backbone):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        # copying modules from pretrained models
        if backbone == 'resnet50':
            from torchvision.models import resnet50
            self.pretrained = resnet50(pretrained=True)
        elif backbone == 'resnet101':
            from torchvision.models import resnet101
            self.pretrained = resnet101(pretrained=True)
        elif backbone == 'resnet152':
            from torchvision.models import resnet152
            self.pretrained = resnet152(pretrained=True)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

def base_forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4
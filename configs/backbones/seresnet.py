import torch.utils.checkpoint as cp
from ..common.se_layer import SELayer
from .resnet import Bottleneck, ResLayer, ResNet,BasicBlock


class SEBasicBlock(BasicBlock):
    def __init__(self, in_channels, out_channels, se_ratio=16, **kwargs):
        super(SEBasicBlock, self).__init__(in_channels, out_channels, **kwargs)
        self.se_layer = SELayer(out_channels, ratio=se_ratio)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = self.se_layer(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class SEBottleneck(Bottleneck):

    def __init__(self, in_channels, out_channels, se_ratio=16, **kwargs):
        super(SEBottleneck, self).__init__(in_channels, out_channels, **kwargs)
        self.se_layer = SELayer(out_channels, ratio=se_ratio)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            out = self.se_layer(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class SEResNet(ResNet):

    arch_settings = {
        34: (SEBasicBlock, (3, 4, 6, 3)),
        50: (SEBottleneck, (3, 4, 6, 3)),
        101: (SEBottleneck, (3, 4, 23, 3)),
        152: (SEBottleneck, (3, 8, 36, 3))
    }

    def __init__(self, depth, se_ratio=16, **kwargs):
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for SEResNet')
        self.se_ratio = se_ratio
        super(SEResNet, self).__init__(depth, **kwargs)

    def make_res_layer(self, **kwargs):
        return ResLayer(se_ratio=self.se_ratio, **kwargs)
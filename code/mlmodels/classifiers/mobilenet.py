import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from typing import List, NamedTuple
from torchinfo import summary

MBLOCK_v1 = namedtuple('block', 'in_features,out_features,stride')
ARCH_V1 = [
    MBLOCK_v1(32, 64, 1),
    MBLOCK_v1(64, 128, 2),
    
    MBLOCK_v1(128, 128, 1),
    MBLOCK_v1(128, 256, 2),
    
    MBLOCK_v1(256, 256, 1),
    MBLOCK_v1(256, 512, 2),    
    
    * 5*[MBLOCK_v1(512, 512, 1)],
    
    MBLOCK_v1(512, 1024, 2),
    MBLOCK_v1(1024, 1024, 1),
]

MBLOCK_v2 = namedtuple('block', 'blocks,expansion,out_features,stride')
ARCH_V2 = [
    MBLOCK_v2(1, 1, 16, 1),
    MBLOCK_v2(2, 6, 24, 2),
    MBLOCK_v2(3, 6, 32, 2),
    MBLOCK_v2(4, 6, 64, 2),
    MBLOCK_v2(3, 6, 96, 1),
    MBLOCK_v2(3, 6, 160, 2),
    MBLOCK_v2(1, 6, 320, 1)
]


class DepthwiseConv1Version(nn.Module):
    """
    My way of doing depthwise convolution before I discovered groups parameter of Conv2D module
    """
    def __init__(self, layers: int, stride: int) -> None:
        super().__init__()
        self.layers = layers
        self.depth_convs = [nn.Conv2d(1, 1, kernel_size=3, stride=stride, padding=1) for _ in range(layers)]
        self.depth_bns = [nn.BatchNorm2d(1) for _ in range(layers)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_layers = []
        for f in range(self.layers):
            x_f = x[:,f,:,:].unsqueeze(dim=1)
            x_c = self.depth_convs[f](x_f)
            x_n = self.depth_bns[f](x_c)
            x_d = F.relu(x_n)
            conv_layers.append(x_d)
        x = torch.concat(conv_layers, dim=1)  # out: X, Y, in_features
        return x


class DepthwiseConv(nn.Module):
    def __init__(self, in_features: int, stride: int, activation: nn.Module = nn.ReLU) -> None:
        super().__init__()
        self.depth_conv = nn.Conv2d(in_features, in_features, kernel_size=3, stride=stride, padding=1, groups=in_features)
        self.depth_bn = nn.BatchNorm2d(in_features)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)         # out: X, Y, out_features
        x = self.depth_bn(x)
        x = self.activation(x) # Originally activation was .relu(); .relu6() was introduced in v2
        return x

class PointwiseConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: nn.Module = nn.ReLU) -> None:
        super().__init__()
        self.point_conv = nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, padding=0)
        self.point_bn = nn.BatchNorm2d(out_features)
        self.activation = activation()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.point_conv(x)
        x = self.point_bn(x)
        x = self.activation(x)
        return x


class MobileBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, stride: int) -> None:
        super().__init__()
        self.depth_conv = DepthwiseConv(in_features, stride)
        self.point_conv = PointwiseConv(in_features, out_features)
    def forward(self, x: torch.Tensor):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MobileNetV1(nn.Module):
    def __init__(self, architecture: List[NamedTuple]) -> None:
        super().__init__()        
        self.conv = nn.Conv2d(3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.mobile_blocks = nn.Sequential()
        for i, m_block in enumerate(architecture):
            self.mobile_blocks.add_module(
                f'mb_{i+1}',
                MobileBlock(m_block.in_features, m_block.out_features, stride=m_block.stride)
            )
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=7, padding=3)
        self.fc = nn.Linear(in_features=1024, out_features=1000)
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape == torch.Size([x.shape[0], 3, 224, 224])
        x = self.conv(x) # out: 112x112x32
        x = self.mobile_blocks(x)  # out: 7x7x1024
        x = self.avg_pool(x)    # out: 1x1x1024
        x = torch.flatten(x, start_dim=1)    # out: 1x1024
        x = self.fc(x)          # out: 1x1024
        return self.softmax(x)        

class BottleneckBlock(nn.Module):
    def __init__(self, residual: bool, in_features: int, out_features: int, stride: int, expansion: int) -> None:
        super().__init__()
        self.residual = residual
        self.expansion = nn.Conv2d(in_features, expansion*in_features, kernel_size=1, stride=1, padding=0) 
        self.dwise = DepthwiseConv(expansion*in_features, stride)
        self.pwise = PointwiseConv(expansion*in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expansion(x)   # out: h × w × (tk)
        out = F.relu6(out)
        out = self.dwise(out)   # out: h/s × w/s × (tk)
        out = self.pwise(out)   # out: h/s × w/s × k′
        if self.residual:
            out+= x
        return out


class MobileNetV2(nn.Module):
    def __init__(self, architecture: List[NamedTuple]) -> None:
        super().__init__()
        
        self.input_conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bottleneck_blocks = nn.Sequential()
        
        in_features = 32
        for i, b_block in enumerate(architecture):
            self.b_blocks = nn.Sequential()
            for b in range(1, b_block.blocks+1):
                l = self._make_layer(i, b, b_block.stride, in_features, b_block.out_features, b_block.expansion)
                self.b_blocks.add_module(f'm{b}', l)
                in_features = b_block.out_features
            self.bottleneck_blocks.add_module(f'b{i}', self.b_blocks)
            
            
        self.output_conv = nn.Conv2d(in_features, 1280, kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=7, padding=3)
        self.fc = nn.Linear(in_features=1280, out_features=1000)
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax()

    def _make_layer(self, i: int, b: int, stride: int, in_features: int, out_features: int, expansion: int):
        stride = stride if b == 1 else 1
        residual = True if b > 1 and i !=0 else False        
        layer = BottleneckBlock(residual, in_features, out_features, stride, expansion)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.input_conv(x)
        x = self.bottleneck_blocks(x) # out: 7x7x320
        x = self.output_conv(x) # NO ReLU
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x


if __name__ == '__main__':
    
    # test MobileBlock
    X = torch.rand(32, 32, 112, 112)
    mb_v1 = MobileBlock(32, 32, 1)
    #print(mb_v1)
    out = mb_v1(X)
    assert out.shape == torch.Size([32, 32, 112, 112])

    # Test full model V1
    X = torch.rand(1, 3, 224, 224)
    m = MobileNetV1(ARCH_V1)
    # print(m)
    # summary(m, X.shape)
    out = m(X)
    out
    
    
    # Test BottleNeck
    X = torch.rand(32, 32, 112, 112)
    br_v2 = BottleneckBlock(residual=True, in_features=32, out_features=32, stride=1, expansion=6)
    out = br_v2(X)
    
    X = torch.rand(32, 32, 112, 112)
    b_v2 = BottleneckBlock(residual=False, in_features=32, out_features=64, stride=2, expansion=6)
    out = b_v2(X)
    
    # Test full model V2
    X = torch.rand(1, 3, 224, 224)
    m = MobileNetV2(ARCH_V2)
    # print(m)
    summary(m, X.shape)
    out = m(X)
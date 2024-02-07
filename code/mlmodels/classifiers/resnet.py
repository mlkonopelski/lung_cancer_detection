from collections import namedtuple
from typing import List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

LAYER = namedtuple('arch_type', 'layers, in_features, out_features')
ARCH = {
    '152': [LAYER(3, 64, 256), 
            LAYER(8, 256, 512), 
            LAYER(36, 512, 1024), 
            LAYER(3, 1024, 2048)],
    '34': [LAYER(2, 64, 64), 
           LAYER(2, 64, 128), 
           LAYER(2, 128, 256), 
           LAYER(2, 256, 512)]
}

class PadDownSampler(nn.Module):
    def __init__(self, in_features: int, out_features: int, stride: int) -> None:
        super(PadDownSampler, self).__init__()
        
        self.stride = stride
        self.pad = out_features - in_features        
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor):
        
        if self.stride == 2:
            x = self.max_pool(x)
        x = F.pad(x, pad=(0, 0, 0, 0, 0, self.pad), mode='constant', value=0)
        return x

class ConvDownSampler(nn.Module):
    def __init__(self, in_features: int, out_features: int, stride: int) -> None:     
        super(ConvDownSampler, self).__init__()
        
        self.downsampler = nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, stride=stride)
        self.bn = nn.BatchNorm2d(num_features=out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsampler(x)
        x = self.bn(x)
        return x

class ResNetBasicBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, stride: int, down_sampler = Optional[nn.Module]) -> None:
        super(ResNetBasicBlock, self).__init__()
        
        self.down_sampler = down_sampler
        
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.down_sampler:
            residual = self.down_sampler(x)
        else:
            residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        out += residual
        
        return out
        

class ResNetBottleneckBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, stride: int, down_sampler = Optional[nn.Module]) -> None:
        super(ResNetBottleneckBlock, self).__init__()
        
        self.down_sampler = down_sampler
        
        inter_features = out_features // 4
        
        self.conv1 = nn.Conv2d(in_features, inter_features, kernel_size=1, padding=0, stride=stride)
        self.bn1 = nn.BatchNorm2d(inter_features)
        self.conv2 = nn.Conv2d(inter_features, inter_features, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(inter_features)
        self.conv3 = nn.Conv2d(inter_features, out_features, kernel_size=1, padding=0, stride=1)
        self.bn3 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_sampler:
            residual = self.down_sampler(x)
        else:
            residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
            
        out += residual
        
        return out        
        

class ResNet(nn.Module):
    def __init__(self, layer: List[NamedTuple], resnet_block: nn.Module, down_sampler: nn.Module) -> None:
        """Building Module for one of ResNet architectures. 

        Args:
            layer (List[NamedTuple]): Detailed information about each of layer 2-5 should have. Requiered attributes: layers, in_features, out_features
            resnet_block (nn.Module): ResNet can be build using one of two Residual Blocks: Basic and Bottleneck. Detailed difference between them is in documentation.
            down_sampler (nn.Module): ResNet can be build using one of two Samplers: Parameter free simple sampler and 'learned' using 1x1 convolutions. 
            Samplers are crucial to build "shortcut connection" between input to block and output. 
        """
        super(ResNet, self).__init__()
        
        self.ResnetBlock = resnet_block
        self.DownSampler = down_sampler
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = self._make_layer(layers=layer[0].layers, in_features=layer[0].in_features, out_features=layer[0].out_features, init_downsampling=False)
        self.conv3 = self._make_layer(layers=layer[1].layers, in_features=layer[1].in_features, out_features=layer[1].out_features, init_downsampling=True)
        self.conv4 = self._make_layer(layers=layer[2].layers, in_features=layer[2].in_features, out_features=layer[2].out_features, init_downsampling=True)
        self.conv5 = self._make_layer(layers=layer[3].layers, in_features=layer[3].in_features, out_features=layer[3].out_features, init_downsampling=True)
        
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=7, padding=0)
        self.fc = nn.Linear(layer[3].out_features, 1000)
        self.softmax = nn.Softmax(dim=-1)
        
    def _make_layer(self, layers: int, in_features: int, out_features: int, init_downsampling: bool = True):
        
        conv_layers = []
        
        for l in range(1, layers + 1):
            
            if l == 1 and init_downsampling:
                # CONV3_1 ; CONV4_1 ; CONV5_1
                down_sampler = self.DownSampler(in_features, out_features=out_features, stride=2)
                conv_layers.append(self.ResnetBlock(in_features=in_features, out_features=out_features, stride=2, down_sampler=down_sampler))
            elif l ==1 and not init_downsampling:
                # CONV2_1
                down_sampler = self.DownSampler(in_features, out_features=out_features, stride=1)
                conv_layers.append(self.ResnetBlock(in_features=in_features, out_features=out_features, stride=1, down_sampler=down_sampler))
            else:
                # CONV2_2:CONV2_3 ; CONV3_2:CONV3_8 ; CONV4_2:CONV4_36 ; CONV5_2:CONV5_3
                conv_layers.append(self.ResnetBlock(in_features=out_features, out_features=out_features, stride=1, down_sampler=None))
            
        return nn.Sequential(*conv_layers)


    def forward(self, x: torch.Tensor): 
        
        assert x.shape[1:] == torch.Size([3, 224, 224])
        
        # Output sizes for Resnet152
        x = self.conv1(x)   # out: 64x112x112
        x = self.max_pool(x)   # out: 64x56x56
        
        x = self.conv2(x)   # out: 256x56x56
        x = self.conv3(x)   # out: 512x28x28
        x = self.conv4(x)   # out: 1024x14x14
        x = self.conv5(x)   # out: 2048x7x7
        x = self.avg_pool(x)    # out: 2048x1x1
        x = self.fc(torch.flatten(x))   # out 1x2048

        return F.softmax(x)
    
    
    
if __name__ == '__main__':
    
    
    X = torch.rand(1, 3, 224, 224)
    
    
    resnet152 = ResNet(layer=ARCH['152'],
                       resnet_block=ResNetBottleneckBlock,
                       down_sampler=ConvDownSampler)
    out = resnet152(X)
    
    resnet34 = ResNet(layer=ARCH['34'],
                      resnet_block=ResNetBasicBlock,
                      down_sampler=PadDownSampler)
    out = resnet34(X)

    print(resnet152)
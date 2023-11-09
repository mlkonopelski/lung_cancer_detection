import  torch.nn as nn
import torch
from typing import Tuple
from torch import Tensor
import math 


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, conv_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, 
                               out_channels=conv_channels,
                               kernel_size=3,
                               padding=1,
                               bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels=conv_channels,
                               out_channels=conv_channels,
                               kernel_size=3,
                               padding=1,
                               bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(2, 2)
    
    def forward(self, x: Tensor) -> Tensor:
        out_block = self.relu1(self.conv1(x))
        out_block = self.relu2(self.conv2(out_block))
        out_block = self.max_pool(out_block)
        return out_block


class CNN(nn.Module):
    def __init__(self, in_channels: int = 1, conv_channels: int = 8) -> None:
        super().__init__()
        
        self.batch_norm = nn.BatchNorm3d(1)
        self.block1 = CNNBlock(in_channels=in_channels, conv_channels=conv_channels)
        self.block2 = CNNBlock(in_channels=conv_channels, conv_channels= 2 * conv_channels)
        self.block3 = CNNBlock(in_channels=2 * conv_channels, conv_channels= 4 * conv_channels)
        self.block4 = CNNBlock(in_channels=4 * conv_channels, conv_channels= 8 * conv_channels)
        self.linear = nn.Linear(in_features=1152, out_features=2)
        self.softmax = nn.Softmax(dim=1)
        
    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv3d}:
                nn.init.kaiming_normal_(m.weight.data,
                                        a=0,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    _, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1/ math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.batch_norm(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, self.softmax(out)
    
if __name__ == '__main__':
    
    import numpy as np
    
    DEVICE = 'cpu'
    model = CNN(in_channels=1, conv_channels=8)
    model = model.to(DEVICE)
    example_tensor = torch.from_numpy(np.random.random([10, 1, 32, 48, 48]))
    example_tensor = example_tensor.to(DEVICE, dtype=torch.float32)

    
    # Test model on sample DL:
    from _data import DevClassifierTrainingApp    
    DevClassifierTrainingApp(model).run(epochs=2)
    
    
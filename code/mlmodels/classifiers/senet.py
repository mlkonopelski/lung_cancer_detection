import torch
import torch.nn as nn
import torch.nn.functional as F


class SENet(nn.Module):
    def __init__(self, in_features: int, s_ratio: float, activation: nn.Module = nn.ReLU) -> None:
        super().__init__()
        squeeze_features = max(1, in_features // s_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.squeeze_conv = nn.Conv2d(in_features, squeeze_features, kernel_size=1, padding=0)
        self.expand_conv = nn.Conv2d(squeeze_features, in_features, kernel_size=1, padding=0)
        self.activation = activation()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.avg_pool(x)
        out = self.squeeze_conv(out)
        out = self.activation(out)
        out = self.expand_conv(out)
        out = F.sigmoid(out)

        return out * x



if __name__ == '__main__':
    
    X = torch.rand(1, 64, 32, 32)
    sn = SENet(64, 2)
    print(sn)
    out = sn(X)
    
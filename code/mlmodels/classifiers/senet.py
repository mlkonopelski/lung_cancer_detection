import torch
import torch.nn as nn
import torch.nn.functional as F


class SENet(nn.Module):
    def __init__(self, in_features: int, s_ratio: float) -> None:
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.squeeze_conv = nn.Conv2d(in_features, in_features // s_ratio, kernel_size=1, padding=0)
        self.expand_conv = nn.Conv2d(in_features // s_ratio, in_features, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.avg_pool(x)
        out = self.squeeze_conv(out)
        out = F.relu(out)
        out = self.expand_conv(out)
        out = F.sigmoid(out)

        return out * x



if __name__ == '__main__':
    
    X = torch.rand(1, 64, 32, 32)
    sn = SENet(64, 2)
    print(sn)
    out = sn(X)
    
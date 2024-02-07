import torch
import torch.nn as nn
import torch.nn.functional as F


class SENet(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 s_ratio: float, 
                 activation: nn.Module = nn.ReLU, 
                 conv_block: nn.Module = nn.Conv2d) -> None:
        """Improving the quality of representations produced by a network by explicitly modelling the interdependencies 
        between the channels of its convolutional features. To this end, 
        we propose a mechanism that allows the network to perform feature recalibration, 
        through which it can learn to use global information to selectively emphasise informative features 
        and suppress less useful ones.
        
        The SE block consists of 3 operations:
            1. **Squeeze** operation, which produces a channel descriptor by aggregating feature maps across their spatial dimensions (H × W )
            2. **Excitation** takes the embedding as input and produces a collection of per-channel modulation weights. To limit modelcomplexity 
            and aid generalisation, we parameterise the gating mechanism by forming a bottleneck with two fully-connected (FC) layers 
            around the non-linearity, i.e. a dimensionality-reduction layer with reduction ratio: s_ratio
            3. Scalling by applying (multiplying) weights to original input. 

        Args:
            in_features (int): # features
            s_ratio (float): multiplier in Excitation block to introduce bottleneck block
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU.
            conv_block (nn.Module, optional): Convolutional block. Defaults to nn.Conv2d.
        """
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        excite_features = max(1, in_features // s_ratio)
        self.excite_conv1 = conv_block(in_features, excite_features, kernel_size=1, padding=0)
        self.excite_conv2 = conv_block(excite_features, in_features, kernel_size=1, padding=0)
        self.activation = activation()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 1. Squeeze
        # C X H X W -> C X 1 X 1
        out = self.squeeze(x)
        # 2. Excite
        # C * W X 1 X 1
        out = self.excite_conv1(out)
        out = self.activation(out)
        out = self.excite_conv2(out)
        out = F.sigmoid(out)
        # 3. Scale (Join channels weights with original input)
        return out * x



if __name__ == '__main__':
    
    # 2D Data
    X = torch.rand(1, 64, 32, 32)
    sn = SENet(64, 2)
    print(sn)
    out = sn(X)
    
    # 3D Data
    X = torch.rand(1, 64, 48, 32, 32)
    sn = SENet(64, 2, conv_block=nn.Conv3d)
    out = sn(X)
    print(out.shape)
import torch.nn as nn
import torchvision.transforms.functional as F
import torch 
import torchinfo

class ContractCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv_kernel: int = 3) -> None:
        super(ContractCNN, self).__init__()
        
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=conv_kernel)
        
    def forward(self, x):
        out = self.max_pool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out
    
class ExpandCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, transpose_conv_kernel: int = 2, transpose_conv_stride: int = 2) -> None:
        super(ExpandCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.expand_conv = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=transpose_conv_kernel, stride=transpose_conv_stride)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.expand_conv(out)
        return out



class UNET(nn.Module):
    
    def __init__(self) -> None:
        super(UNET, self).__init__()
        
        self.contract_conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
        )
        self.contract_conv_layer2 = ContractCNN(in_channels=64, out_channels=128) 
        self.contract_conv_layer3 = ContractCNN(in_channels=128, out_channels=256)
        self.contract_conv_layer4 = ContractCNN(in_channels=256, out_channels=512)
        self.contract_conv_layer5 = ContractCNN(in_channels=512, out_channels=1024)
        
        self.expand_conv_layer1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.expand_conv_layer2 = ExpandCNN(in_channels=1024, out_channels=256)
        self.expand_conv_layer3 = ExpandCNN(in_channels=512, out_channels=128)
        self.expand_conv_layer4 = ExpandCNN(in_channels=256, out_channels=64)
        self.expand_conv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)
        )        
        
    def _crop_img(self, tensor, target_tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]
        
    def forward(self, x):
        
        c1 = self.contract_conv_layer1(x) 
        c2 = self.contract_conv_layer2(c1)
        c3 = self.contract_conv_layer3(c2)
        c4 = self.contract_conv_layer4(c3)
        c5 = self.contract_conv_layer5(c4)
        
        e1 = self.expand_conv_layer1(c5)
        c4 = self._crop_img(c4, e1)
        e1 = torch.cat([c4, e1], dim=1)
        
        e2 = self.expand_conv_layer2(e1) 
        c3 = self._crop_img(c3, e2)
        e2 = torch.cat([c3, e2], dim=1)
        
        e3 = self.expand_conv_layer3(e2)
        c2 = self._crop_img(c2, e3)
        e3 = torch.cat([c2, e3], dim=1)
        
        e4 = self.expand_conv_layer4(e3)
        c1 = self._crop_img(c1, e4)
        e4 = torch.cat([c1, e4], dim=1)

        out = self.expand_conv_layer5(e4)

        return out
    

if __name__ == '__main__':
     
    unet = UNET()
    torchinfo.torchinfo.summary(unet, input_size=(1, 1, 572, 572))
    
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# UNET                                     [1, 2, 388, 388]          --
# ├─Sequential: 1-1                        [1, 64, 568, 568]         --
# │    └─Conv2d: 2-1                       [1, 64, 570, 570]         640
# │    └─ReLU: 2-2                         [1, 64, 570, 570]         --
# │    └─Conv2d: 2-3                       [1, 64, 568, 568]         36,928
# ├─ContractCNN: 1-2                       [1, 128, 280, 280]        --
# │    └─MaxPool2d: 2-4                    [1, 64, 284, 284]         --
# │    └─Conv2d: 2-5                       [1, 128, 282, 282]        73,856
# │    └─ReLU: 2-6                         [1, 128, 282, 282]        --
# │    └─Conv2d: 2-7                       [1, 128, 280, 280]        147,584
# │    └─ReLU: 2-8                         [1, 128, 280, 280]        --
# ├─ContractCNN: 1-3                       [1, 256, 136, 136]        --
# │    └─MaxPool2d: 2-9                    [1, 128, 140, 140]        --
# │    └─Conv2d: 2-10                      [1, 256, 138, 138]        295,168
# │    └─ReLU: 2-11                        [1, 256, 138, 138]        --
# │    └─Conv2d: 2-12                      [1, 256, 136, 136]        590,080
# │    └─ReLU: 2-13                        [1, 256, 136, 136]        --
# ├─ContractCNN: 1-4                       [1, 512, 64, 64]          --
# │    └─MaxPool2d: 2-14                   [1, 256, 68, 68]          --
# │    └─Conv2d: 2-15                      [1, 512, 66, 66]          1,180,160
# │    └─ReLU: 2-16                        [1, 512, 66, 66]          --
# │    └─Conv2d: 2-17                      [1, 512, 64, 64]          2,359,808
# │    └─ReLU: 2-18                        [1, 512, 64, 64]          --
# ├─ContractCNN: 1-5                       [1, 1024, 28, 28]         --
# │    └─MaxPool2d: 2-19                   [1, 512, 32, 32]          --
# │    └─Conv2d: 2-20                      [1, 1024, 30, 30]         4,719,616
# │    └─ReLU: 2-21                        [1, 1024, 30, 30]         --
# │    └─Conv2d: 2-22                      [1, 1024, 28, 28]         9,438,208
# │    └─ReLU: 2-23                        [1, 1024, 28, 28]         --
# ├─ConvTranspose2d: 1-6                   [1, 512, 56, 56]          2,097,664
# ├─ExpandCNN: 1-7                         [1, 256, 104, 104]        --
# │    └─Conv2d: 2-24                      [1, 256, 54, 54]          2,359,552
# │    └─ReLU: 2-25                        [1, 256, 54, 54]          --
# │    └─Conv2d: 2-26                      [1, 256, 52, 52]          590,080
# │    └─ConvTranspose2d: 2-27             [1, 256, 104, 104]        262,400
# ├─ExpandCNN: 1-8                         [1, 128, 200, 200]        --
# │    └─Conv2d: 2-28                      [1, 128, 102, 102]        589,952
# │    └─ReLU: 2-29                        [1, 128, 102, 102]        --
# │    └─Conv2d: 2-30                      [1, 128, 100, 100]        147,584
# │    └─ConvTranspose2d: 2-31             [1, 128, 200, 200]        65,664
# ├─ExpandCNN: 1-9                         [1, 64, 392, 392]         --
# │    └─Conv2d: 2-32                      [1, 64, 198, 198]         147,520
# │    └─ReLU: 2-33                        [1, 64, 198, 198]         --
# │    └─Conv2d: 2-34                      [1, 64, 196, 196]         36,928
# │    └─ConvTranspose2d: 2-35             [1, 64, 392, 392]         16,448
# ├─Sequential: 1-10                       [1, 2, 388, 388]          --
# │    └─Conv2d: 2-36                      [1, 64, 390, 390]         73,792
# │    └─Conv2d: 2-37                      [1, 64, 388, 388]         36,928
# │    └─Conv2d: 2-38                      [1, 2, 388, 388]          130
# ==========================================================================================
# Total params: 25,266,690
# Trainable params: 25,266,690
# Non-trainable params: 0
# Total mult-adds (G): 127.22
# ==========================================================================================
# Input size (MB): 1.31
# Forward/backward pass size (MB): 1002.68
# Params size (MB): 101.07
# Estimated Total Size (MB): 1105.06
# ==========================================================================================
    
    
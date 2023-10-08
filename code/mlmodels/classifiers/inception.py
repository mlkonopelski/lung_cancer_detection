import torch
import torch.nn as nn
from torchinfo import summary

class InceptionBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 b1_conv_out_chanels: int,
                 b2_r_conv_out_chanels: int,
                 b2_conv_out_chanels: int,
                 b3_r_conv_out_channels: int, 
                 b3_conv_out_channels: int,
                 b4_conv_out_channels: int
                 ) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=b1_conv_out_chanels ,kernel_size=1, padding=0, stride=1),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=b2_r_conv_out_chanels, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=b2_r_conv_out_chanels, out_channels=b2_conv_out_chanels, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=b3_r_conv_out_channels, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=b3_r_conv_out_channels, out_channels=b3_conv_out_channels, kernel_size=5, padding=2, stride=1),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(in_channels=in_channels, out_channels=b4_conv_out_channels, kernel_size=1, padding=0),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)


class InceptionV1(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000) -> None:
        super().__init__()
        self.num_classes = num_classes
        
        self.fc_input = self._init_fc_input(in_channels)
        self.inc_3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32) # out: 28 x 28 x 256 
        self.inc_3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64) # out: 28 x 28 x 480
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # out: 14 x 14 x 480
        self.inc_4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64) # out: 14 x 14 x 512
        self.fc_intermediate_1 = self._init_fc_intermediate(512)
        self.inc_4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64) # out: 14 x 14 x 512
        self.inc_4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64) # out: 14 x 14 x 512
        self.inc_4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64) # out: 14 x 14 x 528
        self.fc_intermediate_2 = self._init_fc_intermediate(528)
        self.inc_4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128) # out: 14 x 14 x 832
        self.max_pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # out: 7 x 7 x 832
        self.inc_5a =  InceptionBlock(832, 256, 160, 320, 32, 128, 128) # out: 7 x 7 x 832
        self.inc_5b =  InceptionBlock(832, 384, 192, 384, 48, 128, 128) # out: 7 x 7 x 1024
        self.fc_output = self._init_fc_output(1024)  # out 1 x 1 x 1000
        
    def _init_fc_intermediate(self, in_channels: int):
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3), # out: 4 x 4 x in_channels (512 for fc_intermediate_1)
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=4 * 4 * 128, out_features=1024),  # out: 1 x 1000,
            nn.ReLU(),
            nn.Dropout(p=0.7), 
            nn.Linear(in_features=1024, out_features=self.num_classes),  # out: 1 x 1000
            nn.Softmax()
        )
        
    def _init_fc_output(self, in_channels: int):
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1), # out: 1 x 1 x in_channels
            nn.Flatten(),
            nn.Dropout(p=0.4), 
            nn.Linear(in_features=in_channels, out_features=self.num_classes),
            nn.ReLU(),
            nn.Softmax()
        )
        
    def _init_fc_input(self, in_channels):
        return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3, stride=2), # out: 112 x 112 x 64
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, padding=1, stride=2), # out: 56 x 56 x 64
        nn.LocalResponseNorm(size=64),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1), # out: 56 x 56 x 64
        # nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=1), # out 56 x 56 x 192
        nn.ReLU(),
        nn.LocalResponseNorm(size=192),
        nn.MaxPool2d(kernel_size=3, padding=1, stride=2), # out: 28 x 28 x 192
    )


    def forward(self, x):
        fc_input = self.fc_input(x) # out: 28 x 28, 192
        inc_3a = self.inc_3a(fc_input) # out: 28 x 28 x 256 
        inc_3b = self.inc_3b(inc_3a) # out: 28 x 28 x 480
        max_pool3 = self.max_pool3(inc_3b) # out: 14 x 14 x 480
        inc_4a = self.inc_4a(max_pool3) # out: 14 x 14 x 512
        fc_intermediate_1 = self.fc_intermediate_1(inc_4a)
        inc_4b = self.inc_4b(inc_4a) # out: 14 x 14 x 512
        inc_4c = self.inc_4c(inc_4b) # out: 14 x 14 x 512
        inc_4d = self.inc_4d(inc_4c) # out: 14 x 14 x 528
        fc_intermediate_2 = self.fc_intermediate_2(inc_4d)
        inc_4e = self.inc_4e(inc_4d) # out: 14 x 14 x 832
        max_pool4 = self.max_pool4(inc_4e) # out: 7 x 7 x 832
        inc_5a =  self.inc_5a(max_pool4) # out: 7 x 7 x 832
        inc_5b =  self.inc_5b(inc_5a) # out: 7 x 7 x 1024
        fc_output = self.fc_output(inc_5b) # out 1 x 1 x 1000
        
        return fc_intermediate_1, fc_intermediate_2, fc_output
    
    
if __name__ == '__main__':
    
    X = torch.rand(1, 3, 224, 224)
    
    inception_v1 = InceptionV1()
    inception_v1(X)
    
    summary(model=inception_v1, input_size=((1, 3, 224, 224)))
    
    # ==========================================================================================
    # Total params: 13,378,280
    # Trainable params: 13,378,280
    # Non-trainable params: 0
    # Total mult-adds (G): 1.59
    # ==========================================================================================
    # Input size (MB): 0.60
    # Forward/backward pass size (MB): 25.88
    # Params size (MB): 53.51
    # Estimated Total Size (MB): 80.00
    # ==========================================================================================
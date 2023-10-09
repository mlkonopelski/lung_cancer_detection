import torch
import torch.nn as nn
from torchinfo import summary
from typing import Any, Callable, List, Optional, Tuple, Union


class InceptionV1Block(nn.Module):
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
        self.inc_3a = InceptionV1Block(192, 64, 96, 128, 16, 32, 32) # out: 28 x 28 x 256 
        self.inc_3b = InceptionV1Block(256, 128, 128, 192, 32, 96, 64) # out: 28 x 28 x 480
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # out: 14 x 14 x 480
        self.inc_4a = InceptionV1Block(480, 192, 96, 208, 16, 48, 64) # out: 14 x 14 x 512
        self.fc_intermediate_1 = self._init_fc_intermediate(512)
        self.inc_4b = InceptionV1Block(512, 160, 112, 224, 24, 64, 64) # out: 14 x 14 x 512
        self.inc_4c = InceptionV1Block(512, 128, 128, 256, 24, 64, 64) # out: 14 x 14 x 512
        self.inc_4d = InceptionV1Block(512, 112, 144, 288, 32, 64, 64) # out: 14 x 14 x 528
        self.fc_intermediate_2 = self._init_fc_intermediate(528)
        self.inc_4e = InceptionV1Block(528, 256, 160, 320, 32, 128, 128) # out: 14 x 14 x 832
        self.max_pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # out: 7 x 7 x 832
        self.inc_5a =  InceptionV1Block(832, 256, 160, 320, 32, 128, 128) # out: 7 x 7 x 832
        self.inc_5b =  InceptionV1Block(832, 384, 192, 384, 48, 128, 128) # out: 7 x 7 x 1024
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
    
# --------------------------------------------------------------------------------
#                               INCEPTION V2
# --------------------------------------------------------------------------------
class InceptionV2ConvBlock(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 kernel: Union[Tuple[int], int], 
                 stride: int, 
                 padding: Union[Tuple[int], int]) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features, 
                              out_channels=out_features,
                              kernel_size=kernel, 
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_features, eps=0.001)
        self.relu = nn.ReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class InceptionV2FCInput(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super(InceptionV2FCInput, self).__init__()
        self.conv1 =  InceptionV2ConvBlock(in_channels, out_features=32, kernel=(3, 3), stride=2, padding=0) #out: 149 x 149 x 32
        self.conv2 = InceptionV2ConvBlock(32, out_features=32, kernel=(3, 3), stride=1, padding=0) # out: 147 x 147 x 32
        self.conv3 = InceptionV2ConvBlock(in_features=32, out_features=64, kernel=(3, 3), stride=1, padding=1) # out: 147 x 147 x 32
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0) # out: 73 x 73 x 64
        self.conv4 = InceptionV2ConvBlock(in_features=64, out_features=80, kernel=(3, 3), stride=1, padding=0) # out: 71 x 71 x 80
        self.conv5 = InceptionV2ConvBlock(in_features=80, out_features=192, kernel=(3, 3), stride=1, padding=1) # out: 71 x 71 x 192
        self.conv6 = InceptionV2ConvBlock(in_features=192, out_features=288, kernel=(3, 3), stride=2, padding=0) # out: 35 x 35 x 288
        
    def test_dimensions(self):
        x_test = torch.rand(1, 3, 299, 299)
        output = self.forward(x_test)
        assert output.shape== torch.Size([1, 288, 35, 35])    
        
    def forward(self, x):
        assert x.shape[2: 4] == torch.Size([299, 299])
        x = self.conv1(x) #out: 149 x 149 x 32
        x = self.conv2(x) # out: 147 x 147 x 32
        x = self.conv3(x) # out: 147 x 147 x 64
        x = self.max_pool(x) # out: 73 x 73 x 64
        x = self.conv4(x) # out: 71 x 71 x 80
        x = self.conv5(x) # out: 71 x 71 x 192
        x = self.conv6(x) # out: 35 x 35 x 288

        return x


class InceptionV2SizeReduction(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(InceptionV2SizeReduction, self).__init__()
        conv_1x1_features = int(in_features/8)
        conv_3x3_features = int(in_features/4)
        conv_features = int((out_features - in_features) / 2)
        self.branch1 = nn.Sequential(
            InceptionV2ConvBlock(in_features=in_features, out_features=conv_1x1_features, kernel=(1,1), stride=1, padding=0),
            InceptionV2ConvBlock(in_features=conv_1x1_features, out_features=conv_3x3_features, kernel=(3,3), stride=1, padding=1),
            InceptionV2ConvBlock(in_features=conv_3x3_features, out_features=conv_features, kernel=(3,3), stride=2, padding=0),
        )
        self.branch2 = nn.Sequential(
            InceptionV2ConvBlock(in_features=in_features, out_features=conv_1x1_features, kernel=(1,1), stride=1, padding=0),
            InceptionV2ConvBlock(in_features=conv_1x1_features, out_features=conv_features, kernel=(3,3), stride=2, padding=0),
        )
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def test_dimensions(self, input_shape: Tuple[int], output_shape: Tuple[int]):
        x_test = torch.rand(*input_shape)
        output = self.forward(x_test)
        assert output.shape == torch.Size(output_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim=1)
        

class InceptionV2BlockF5(nn.Module):
    '''
    The name InceptionV2Block5 comes from original paper and Inception block showed on fig. 5
    '''
    def __init__(self) -> None:
        super(InceptionV2BlockF5, self).__init__()
        in_channels = 288
        self.branch1 = nn.Sequential(
            InceptionV2ConvBlock(in_features=in_channels, out_features=64, kernel=1, stride=1, padding=0),
            InceptionV2ConvBlock(in_features=64, out_features=96, kernel=3, stride=1, padding=1),
            InceptionV2ConvBlock(in_features=96, out_features=96, kernel=3, stride=1, padding=1)
        ) 
        self.branch2 = nn.Sequential(
            InceptionV2ConvBlock(in_features=in_channels, out_features=48, kernel=1, stride=1, padding=0),
            InceptionV2ConvBlock(in_features=48, out_features=64, kernel=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            InceptionV2ConvBlock(in_features=in_channels, out_features=64, kernel=1, stride=1, padding=0),
        )
        self.branch4 = InceptionV2ConvBlock(in_features=in_channels, out_features=64, kernel=1, stride=1, padding=0)
        
    def test_dimensions(self):
        x_test = torch.rand(1, 288, 35, 35)
        output = self.forward(x_test)
        assert output.shape== torch.Size([1, 288, 35, 35])    
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        return torch.cat([b1, b2, b3, b4], dim=1) # out: 35 x 35 x 288


class InceptionV2BlockF6(nn.Module):
    '''
    The name InceptionV2Block6 comes from original paper and Inception block showed on fig. 6
    '''
    def __init__(self) -> None:
        super(InceptionV2BlockF6, self).__init__()
        in_channels = 768
        self.branch1 = nn.Sequential(
            InceptionV2ConvBlock(in_features=in_channels, out_features=128, kernel=1, stride=1, padding=0),
            InceptionV2ConvBlock(in_features=128, out_features=256, kernel=(1,7), stride=1, padding=(0,3)),
            InceptionV2ConvBlock(in_features=256, out_features=256, kernel=(7, 1), stride=1, padding=(3,0)),
            InceptionV2ConvBlock(in_features=256, out_features=512, kernel=(1,7), stride=1, padding=(0,3)),
            InceptionV2ConvBlock(in_features=512, out_features=512, kernel=(7, 1), stride=1, padding=(3,0)),
        )
        
        self.branch2 = nn.Sequential(
            InceptionV2ConvBlock(in_features=in_channels, out_features=64, kernel=(1,1), stride=1, padding=0),
            InceptionV2ConvBlock(in_features=64, out_features=128, kernel=(1,7), stride=1, padding=(0, 3)),
            InceptionV2ConvBlock(in_features=128, out_features=128, kernel=(7,1), stride=1, padding=(3, 0)),
        )
        
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            InceptionV2ConvBlock(in_features=in_channels, out_features=64, kernel=(1,1), stride=1, padding=0),
        )
                
        self.branch4 = InceptionV2ConvBlock(in_features=in_channels, out_features=64, kernel=(1,1), stride=1, padding=0)
    
    def test_dimensions(self):
        x_test = torch.rand(1, 768, 17, 17)
        output = self.forward(x_test)
        assert output.shape== torch.Size([1, 768, 17, 17])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1) # out: 17 x 17 x 768

class InceptionV2BlockF7(nn.Module):
    '''
    The name InceptionV2Block7 comes from original paper and Inception block showed on fig. 7
    '''
    def __init__(self, in_features: int) -> None:
        super(InceptionV2BlockF7, self).__init__()

        branch1_out_features = 768
        branch2_out_features = 768
        branch3_out_features = 256
        branch4_out_features = 256
        
        
        self.branch1 = nn.Sequential(
            InceptionV2ConvBlock(in_features=in_features, out_features=128, kernel=1, stride=1, padding=0),
            InceptionV2ConvBlock(in_features=128, out_features=256, kernel=3, stride=1, padding=1),
        )
        self.branch1_1 = InceptionV2ConvBlock(in_features=256, out_features=int(branch1_out_features/2), kernel=(1,3), stride=1, padding=(0,1))
        self.branch1_2 = InceptionV2ConvBlock(in_features=256, out_features=int(branch1_out_features/2), kernel=(3,1), stride=1, padding=(1,0))
        
        self.branch2 = InceptionV2ConvBlock(in_features=in_features, out_features=256, kernel=1, stride=1, padding=0)
        self.branch2_1 = InceptionV2ConvBlock(in_features=256, out_features=int(branch2_out_features/2), kernel=(1,3), stride=1, padding=(0,1))
        self.branch2_2 = InceptionV2ConvBlock(in_features=256, out_features=int(branch2_out_features/2), kernel=(3,1), stride=1, padding=(1,0))
        
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            InceptionV2ConvBlock(in_features=in_features, out_features=branch3_out_features, kernel=(1,1), stride=1, padding=0), 
        )
        
        self.branch4 = InceptionV2ConvBlock(in_features=in_features, out_features=branch4_out_features, kernel=(1,1), stride=1, padding=0)
    
    def test_dimensions(self, in_features: int):
        x_test = torch.rand(1, in_features, 17, 17)
        output = self.forward(x_test)
        assert output.shape == torch.Size([1, 2048, 17, 17])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b1_1 = self.branch1_1(b1)
        b1_2 = self.branch1_2(b1)
        b2 = self.branch2(x)
        b2_1 = self.branch2_1(b2)
        b2_2 = self.branch2_2(b2)        
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1_1, b1_2, b2_1, b2_2, b3, b4], dim=1)


class InceptionV2FCAuxilaryOutput(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1 = nn.Conv2d(768, 128, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(num_features=128)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.ln1 = nn.Linear(3200, 1024)
        self.dropout = nn.Dropout(0.7)
        self.ln2 = nn.Linear(1024, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.ln1(x)
        x = self.dropout(x)
        x = self.ln2(x)

        return x
        
class InceptionV2FCOuput(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(InceptionV2FCOuput, self).__init__()
        
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=8)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.ln1 = nn.Linear(in_features=2048, out_features=num_classes)
        self.dropout = nn.Dropout(0.4)
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.ln1(x)
        x = self.dropout(x)
        x = self.softmax(x)
        return x

class InceptionV2(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000) -> None:
        super(InceptionV2, self).__init__()
        self.in_channels = in_channels
        
        self.fc_input = InceptionV2FCInput(in_channels)
        self.blocks_f5 = nn.Sequential(*[InceptionV2BlockF5() for _ in range(3)])
        self.size_reduction_f5 = InceptionV2SizeReduction(in_features=288, out_features=768)
        self.block_f6 =  nn.Sequential(*[InceptionV2BlockF6() for _ in range(5)])
        self.size_reduction_f6 = InceptionV2SizeReduction(in_features=768, out_features=1280)
        self.auxilary_fc_output = InceptionV2FCAuxilaryOutput(num_classes=num_classes)
        self.block_f7 = nn.Sequential(
            InceptionV2BlockF7(1280),
            InceptionV2BlockF7(2048),
        )
        self.fc_output = InceptionV2FCOuput(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.fc_input(x)
        x = self.blocks_f5(x)
        x = self.size_reduction_f5(x)
        x =  self.block_f6(x)
        if self.training:
            aux = self.auxilary_fc_output(x)
        else:
            aux = None
        x = self.size_reduction_f6(x)
        x = self.block_f7(x)
        x = self.fc_output(x)
        
        return x# torchinfo cannot handle double output, aux

# --------------------------------------------------------------------------------
#                               INCEPTION V2
# --------------------------------------------------------------------------------
    
if __name__ == '__main__':
    
    X = torch.rand(1, 3, 224, 224)
    
    inception_v1 = InceptionV1()
    print(inception_v1)
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
    
    # For test purpose check first dimensions of each block
    m_input = InceptionV2FCInput()
    m_input.test_dimensions()
    
    m_block_f5 = InceptionV2BlockF5()
    m_block_f5.test_dimensions()
    
    m_size_reduction = InceptionV2SizeReduction(in_features=288, out_features=768)
    m_size_reduction.test_dimensions(input_shape=(1, 288, 35, 35), output_shape=(1, 768, 17, 17))
    
    m_block_f6 = InceptionV2BlockF6()
    m_block_f6.test_dimensions()
    
    m_size_reduction = InceptionV2SizeReduction(in_features=768, out_features=1280)
    m_size_reduction.test_dimensions(input_shape=(1, 768, 17, 17), output_shape=(1, 1280, 8, 8))
    
    m_block_f7 = InceptionV2BlockF7(in_features=1280)
    m_block_f7.test_dimensions(in_features=1280)
    m_block_f7 = InceptionV2BlockF7(in_features=2048)
    m_block_f7.test_dimensions(in_features=2048)
    
    # Summary
    inception_v2 = InceptionV2()
    print(inception_v2)
    summary(model=inception_v2, input_size=((1, 3, 299, 299)))
    
    # ====================================================================================================
    # Total params: 34,402,336
    # Trainable params: 34,402,336
    # Non-trainable params: 0
    # Total mult-adds (G): 9.24
    # ====================================================================================================
    # Input size (MB): 1.07
    # Forward/backward pass size (MB): 162.99
    # Params size (MB): 120.00
    # Estimated Total Size (MB): 284.07
    # ====================================================================================================
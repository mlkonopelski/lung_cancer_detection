import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

c3_mapping = {'c3_0':[3, [0, 1, 2]],
              'c3_1':[3, [1, 2, 3]],
              'c3_2':[3, [2, 3, 4]],
              'c3_3':[3, [3, 4, 5]],
              'c3_4':[3, [0, 4, 5]],
              'c3_5':[3, [0, 1, 5]],
              'c3_6':[4, [0, 1, 2, 3]],
              'c3_7':[4, [1, 2, 3, 4]],
              'c3_8':[4, [2, 3, 4, 5]],
              'c3_9':[4, [0, 3, 4, 5]],
              'c3_10':[4, [0, 1, 4, 5]],
              'c3_11':[4, [0, 1, 2, 5]],
              'c3_12':[4, [0, 1, 3, 4]],
              'c3_13':[4, [1, 2, 4, 5]],
              'c3_14':[4, [0, 2, 3, 5]],
              'c3_15':[6, [0, 1, 2, 3, 4, 5]]}

class LeNet5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        for f_name, (out_features, _) in c3_mapping.items():
            setattr(self, f_name, nn.Conv2d(in_channels=out_features, out_channels=1, kernel_size=5, stride=1, padding=0))

        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, padding=0)
        self.flatten = nn.Flatten(start_dim=0, end_dim=2)
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.output = nn.Linear(in_features=84, out_features=10)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        c1 = F.tanh(self.c1(x))
        s2 = self.s2(c1)
        
        c3_f_list = []
        for f_name, (_, f_slice) in c3_mapping.items():
            c3_f = getattr(self, f_name)(s2[:, f_slice])
            c3_f_list.append(c3_f)
        c3 = torch.concat(c3_f_list, dim=1)
        c3 = F.tanh(c3)
        
        s4 = self.s4(c3)
        
        c5 = F.tanh(self.c5(s4))
        c5  = self.flatten(c5)
        
        f6 = F.tanh(self.f6(c5))
        
        return self.output(f6)
        

class LunaLenet(LeNet5):
    def __init__(self) -> None:
        super().__init__()
        
        self.c1 = nn.Conv3d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.s2 = nn.AvgPool3d(kernel_size=2, stride=2)
        
        for f_name, (out_features, _) in c3_mapping.items():
            setattr(self, f_name, nn.Conv3d(in_channels=out_features, out_channels=1, kernel_size=5, stride=1, padding=0))

        self.s4 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.c5 = nn.Conv3d(in_channels=16, out_channels=120, kernel_size=5, padding=0)
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(in_features=3000, out_features=84)
        
        self.output = nn.Sequential(
            nn.Linear(in_features=84, out_features=2),
            nn.Softmax(dim=1)
        )

if __name__ == '__main__':
    
    # lenet5 = LeNet5()
    # summary(lenet5, input_size=(1, 32, 32))
    
    luna_lenet5 = LunaLenet()    
    summary(luna_lenet5, input_size=(1, 1, 32, 48, 48))

import math
from collections import namedtuple
from typing import Union, List, NamedTuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torchvision.ops.stochastic_depth import StochasticDepth

import sys
sys.path.append('code/mlmodels/classifiers')
from mobilenet import DepthwiseConv, PointwiseConv
from senet import SENet

STAGE = namedtuple('block', 'layers,out_features,stride,kernel,expansion')

STAGES_V1 = [
    STAGE(1, 16, 1, 3, 1),
    STAGE(2, 24, 2, 3, 6),
    STAGE(2, 40, 2, 5, 6),
    STAGE(3, 80, 2, 3, 6),
    STAGE(3, 112, 1, 5, 6),
    STAGE(4, 192, 2, 5, 6),
    STAGE(1, 320, 1, 6, 6),
]

scaling = namedtuple('v', 'depth_multiplier,width_multiplier,dropout')

SCALLING = {
    'en-b0': scaling(1.0, 1.0, 0.2),
    'en-b1': scaling(1.0, 1.1, 0.2),
    'en-b2': scaling(1.1, 1.2, 0.2),
    'en-b3': scaling(1.2, 1.4, 0.2),
    'en-b4': scaling(1.4, 1.8, 0.2),
    'en-b5': scaling(1.6, 2.2, 0.2),
    'en-b6': scaling(1.8, 2.6, 0.2),
    'en-b7': scaling(2.0, 3.1, 0.2),
}


class MBConv(nn.Module):
    def __init__(self, residual: bool, in_features: int, out_features: int, stride: int, expansion: int, conv_block: nn.Module, bn_block: nn.Module) -> None:
        super().__init__()
        
        self.activation = nn.SiLU
        self.residual = residual
        self.expansion = conv_block(in_features, expansion*in_features, kernel_size=1, stride=1, padding=0) 
        self.dwise = DepthwiseConv(expansion*in_features, stride, self.activation, conv_block=conv_block, bn_block=bn_block)
        self.se = SENet(expansion*in_features, s_ratio=4, activation=self.activation, conv_block=conv_block)
        self.pwise = PointwiseConv(expansion*in_features, out_features, self.activation, conv_block=conv_block, bn_block=bn_block)
        self.stochastic_depth = StochasticDepth(p=0.2, mode='row')

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.expansion(x)   # out: h × w × (tk)
        out = self.activation()(out)
        out = self.dwise(out)   # out: h/s × w/s × (tk)
        out = self.se(out)      # out: h/s × w/s × (tk)
        out = self.pwise(out)   # out: h/s × w/s × k′
        if self.residual:
            out = self.stochastic_depth(out)
            out += x
        return out


class EfficientNet(nn.Module):
    def __init__(self, 
                 architecture: Union[List[NamedTuple], List[Callable]], 
                 scalling: List[NamedTuple],
                 dim: str = '2d', 
                 img_channels: int = 1, 
                 labels: int = 2) -> None:
        """_summary_

        Args:
            architecture (Union[List[NamedTuple], List[Callable]]): _description_
            scalling (List[NamedTuple]): _description_
            dim (str, optional): _description_. Defaults to '2d'.
            img_channels (int, optional): _description_. Defaults to 1.
            labels (int, optional): _description_. Defaults to 2.
        """
        super().__init__()
        
        #assert dim not in ['2d', '3d'],  ("Input data only in ['2d', '3d']")
        
        if dim in '2d':
            self.conv_block = nn.Conv2d
            self.bn_block = nn.BatchNorm2d
            self.gl_pool = nn.AdaptiveAvgPool2d
        else:
            self.conv_block = nn.Conv3d
            self.bn_block = nn.BatchNorm3d
            self.gl_pool = nn.AdaptiveAvgPool3d
            
        
        architecture = self._adjust_architecture(architecture, scalling)
        
        self.activation = nn.SiLU()
        self.stages = nn.Sequential()
        
        stage1 = nn.Sequential(
            self.conv_block(img_channels, 32, kernel_size=3, stride=2, padding=1),
            self.bn_block(32),
            self.activation
        )
        self.stages.add_module( 'stage:1', stage1)

        in_features = 32
        for s, stage in enumerate(architecture):
            layer = nn.Sequential()
            for l in  range(1, stage.layers+1): 
                block = self._make_block(s, l, in_features, stage.out_features, stage.stride, stage.expansion)
                layer.add_module(f'l:{l}', block)
                in_features = stage.out_features
            self.stages.add_module(f'stage:{s+2}', layer)  # MBConv starts from 2nd stage in paper architecture

        stage9 = nn.Sequential(
            self.conv_block(in_features, 1280, kernel_size=1, stride=1, padding=3),
            self.activation,
            self.gl_pool(1),
        )
        self.stages.add_module('stage:9', stage9)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(scalling.dropout, inplace=True),
            nn.Linear(in_features=1280, out_features=labels)
        )

    def _adjust_architecture(self, arch, scalling):

        width_m = scalling.width_multiplier
        depth_m = scalling.depth_multiplier

        new_arch = []
        for stage in arch:
            layers = int(math.ceil(stage.layers * depth_m))
            out_features = int((stage.out_features * width_m // 8) * 8)
            new_arch.append(
                STAGE(layers,
                      out_features,
                      stage.stride,
                      stage.kernel,
                      stage.expansion)
            )
            
        return new_arch

    def _make_block(self, s: int, l: int, in_features: int, out_features: int, stride: int, expansion: int):

        stride = stride if l == 1 else 1 # downsampling only during first layer of each stage
        residual = True if l > 1 and s !=0 else False        

        return MBConv(residual, in_features, out_features, stride, expansion, self.conv_block, self.bn_block)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stages(x)
        logits = self.classifier(out)
        preds = F.softmax(logits, dim=1)
        return logits, preds
        
if __name__ == '__main__':
    
    # EF 2D
    X = torch.rand(1, 3, 224, 224)
    en_b0_2d = EfficientNet(STAGES_V1, SCALLING['en-b0'], dim='2d', img_channels=3)
    en_b0_2d(X)
    summary(en_b0_2d, X.shape)
    
    
    # TEST ON LUNA SAMPLE DATASET
    from _data import DevClassifierTrainingApp
    
    # EF B0 3D
    X = torch.rand(32, 1, 32, 48, 48)
    en_b0 = EfficientNet(STAGES_V1, SCALLING['en-b0'], dim='3d')
    logits, preds = en_b0(X)
    
    DevClassifierTrainingApp(en_b0, 
                            optimizer=torch.optim.SGD(params=en_b0.parameters(), lr=0.01, momentum=0.9)
                            ).run(epochs=1000)
    
    
    # EF b7 3D
    X = torch.rand(32, 1, 32, 48, 48)
    en_b7 = EfficientNet(STAGES_V1, SCALLING['en-b7'], dim='3d')
    logits, preds = en_b7(X)
    DevClassifierTrainingApp(en_b7, 
                        optimizer=torch.optim.Adam(params=en_b0.parameters(), lr=0.01)
                        ).run(epochs=1000)


# epoch: 1 | training time:  0:00:07.202790
#  train: loss: 0.6945441365242004 | acc: 0.515625
#  val: loss: 0.6936302781105042 | acc: 0.5 
# ---------- 
# epoch: 10 | training time:  0:01:07.533285
#  train: loss: 0.693400502204895 | acc: 0.4375
#  val: loss: 0.6931482553482056 | acc: 0.5 
# ----------
# epoch: 50 | training time:  0:05:46.100596
#  train: loss: 0.6928103566169739 | acc: 0.53125
#  val: loss: 0.6932681202888489 | acc: 0.5 
# ----------
# epoch: 100 | training time:  0:11:34.921783
#  train: loss: 0.6937776207923889 | acc: 0.5
#  val: loss: 0.6930646896362305 | acc: 0.515625 
# ----------
# epoch: 150 | training time:  0:17:25.586028
#  train: loss: 0.6922886371612549 | acc: 0.578125
#  val: loss: 0.6929880976676941 | acc: 0.5 
# ----------
# epoch: 300 | training time:  0:35:02.506843
#  train: loss: 0.6928825974464417 | acc: 0.4375
#  val: loss: 0.6927729845046997 | acc: 0.5625 
# ----------
# epoch: 350 | training time:  0:40:55.166191
#  train: loss: 0.6877304911613464 | acc: 0.59375
#  val: loss: 0.6923332214355469 | acc: 0.53125 
# ----------
# epoch: 400 | training time:  0:46:48.268980
#  train: loss: 0.6825998425483704 | acc: 0.65625
#  val: loss: 0.6916866898536682 | acc: 0.546875 
# ----------
# epoch: 450 | training time:  0:52:42.000922
#  train: loss: 0.6758546829223633 | acc: 0.6875
#  val: loss: 0.6931211948394775 | acc: 0.5 
# ----------
# epoch: 500 | training time:  0:58:34.667981
#  train: loss: 0.6673060655593872 | acc: 0.671875
#  val: loss: 0.6895421147346497 | acc: 0.515625 
# ----------
# epoch: 600 | training time:  1:10:14.191447
#  train: loss: 0.6281964182853699 | acc: 0.765625
#  val: loss: 0.6648959517478943 | acc: 0.59375 
# ----------
# epoch: 700 | training time:  1:21:55.482652
#  train: loss: 0.4863242208957672 | acc: 0.9375
#  val: loss: 0.5274335741996765 | acc: 0.890625 
# ----------
# epoch: 800 | training time:  1:33:39.674377
#  train: loss: 0.2844874858856201 | acc: 0.984375
#  val: loss: 0.3153885304927826 | acc: 0.984375 
# ----------
# epoch: 900 | training time:  1:45:22.709798
#  train: loss: 0.12050967663526535 | acc: 0.984375
#  val: loss: 0.1616237610578537 | acc: 1.0 
# ----------
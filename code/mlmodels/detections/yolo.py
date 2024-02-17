import torch
import torch.nn as nn
from typing import List, Dict, Union
from dataclasses import dataclass
import random
from torchinfo import summary

@dataclass
class BoundingBox:
    """The (x, y) coordinates represent the center of the box relative to the bounds of the grid cell. 
    The width and height are predicted relative to the whole image. 
    Finally the confidence prediction represents the IOU between the predicted box and any ground truth box.
    """
    confidence: float
    x: int
    y: int
    w: int
    h: int
    c: int

    def __post_init__(self):
        self.x1 = self.x - 0.5 * self.w
        self.y1 = self.y - 0.5 * self.h
        self.x2 = self.x + 0.5 * self.w
        self.y2 = self.y + 0.5 * self.h

    def __lt__(self, other):
         return self.confidence < other.confidence


def intersaction_over_union(bb1: BoundingBox, bb2: BoundingBox):
    """_summary_

    Args:
        bb1 (BoundingBox): bounding box object 1
        bb2 (BoundingBox): bounding box object 2

    Returns:
        _type_: _description_
    """
    
    x1 = max(bb1.x1, bb2.x1)
    y1 = max(bb1.y1, bb2.y1)
    x2 = min(bb1.x2, bb2.x2)
    y2 = min(bb1.y2, bb2.y2)

    i_w = max(0, x2 - x1)
    i_h = max(0, y2 - y1)

    i_area = i_w * i_h
    bb1_area = bb1.w * bb1.h
    bb2_area = bb2.w * bb2.h
    t_area = bb1_area + bb2_area - i_area
    iou = i_area / t_area

    return iou


def non_maximum_suppression(B: List[Dict], iou_treshold: float = 0.5, confidence_treshold: float = 0.5):
    """
    Require: 
        Set of predicted bounding boxes B, 
        confidence scores S, 
        IoU threshold τ, 
        confidence threshold T 
    Ensure: Set of filtered bounding boxes F

    F ← ∅
    Filtertheboxes:B←{b∈B|S(b)≥T}
    Sort the boxes B by their confidence scores in descending order 
    while B̸∅` do
        Select the box b with the highest confidence score Add b to the set of final boxes F : F ← F ∪ {b} 
        Remove b from the set of boxes B: B ← B − {b} 
        for all remaining boxes r in B do
            Calculate the IoU between b and r: iou ← IoU(b,r) 
            if iou ≥ τ then
                Remove r from the set of boxes B: B ← B − {r} 
            end if
        end for 
    end while

    Args:
        bb1 (_type_): _description_
        bb2 (_type_): _description_
        target (_type_): _description_

    Returns:
        _type_: _description_
    """

    B.sort(key=lambda b: b.confidence, reverse=True)

    categories = set()
    for b in B:
        categories.add(b.c)

    BB = {}
    for c in categories:
        BB[c] = []

    for b in B:
        BB[b.c].append(b)

    B_FINAL = []
    for _, B in BB.items():
        while B:
            r_list = []
            b_best = B.pop(0)
            B_FINAL.append(b_best)
            for b in B:
                iou_score = intersaction_over_union(b_best, b)
                if iou_score > iou_treshold:
                    r_list.append(b)
            [B.remove(b) for b in r_list]

    return B_FINAL


class YOLOv1(nn.Module):
    def __init__(self, c: int, b: int = 2, s: int = 7) -> None:
        """
        

        Args:
            c (int): how many classes to predict
            b (int, optional): how many bounding boxes to predict in each cell. Defaults to 2.
            s (int, optional): how many cells in image. Defaults to 7.
        """
        super().__init__()
        
        self.size = s
        self.depth = b * 5 + c # 5 dims: confidance, x, y, w, h
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),          
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        
        self.conv3 = nn.ModuleList()
        for _ in range(4):
            self.conv3.append(nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.LeakyReLU())
            )
        self.conv3 = nn.Sequential(*self.conv3)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(),      
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        
        self.conv5 = nn.ModuleList()
        for _ in range(2):
            self.conv5.append(nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU())                  
            )
        self.conv5 = nn.Sequential(*self.conv5)
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50176, 4096),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, self.size * self.size * self.depth)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x  = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.fc(x)
        
        x = torch.reshape(x,  (x.size(dim=0), self.size, self.size, self.depth))
        
        return x


class DarknetConv(nn.Module):
    
    def __init__(self, depth: int, in_channels: int, out_channels: Union[None, int]= None) -> None:
        super().__init__()
        self.depth = depth
        self.layer_list = nn.ModuleList()

        first_layer_channels = (in_channels, in_channels * 2 if depth > 1 else out_channels)
        even_layer_channels = (in_channels * 2, in_channels)

        for d in range(1, depth + 1):
            if d % 2 != 0:
                self.layer_list.append(nn.Conv2d(in_channels=first_layer_channels[0],
                                                out_channels=first_layer_channels[1],
                                                kernel_size=3, padding=1))
                self.layer_list.append(nn.BatchNorm2d(num_features=first_layer_channels[1]))
            else:
                self.layer_list.append(nn.Conv2d(in_channels=even_layer_channels[0],
                                                out_channels=even_layer_channels[1],
                                                kernel_size=1))
                self.layer_list.append(nn.BatchNorm2d(num_features=even_layer_channels[1]))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)            

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
            x = self.leaky_relu(x)
        
        return x


class Darknet19(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv_list = [DarknetConv(1, 3, 32),
                          DarknetConv(1, 32, 64),
                          DarknetConv(3, 64),
                          DarknetConv(3, 128),
                          DarknetConv(5, 256),
                          DarknetConv(5, 512)]

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _concatenate(self, skip, x):
        sub_layer1 = skip[:, :, :13, :13]
        sub_layer2 = skip[:, :, 13:, :13]
        sub_layer3 = skip[:, :, :13, 13:]
        sub_layer4 = skip[:, :, 13:, 13:]
        out = torch.cat([sub_layer1, sub_layer2, sub_layer3, sub_layer4], dim=1)
        
        return torch.cat([out, x], dim=1)
        

    def forward(self, x):
        for i, conv_layer in enumerate(self.conv_list[:-1]):
            x = conv_layer(x)
            if i == 4:
                pass_through = x
            x = self.max_pool(x)
        x = self.conv_list[-1](x) # last layer without max pooling
        x = self._concatenate(pass_through, x)
        assert x.size() == torch.Size([x.size()[0], 3072, 13, 13])
        return x


class YOLOv2(nn.Module):
    def __init__(self, classes: int, anchor_boxes: int = 5) -> None:
        super().__init__()
        self.darknet19 = Darknet19()
        self.output = nn.Conv2d(in_channels=...,
                                out_channels=anchor_boxes*(5+classes),
                                kernel_size=1)

    def forward(self, x):
        return x


if __name__ == '__main__':

    # Test IOU
    bb1 = BoundingBox(0.8, 10, 10, 4, 5, 0)
    bb2 = BoundingBox(0.6, 11, 11, 4, 5, 0)
    iou = intersaction_over_union(bb1, bb2)
    # print(f'test iou: {iou}')

    # Test NMS
    B = []
    for _ in range(200):
        condfidance = random.random()
        x = random.randint(8, 10)
        y = random.randint(8, 10)
        w = random.randint(8, 10)
        h = random.randint(8, 10)
        c = random.choice([0, 1, 2, 3, 4])
        B.append(BoundingBox(condfidance, x, y, w, h, c))
    B_final = non_maximum_suppression(B, iou_treshold=0.5)
    # print(f'final BB: {B_final}')

    # YOLOv1
    X =  torch.rand(1, 3, 448, 448)
    yolo_v1 = YOLOv1(c=3)
    out = yolo_v1(X)
    B = []
    for img in range(out.shape[0]):
        c = torch.argmax(out[img, :, :, 10:], dim=2).flatten()
        B_array = out[img, :, :, :10].reshape(-1, 5)
        for i, bb in enumerate(B_array):
            B.append(
                BoundingBox(
                    confidence=bb[0],
                    x=bb[0],
                    y=bb[1],
                    w=bb[2],
                    h=bb[3],
                    c=c[i//2]
                )
            )

    B_final = non_maximum_suppression(B)
    # print(f'final BB: {B_final}')

    # YOLO v2
    X = torch.rand(1, 3, 224, 224)
    darknet_conv = DarknetConv(1, 3, 32)
    out = darknet_conv(X)
    X = torch.rand(1, 64, 56, 56)
    darknet_conv =  DarknetConv(3, 64)
    out = darknet_conv(X)
    X = torch.rand(1, 3, 416, 416)
    darknet19 = Darknet19()
    out = darknet19(X)
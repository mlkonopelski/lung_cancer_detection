import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Union, Tuple, Optional
from dataclasses import dataclass
import random
# from torchinfo import summary
import os
import cv2
import numpy as np
# from sklearn.cluster import KMeans
import math

@dataclass
class BoundingBox:
    """The (x, y) coordinates represent the center of the box relative to the bounds of the grid cell. 
    The width and height are predicted relative to the whole image. 
    Finally the confidence prediction represents the IOU between the predicted box and any ground truth box.
    """
    confidence: float
    c: int
    x: Union[int, float] = None
    y: Union[int, float] = None
    w: Union[int, float] = None
    h: Union[int, float] = None
    t_x: Optional[float] = None
    t_y: Optional[float] = None
    t_w: Optional[float] = None
    t_h: Optional[float] = None
    anchor: Optional[int] = None
    x_grid: Optional[int] = None
    y_grid: Optional[int] = None

    def __post_init__(self):
        self.x1 = self.x - 0.5 * self.w
        self.y1 = self.y - 0.5 * self.h
        self.x2 = self.x + 0.5 * self.w
        self.y2 = self.y + 0.5 * self.h
    
    def calculate_offsets(self, grid_size: int, anchor_boxes: Dict):
        self.t_x = self.x - self.x_grid/grid_size
        self.t_y = self.y - self.y_grid/grid_size
        self.t_w = math.log(self.w / anchor_boxes[self.anchor]['w'])
        self.t_h = math.log(self.h / anchor_boxes[self.anchor]['h'])

    def calculate_from_offests(self, grid_size: int, anchor_boxes: Dict):
        self.x = self.t_x + self.x_grid/grid_size
        self.y = self.t_y + self.y_grid/grid_size
        self.w = anchor_boxes[self.anchor]['w'] * math.exp(self.t_w)
        self.h = anchor_boxes[self.anchor]['h'] * math.exp(self.t_h)

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

    # NOT SURE TO BE HONEST IF THIS IS CORRECT        
    # i_w = np.minimum(bb1.w, bb2.w) 
    # i_h = np.minimum(bb1.h, bb2.h)

    i_area = i_w * i_h
    bb1_area = bb1.w * bb1.h
    bb2_area = bb2.w * bb2.h
    t_area = bb1_area + bb2_area - i_area
    iou = i_area / t_area

    return iou

def plot_one_box(x, image, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def intersaction_over_union_from_arrays(clusters: np.ndarray, box: np.ndarray) -> float:

    x = np.minimum(clusters[0], box[:, 0]) 
    y = np.minimum(clusters[1], box[:, 1])

    i_area = x * y
    bb_area = box[:, 0] * box[:, 1]
    cluster_area = clusters[0] * clusters[1]

    t_area = bb_area + cluster_area - i_area
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


class YOLOTensorCreator(nn.Module):
    def __init__(self, 
                 path: str, 
                 label_format: str = '.txt',
                 img_format: str = '.jpg', 
                 grid_size: int = 13,
                 anchor_boxes: int = 5,
                 out_size: Tuple[int, int] = (448, 488),
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.path = path
        self.label_format = label_format
        self.img_format = img_format
        self.grid_size = grid_size
        self.anchor_boxes_count = anchor_boxes
        self.out_size = out_size
    
        # run helper methods to prepare class before __call__
        self._create_list_of_labels()
        self._find_anchor_boxes()
    
    def _create_list_of_labels(self):
        labels_files = [file for file in os.listdir(self.path) if file.endswith(f'{self.label_format}')]
        labels_dict = {}
        classes = []
        for label_file in labels_files:
            bbox_list = []
            id = label_file.replace(f'{self.label_format}', '')
            with open(os.path.join(self.path, label_file), 'r') as f:
                for bbox_details in f.readlines():
                    bbox_details = bbox_details.split(' ')
                    classes.append(bbox_details[0])
                    bbox_list += [
                        BoundingBox(
                            confidence=1,
                            x=float(bbox_details[1]),
                            y=float(bbox_details[2]),
                            w=float(bbox_details[3]),
                            h=float(bbox_details[4]),
                            c=int(bbox_details[0]),
                            x_grid = int(float(bbox_details[1]) * self.grid_size),
                            y_grid = int(float(bbox_details[2]) * self.grid_size) 
                        )
                    ]
            labels_dict[id] = bbox_list

        self.labels = labels_dict
        self.classes = list(set(classes))
        self.classes_count = len(self.classes)
        
    def _find_anchor_boxes(self):
        dist = np.mean
        
        boxes = []
        i = 0
        for _, bbox in self.labels.items():
            for bb in bbox:
                boxes += [[bb.w, bb.h]]
        boxes = np.array(boxes)
        samples = boxes.shape[0]
        
        distances = np.empty((samples, 5))
        last_clusters = np.zeros((samples,))
        
        clusters = boxes[np.random.choice(samples, 5, replace=False)]
        
        while True:
            for k in range(5):
                # calculate distance between each coordinate and each cluster (separatly)
                distances[:, k] = 1 - intersaction_over_union_from_arrays(clusters[k], boxes)

            # calculate which cluster is nearest to each coordinate
            nearest_clusters = np.argmin(distances, axis=1)
            
            # for each coordinate we found nearest cluster
            if (last_clusters == nearest_clusters).all():
                break
            
            # calcuate mean if multiple boxes are one of the clusters best iou
            for k in range(5):
                clusters[k] = dist(boxes[nearest_clusters == k], axis=0)

            last_clusters = nearest_clusters
        
        # Add to self.labels information about each bounding box anchor ix
        nc_ix = 0
        for id, bbox in self.labels.items():
            for i, bb in enumerate(bbox):
                self.labels[id][i].anchor = nearest_clusters[nc_ix]
                nc_ix += 1
 
        # Create dictionary for anchor boxes coordinates
        self.anchor_boxes = {}
        for i, row in enumerate(clusters):
            self.anchor_boxes[i] = {'w': row[0], 'h': row[1]}
        self.anchor_boxes_indexes = np.arange(0, self.anchor_boxes_count * (5 + self.classes_count), 5 + self.classes_count)
        
    def read_label_as_tensor(self, id: str):
        label_tensor = torch.zeros((self.grid_size, self.grid_size, self.anchor_boxes_count * (5 + self.classes_count)))
        for bbox in self.labels[id]:
            class_onehot = torch.zeros(self.classes_count) 
            class_onehot[bbox.c] = 1
            anchor_ix = self.anchor_boxes_indexes[bbox.anchor]
            bbox.calculate_offsets(self.grid_size, self.anchor_boxes)
            bbox_tensor = torch.Tensor([
                bbox.confidence,
                bbox.t_x,
                bbox.t_y,
                bbox.t_w,
                bbox.t_h                
            ])
            bbox_tensor = torch.cat([bbox_tensor, class_onehot])
            label_tensor[bbox.x_grid, bbox.y_grid, anchor_ix: anchor_ix + 5 + self.classes_count] = bbox_tensor
        return label_tensor
    
    def read_image_as_tensor(self, id: str):
        img = cv2.imread(os.path.join(self.path, id + self.img_format)) 
        img_resized = cv2.resize(img, (416, 416), interpolation = cv2.INTER_LINEAR)
        return torch.from_numpy(img_resized).permute(2, 0, 1).to(torch.float)

    def _save_bboxes_on_img(self, img):
        img_print = img.permute(1, 2, 0).numpy()
        for bb in self.labels[id]: 
            cv2.rectangle(img_print, (int(bb.x1*448), int(bb.y1*448)), (int(bb.x2*448), int(bb.y2*448)), color=(255, 0, 0) )
        cv2.imwrite(f'.data/traffic-signs/bbox-shows/{id}-bb.jpg', img_print)

    def forward(self, id: str) -> torch.Tensor:
        
        img = self.read_image_as_tensor(id)
        labels = self.read_label_as_tensor(id)
        
        return img, labels
    

class YOLOv2Loss(nn.Module):
    def __init__(self, converter: YOLOTensorCreator) -> None:
        super().__init__()
        self.converter = converter
        self.lambda_no_object = 1.0
        self.lambda_object    = 5.0
        self.lambda_coord     = 1.0
        self.lambda_class     = 1.0
        
    def _loss_class(self, truth, pred):
        return (self.lambda_class / self.N_l) * ((truth * torch.log(pred)).sum(3)).sum(2).sum(1)
    
    def _loss_xywh(self, truth, pred, conf_truth):
       
        xy_truth, xy_pred = truth[:,:,:,:2], pred[:,:,:,:2]
        wh_truth, wh_pred = truth[:,:,:,2:], pred[:,:,:,2:]

        xy_loss = torch.pow(xy_truth - xy_pred, 2).sum(3)
        wh_loss = torch.pow(torch.sqrt(wh_truth) - torch.sqrt(wh_pred), 2).sum(3) # wh_truth is sometimes 
        # wh_loss = torch.pow(wh_truth -wh_pred, 2).sum(3)

        xywh_loss = xy_loss + wh_loss
        xywh_loss = (conf_truth * xywh_loss).sum(2).sum(1)        
        return (self.lambda_coord / self.N_l) * xywh_loss
    
    def _offsets_to_original(self, tensor: torch.Tensor, conf_truth) -> torch.Tensor:
        
        truth_mask = conf_truth.unsqueeze(-1).expand_as(tensor)
        
        x_grid_offset = (torch.arange(0, 13) / 13).unsqueeze(-1).expand(-1, 13).flatten().unsqueeze(-1).expand(-1, 5)
        x = tensor[:,:,:, 0] + x_grid_offset
        
        y_grid_offset = torch.cat([torch.arange(0, 13) / 13 for _ in range(13)]).unsqueeze(-1).expand(-1, 5)
        y = tensor[:,:,:, 1] + y_grid_offset
        
        w_anchors = torch.Tensor([value['w'] for _, value in self.converter.anchor_boxes.items()]).expand(13*13, -1)
        w = w_anchors * torch.exp(tensor[:,:,:, 2])
        
        h_anchors = torch.Tensor([value['h'] for _, value in self.converter.anchor_boxes.items()]).expand(13*13, -1)
        h = h_anchors * torch.exp(tensor[:,:,:, 3])

        return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), w.unsqueeze(-1), h.unsqueeze(-1)], 3) * truth_mask

    def _intersaction_over_union_from_batch(self, truth: torch.Tensor, pred: torch.Tensor):
        
        truth_wh_half = truth[:, :, :, 2:] / 2
        truth_xy_min = truth[:, :, :, :2] - truth_wh_half
        truth_xy_max = truth[:, :, :, 2:] + truth_wh_half
        truth_area = truth[:, :, :, 2] * truth[:, :, :, 3]
        
        pred_wh_half = pred[:, :, :, 2:] / 2
        pred_xy_min = pred[:, :, :, :2] - pred_wh_half
        pred_xy_max = pred[:, :, :, 2:] + pred_wh_half
        pred_area = pred[:, :, :, 1] * pred[:, :, :, 2]

        i_min = torch.max(truth_xy_min, pred_xy_min)
        i_max = torch.min(truth_xy_max, pred_xy_max)
        i_wh = torch.where(i_max - i_min < 0, 0, i_max - i_min)
        i_area = i_wh[:,:,:,0] * i_wh[:,:,:,1]
        
        total_area = truth_area + pred_area - i_area
        total_area = torch.where(total_area == 0, 1e-6, total_area)
        iou = i_area / total_area
        return iou
    
    def _loss_conf(self, truth, pred, xywh_truth, xywh_pred):
        iou = self._intersaction_over_union_from_batch(xywh_truth, xywh_pred)
        loss_obj = self.lambda_object * truth * (iou - pred)
        loss_no_obj = self.lambda_no_object * torch.where(truth == 1, 0, 1) * (0 - pred)
        return loss_obj.sum(2).sum(1) + loss_no_obj.sum(2).sum(1)
        
    def forward(self, xywh_truth, xywh_pred, conf_truth, conf_pred, class_truth, class_pred):
        
        xywh_truth = self._offsets_to_original(xywh_truth, conf_truth)
        xywh_pred = self._offsets_to_original(xywh_pred, conf_truth)
        
        self.N_l = conf_truth[:,:,:].sum()
        
        loss_xywh = self._loss_xywh(xywh_truth, xywh_pred, conf_truth)
        loss_class = self._loss_class(class_truth, class_pred)
        loss_conf = self._loss_conf(conf_truth, conf_pred, xywh_truth, xywh_pred)
        
        return loss_conf + loss_xywh + loss_class

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

    def forward(self, x):
        for i, conv_layer in enumerate(self.conv_list[:-1]):
            x = conv_layer(x)
            if i == 4:
                pass_through = x
            x = self.max_pool(x)
        x = self.conv_list[-1](x) # last layer without max pooling
        return x, pass_through


class YOLOv2(nn.Module):
    def __init__(self, classes: int, anchor_boxes: int = 5) -> None:
        super().__init__()
        self.classes = classes
        self.anchor_boxes = anchor_boxes
        self.darknet19 = Darknet19()
        self.conv = nn.ModuleList()
        for _ in range(3):
            self.conv.append(nn.Sequential(
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.BatchNorm2d(1024)
            ))
        self.global_average_pool = nn.AvgPool2d((1, 1))
        
        self.output = nn.Conv2d(in_channels=3072,
                                out_channels=anchor_boxes*(5+classes),
                                kernel_size=1)

    def _concatenate(self, skip, x):
        sub_layer1 = skip[:, :, :13, :13]
        sub_layer2 = skip[:, :, 13:, :13]
        sub_layer3 = skip[:, :, :13, 13:]
        sub_layer4 = skip[:, :, 13:, 13:]
        out = torch.cat([sub_layer1, sub_layer2, sub_layer3, sub_layer4], dim=1)
        
        return torch.cat([out, x], dim=1)
    
    def forward(self, x):
        x, pass_through = self.darknet19(x)
        for c_l in self.conv:
            x = c_l(x)
        x = self._concatenate(pass_through, x)
        x = self.output(x)

        # resize the array for loss calculations: 5D: batch_size x grid_size * grid_size, anchor_boxes x 5 + classes
        batch_size = x.shape[0]
        x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.anchor_boxes, 5 + self.classes)
        xy_pred = F.sigmoid(x[:,:,:,1:3])
        wh_pred = torch.exp(x[:,:,:,3:5])
        bbox_pred = torch.cat([xy_pred, wh_pred], dim=3)
        confidence_pred = F.sigmoid(x[:,:,:,0])
        class_pred = F.softmax(x[:,:,:,5:], 3)
        
        return bbox_pred, confidence_pred, class_pred





def batch_processing(bbox):
    '''
    useful implementation: https://github.com/longcw/yolo2-pytorch/blob/master/darknet.py
    '''
    return ...


if __name__ == '__main__':
    # import glob
    # yolo_creator = YOLOTensorCreator('.data/traffic-signs/train')
    # for id_path in glob.glob('.data/traffic-signs/train/*.txt'):
    #     id = os.path.basename(id_path).replace('.txt','')
    #     X, y = yolo_creator(id=id)

    yolo_creator = YOLOTensorCreator('.data/traffic-signs/train')
    X, y = yolo_creator(id='00001')
    print(yolo_creator.anchor_boxes)
    
    yolo_v2 = YOLOv2(classes=4)
    
    y_loss = y.reshape(1, 13*13, 5, -1)
    bbox_truth, confidence_truth, class_truth = y_loss[:,:,:,1:5], y_loss[:,:,:,0], y_loss[:,:,:,5:] 
    bbox_pred, confidence_pred, class_pred = yolo_v2(X.unsqueeze(0))
    

    loss_fn = YOLOv2Loss(converter=yolo_creator)
    loss = loss_fn(bbox_truth,  bbox_pred, confidence_truth, confidence_pred, class_truth, class_pred)
    loss
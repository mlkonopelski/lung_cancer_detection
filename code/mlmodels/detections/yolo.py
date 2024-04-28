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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings

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


class YOLOv2TensorCreator(nn.Module):
    def __init__(self, 
                 path: str, 
                 label_format: str = '.txt',
                 img_format: str = '.jpg', 
                 grid_size: int = 13,
                 anchor_boxes: int = 5,
                 out_size: Tuple[int, int] = (416, 416),
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.path = path
        self.label_format = label_format
        self.img_format = img_format
        self.grid_size = grid_size
        self.anchor_boxes_count = anchor_boxes
        if out_size[0] % 32 != 0 or  out_size[1] % 32 != 0:
            raise Exception('The input size needs to be x32')
        self.out_size = out_size
    
        # run helper method which will be used in Dataset.__getitem__
        self.files_id = self._file_id_list()
    
        # run helper methods to prepare class before __call__
        self._create_list_of_labels()
        self._find_anchor_boxes()
    
    def _file_id_list(self):
        labels_files = [file.replace(f'{self.label_format}', '') for file in os.listdir(self.path) if file.endswith(f'{self.label_format}')]
        img_files = [file.replace(f'{self.img_format}', '') for file in os.listdir(self.path) if file.endswith(f'{self.img_format}')]
        return list(set(labels_files).intersection(img_files))
    
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
    def __init__(self, converter: YOLOv2TensorCreator) -> None:
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
        # wh_loss = torch.pow(torch.sqrt(wh_truth) - torch.sqrt(wh_pred), 2).sum(3) # wh_truth is sometimes 
        wh_loss = torch.pow(wh_truth - wh_pred, 2).sum(3)

        xywh_loss = xy_loss + wh_loss
        xywh_loss = (conf_truth * xywh_loss).sum(2).sum(1)        
        return (self.lambda_coord / self.N_l) * xywh_loss
    
    def _offsets_to_original(self, tensor: torch.Tensor, conf_truth, device) -> torch.Tensor:
        
        truth_mask = conf_truth.unsqueeze(-1).expand_as(tensor)
        
        x_grid_offset = (torch.arange(0, 13) / 13).unsqueeze(-1).expand(-1, 13).flatten().unsqueeze(-1).expand(-1, 5).to(device)
        x = tensor[:,:,:, 0] + x_grid_offset
        
        y_grid_offset = torch.cat([torch.arange(0, 13) / 13 for _ in range(13)]).unsqueeze(-1).expand(-1, 5).to(device)
        y = tensor[:,:,:, 1] + y_grid_offset
        
        w_anchors = torch.Tensor([value['w'] for _, value in self.converter.anchor_boxes.items()]).expand(13*13, -1).to(device)
        w = w_anchors * torch.exp(tensor[:,:,:, 2])
        
        h_anchors = torch.Tensor([value['h'] for _, value in self.converter.anchor_boxes.items()]).expand(13*13, -1).to(device)
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
        
        DEVICE = xywh_truth.device
        
        xywh_truth = self._offsets_to_original(xywh_truth, conf_truth, DEVICE)
        xywh_pred = self._offsets_to_original(xywh_pred, conf_truth, DEVICE)
        
        self.N_l = conf_truth.sum(2).sum(1)
        
        loss_xywh = self._loss_xywh(xywh_truth, xywh_pred, conf_truth)
        loss_class = self._loss_class(class_truth, class_pred)
        loss_conf = self._loss_conf(conf_truth, conf_pred, xywh_truth, xywh_pred)
        
        loss = loss_class # +loss_conf + loss_xywh 
        
        return torch.mean(loss)

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
    def __init__(self) -> None:
        super().__init__()
        self.conv_list = nn.ModuleList([DarknetConv(1, 3, 32),
                          DarknetConv(1, 32, 64),
                          DarknetConv(3, 64),
                          DarknetConv(3, 128),
                          DarknetConv(5, 256),
                          DarknetConv(5, 512)])

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


class YOLOv2Dataset(Dataset):
    def __init__(self, img_path: str, transform: nn.Module = None) -> None:
        super().__init__()
        self.yolo_creator = YOLOv2TensorCreator(img_path) # '.data/traffic-signs/train'
        self.files = self.yolo_creator.files_id
        self.transform = transform


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
       X, y = self.yolo_creator(id=self.files[idx])
       if self.transform:
           X = self.transform(X)
       return X, y

def training_yolov2(device='cpu', epochs:int = 2, batch_size: int = 32):
    
    transformations = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    train_dataset = YOLOv2Dataset(img_path='.data/traffic-signs/train', transform=transformations)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    
    yolov2 = YOLOv2(classes=train_dataset.yolo_creator.classes_count).to(device=device)
    loss_fn = YOLOv2Loss(converter=train_dataset.yolo_creator)
    
    optimizer = torch.optim.SGD(yolov2.parameters(), lr=0.001, momentum=0.9)
    torch.autograd.set_detect_anomaly(True)
    
    for epoch_ix in range(1, epochs+1):
        for batch_idx, (X, y) in enumerate(train_dataloader):
            
            optimizer.zero_grad()
            
            X, y = X.to(device=device), y.reshape(X.shape[0], 13*13, 5, -1).to(device=device)            
            bbox_truth, confidence_truth, class_truth = y[:,:,:,1:5], y[:,:,:,0], y[:,:,:,5:]
            bbox_pred, confidence_pred, class_pred = yolov2(X)
            loss = loss_fn(bbox_truth,  bbox_pred, confidence_truth, confidence_pred, class_truth, class_pred)
            
            loss.backward()
            optimizer.step()
            
            print(f'\tbatch"{batch_idx} loss:{loss}')
        print(f'epoch:{epoch_ix} loss:{loss}')


# -------------------------------------------------
#   simplified YOLO V5 based on Ultralytics
# -------------------------------------------------

def initialize_weights(model):
    """Initializes weights of Conv2d, BatchNorm2d, and activations (Hardswish, LeakyReLU, ReLU, ReLU6, SiLU) in the
    model.
    """
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

def make_divisible(x, divisor):
    """Adjusts `x` to be divisible by `divisor`, returning the nearest greater or equal value."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def check_anchor_order(m):
    """Checks and corrects anchor order against stride in YOLOv5 Detect() module if necessary."""
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        m.anchors[:] = m.anchors.flip(0)
        
def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    """Scales an image tensor `img` of shape (bs,3,y,x) by `ratio`, optionally maintaining the original shape, padded to
    multiples of `gs`.
    """
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean



class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initializes a depth-wise convolution layer with optional activation; args: input channels (c1), output
        channels (c2), kernel size (k), stride (s), dilation (d), and activation flag (act).
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initializes a depth-wise transpose convolutional layer for YOLOv5; args: input channels (c1), output channels
        (c2), kernel size (k), stride (s), input padding (p1), output padding (p2).
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

    
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):
        """
        Initializes YOLOv5 SPPF layer with given channels and kernel size for YOLOv5 model, combining convolution and
        max pooling.

        Equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Processes input through a series of convolutions and max pooling operations for feature extraction."""
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

    
class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        """Initializes a Concat module to concatenate tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Concatenates a list of tensors along a specified dimension; `x` is a list of tensors, `dimension` is an
        int.
        """
        return torch.cat(x, self.d)


class Proto(nn.Module):
    # YOLOv5 mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32):
        """Initializes YOLOv5 Proto module for segmentation with input, proto, and mask channels configuration."""
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass using convolutional layers and upsampling on input tensor `x`."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """Initializes YOLOv5 detection layer with specified classes, anchors, channels, and inplace operations."""
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """Initializes YOLOv5 Segment head with options for mask count, protos, and channel adjustments."""
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        """Processes input through the network, returning detections and prototypes; adjusts output based on
        training/export mode.
        """
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


@dataclass
class YOLOv8CONFIG:
    na = 6
    gd = 0.67 # model depth multiple
    gw = 0.75 # layer channel multiple
    ch = [3]
    ch_mul = 8
    
    # BASED ON yolov5m
    backbone = [
        [-1, 1, Conv, [64, 6, 2, 2]], # 0-P1/2
        [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
        [-1, 3, C3, [128]],
        [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
        [-1, 6, C3, [256]],
        [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
        [-1, 9, C3, [512]],
        [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
        [-1, 3, C3, [1024]],
        [-1, 1, SPPF, [1024, 5]], # 9
    ]

    head= [
        [-1, 1, Conv, [512, 1, 1]],
        [-1, 1, nn.Upsample, [None, 2, "nearest"]],
        [[-1, 6], 1, Concat, [1]], # cat backbone P4
        [-1, 3, C3, [512, False]], # 13

        [-1, 1, Conv, [256, 1, 1]],
        [-1, 1, nn.Upsample, [None, 2, "nearest"]],
        [[-1, 4], 1, Concat, [1]], # cat backbone P3
        [-1, 3, C3, [256, False]], # 17 (P3/8-small)

        [-1, 1, Conv, [256, 3, 2]],
        [[-1, 14], 1, Concat, [1]], # cat head P4
        [-1, 3, C3, [512, False]], # 20 (P4/16-medium)

        [-1, 1, Conv, [512, 3, 2]],
        [[-1, 10], 1, Concat, [1]], # cat head P5
        [-1, 3, C3, [1024, False]], # 23 (P5/32-large)
    ]

class YOLOv5(nn.Module):
    def __init__(self, nc, task: Union[Detect, Segment]) -> None:
        super().__init__()
        self.C = YOLOv8CONFIG
        self.C.nc = nc
        self.C.no = self.C.na * (self.C.nc + 5)
        self.task = task
        self.m = self._init_model()
        
    def _init_model(self):
        
        layers, c2 = [], self.C.ch[-1]  # layers, savelist, ch out
        if self.task not in [Detect, Segment]:
            raise Exception('ONLY: Detect, Segment')
        if self.task is Segment:
            final_layer = [[[17, 20, 23], 1, Segment, [self.C.nc, self.C.na, 32, 256]]]
        elif self.task is Detect:
            final_layer = [[[17, 20, 23], 1, Detect, [self.C.nc, self.C.na]]]
        
        for i, (f, n, m, args) in enumerate(self.C.backbone + self.C.head + final_layer): #(from, number, module, args)
            n = max(round(n * self.C.gd), 1) if n > 1 else n  # depth gain
            
            if m in [Conv, SPPF, C3, nn.ConvTranspose2d]:
                c1, c2 = self.C.ch[f], args[0]
                if c2 != self.C.no:  # if not output
                    c2 = make_divisible(c2 * self.C.gw, self.C.ch_mul)
                args = [c1, c2, *args[1:]]
                if m in [C3]: #{BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                    args.insert(2, n)  # number of repeats
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [self.C.ch[f]]
            elif m is Concat:
                c2 = sum(self.C.ch[x] for x in f)
            elif m in [Detect, Segment]:
                args.append([self.C.ch[x] for x in f])
                if isinstance(args[1], int):  # number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f)
                if m is Segment:
                    args[3] = make_divisible(args[3] * self.C.gw, self.C.ch_mul)
            else:
                c2 = self.C.ch[f]
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace("__main__.", "")  # module type
            np = sum(x.numel() for x in m_.parameters())  # number params
            m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
            layers.append(m_)
            if i == 0:  # I don't get this
                self.C.ch = []
            self.C.ch.append(c2)
        return nn.Sequential(*layers)#, sorted(save)
    
    def forward(self, x):
        return self.m(x)


class YOLOv5Advanced(YOLOv5):
    
    def __init__(self, nc, task: Union[Detect, Segment]) -> None:
        super().__init__(nc, task)
    
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        # Build strides, anchors
        m = self.m[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, self.C.ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        
    def _initialize_biases(self, cf=None):
        """
        Initializes biases for YOLOv5's Detect() module, optionally using class frequencies (cf).

        For details see https://arxiv.org/abs/1708.02002 section 3.3.
        """
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Performs single-scale or augmented inference and may include profiling or visualization."""
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        """De-scales predictions from augmented inference, adjusting for flips and image size."""
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p
    
    def _clip_augmented(self, y):
        """Clips augmented inference tails for YOLOv5 models, affecting first and last tensors based on grid points and
        layer counts.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _forward_augment(self, x):
        """Performs augmented inference across different scales and flips, returning combined detections."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train
    
    def forward(self, x, augment=False, profile=False, visualize=False):
        """Performs single-scale or augmented inference and may include profiling or visualization."""
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train



if __name__ == '__main__':

    # YOLOv2
    # training_yolov2(device='mps', epochs=2)
    
    transformations = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    train_dataset = YOLOv2Dataset(img_path='.data/traffic-signs/train', transform=transformations)

    # YOLOv5
    m = YOLOv5(nc=train_dataset.yolo_creator.classes_count, task=Detect)
    print(m)
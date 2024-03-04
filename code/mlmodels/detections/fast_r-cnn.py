import torch.nn as nn
import torch
from torchvision.models import vgg, VGG16_Weights
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from _region_proposal import SelectiveSearch, ImageWarp, RProposalDataset
from typing import Tuple


class ROIPooling(nn.Module):
    """The RoI pooling layer uses max pooling to convert the
features inside any valid region of interest into a small feature map with a fixed spatial extent of H × W (e.g., 7 × 7),
where H and W are layer hyper-parameters that are independent of any particular RoI. In this paper, an RoI is a
rectangular window into a conv feature map. Each RoI is
defined by a four-tuple (r, c, h, w) that specifies its top-left
corner (r, c) and its height and width (h, w).
RoI max pooling works by dividing the h × w RoI window into an H × W grid of sub-windows of approximate
size h/H × w/W and then max-pooling the values in each
sub-window into the corresponding output grid cell. Pooling is applied independently to each feature map channel,
as in standard max pooling.

    """
    def __init__(self, size:Tuple[int] = (7, 7)) -> None:
        super().__init__()
        self.W = size[0]
        self.H = size[1]

    def forward(self, x):
        c, w, h = x.shape[0], x.shape[1], x.shape[2]
        w_grid, h_grid = int(round(w /self.W)), int(round(h /self.H))

        x_pool = torch.empty((c, self.W, self.H), dtype=torch.float32)

        for x_grid in range(self.W):
            for y_grid in range(self.H):
                # FIXME: slicing doesn't work this way
                x_pool[:, x_grid, y_grid] = F.max_pool2d(x[:, x_grid:x_grid+w_grid, y_grid: y_grid+h_grid])

        return x_pool


class FastRCNN(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()

        self.classes = classes
        self.vvg16 = vgg.vgg16(VGG16_Weights.IMAGENET1K_V1)
        self.preprocess = VGG16_Weights.DEFAULT.transforms()
        self.roi_pool = ROIPooling()
        self.fc = self.vvg16.classifier[:-1]
        self.fc_out_features = self.fc[3].out_features
        self.class_layer = nn.Linear(in_features=self.fc_out_features, out_features=classes)
        self.bbox_layer = nn.Linear(in_features=self.fc_out_features, out_features=classes*4)
        self.conv1 = self.vvg16.features[:2]

    def _project_roi_on_feature_map(self, x, r):
        #TODO: projecting method in case feature map.size <> original.size
        r_x1, r_y1, r_w, r_h = r[0], r[1], r[2], r[3]
        return x[:, r_x1: r_x1+r_w, r_y1:r_y1+r_h]
        
    def forward(self, img, regions):
        x = self.preprocess(x)
        x = self.conv1(img)
        
        cls_logits = torch.empty(len(regions), self.classes)
        loc_bbox = torch.empty(len(regions), self.classes*4)
        
        for i, r in enumerate(regions):
            x_roi = self._project_roi_on_feature_map(x, r)
            x_roi = self.roi_pool(x_roi)
            x_roi = self.fc(x_roi)
            cls_logits[i] = self.class_layer(x_roi)
            loc_bbox[i] = self.bbox_layer(x_roi)

        return cls_logits, loc_bbox


def train():
    
    ss = SelectiveSearch().to('mps')
    iw = ImageWarp()
    ds = RProposalDataset('.data/traffic-signs/train', ss, iw, save_regions_on_picture=False)
    
    fast_rcnn = FastRCNN(4)
    
    #criterion
    cls_loss = nn.CrossEntropyLoss()
    loc_loss = nn.MSELoss()
    
    # optim
    optimizer = torch.optim.SGD(fast_rcnn.parameters(), lr=0.001, momentum=0.9)
    
    for ep in range(100):
        for img, regions, labels in ds:

            optimizer.zero_grad()

            cls_label, loc_bbox_true = torch.argmax(labels[:,0]), labels[:,1]

            cls_logits, loc_bbox = fast_rcnn(img, regions)
            cls_loss = cls_loss(cls_logits, cls_label)
            loc_loss = 0 if cls_label == 0 else loc_loss(loc_bbox, loc_bbox_true)
            loss = cls_logits + loc_bbox
            
            loss.backward()
            optimizer.step()
    
    
if __name__ == '__main__':

    train()
import torch.nn as nn
import torch
import cv2
from torch.utils.data import Dataset, DataLoader, BatchSampler

from torchvision.models import EfficientNet_B7_Weights, efficientnet
from sklearn import svm
from torch import optim
from _region_proposal import SelectiveSearch, RProposalDataset, ImageWarp, get_iou
import random
from typing import Iterator

print(f'OPEN CV version: {cv2.__version__}')


class NegativeSampler(BatchSampler):
    """Majority of samples is negative (99.5%) therefore for purpose of training
    for every positive sample a negative is drow in random. If positive samples < batch size
    we do Hard Negative Mining and duplicate positive samples to batch size.
    
    Because 

    Args:
        BatchSampler (_type_): _description_
    """
    def __init__(self, positive_idx, negative_idx, batch_size: int = 32) -> None:
        self.positive_idx = positive_idx
        self.len_positive = len(self.positive_idx)
        self.negative_idx = negative_idx
        self.batch_size = batch_size
        self._upsample_positives()

    def __len__(self):
        return max(self.len_positive * 2, self.batch_size)

    def _upsample_positives(self):
        if self.len_positive * 2 < self.batch_size:
            self.positive_idx = self.positive_idx + [random.choice(self.positive_idx) for _ in range(self.len_positive, self.batch_size / 2)]

    def __iter__(self) -> Iterator[int]:
        positive_idx = random.shuffle(self.positive_idx)
        batch = []
        for pos_idx in positive_idx:
            batch.append(pos_idx)
            batch.append(random.choice(self.negative_idx))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        yield batch
    
class RCNNDataset(Dataset):
    def __init__(self, proposals: torch.Tensor, labels: torch.Tensor) -> None:
        super().__init__()
        assert proposals.shape[0] == labels.shape[0]
        self.proposals = proposals
        self.labels = labels
        self.positive_idx, self.negative_idx = self._find_indexes()

    def _find_indexes(self):
        positive = []
        negative = []

        for i, label in enumerate(self.labels):
            if label.sum() > 1:
                positive.append(i)
            else:
                negative.append(i)
        return positive, negative

    def __len__(self):
        return self.proposals.shape[0]

    def __getitem__(self, idx):
        # TODO: needs to build logic for balanced sampling positive/negatives            
        return self.proposals[idx], self.labels[idx]


class CNN(nn.Module):
    """Original Authors used architecture desrcibed here: https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
        However the important part is that model has been pre-trained on ImageNet so I will just use ENb7 with pretrained weights  

    Args:
        nn (_type_): _description_
    """
    cnn = efficientnet.efficientnet_b7(EfficientNet_B7_Weights.IMAGENET1K_V1)
    def __init__(self) -> None:
        super().__init__()
        self.preprocess = EfficientNet_B7_Weights.DEFAULT.transforms()
        self.freeze_layers()
        self.add_layers()
        
    def freeze_layers(self):
        for layer in self.cnn.parameters():
            layer.requires_grad = False
        
    def add_layers(self):
        self.cnn.classifier = nn.Sequential()   # trick to remove the last layer
                
    def forward(self, x):
        x = self.preprocess(x)
        return self.cnn(x)


def training():
    
    BATCH_SIZE = 100
    
    ss = SelectiveSearch().to('mps')
    iw = ImageWarp()
    ds = RProposalDataset('.data/traffic-signs/train', ss, iw, save_regions_on_picture=False)
    cnn = CNN()
    svc = svm.LinearSVC(dual='auto')
    
    for _, images, labels in ds:
        rcnn_ds = RCNNDataset(images, labels)
        ns = NegativeSampler(rcnn_ds, rcnn_ds, BATCH_SIZE)
        dl = DataLoader(rcnn_ds, batch_size=BATCH_SIZE, shuffle=True, batch_sampler=ns)
        for batch_image, batch_class in dl:
            X = cnn(batch_image)
            X = X.numpy()
            y = torch.argmax(batch_class, dim=1).numpy()
            
            #TODO: Here the weights will be dropped each time .fit is run
            # needs to be done completly different. 
            svc.fit(X, y)


if __name__ == '__main__':

    
    training()

    # ------------------------------
    # # Test custom sampler
    # ns = NegativeSampler([1, 2, 3, 4, 5], 
    #                       [9, 8, 7, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 
    #                       4)
    # for i, bal_batch in enumerate(ns):
    #     print(f"batch_{i}: ", bal_batch)

    # ------------------------------
    # # IF you want to save positives as .jpg for easier retrieval
    # ss = SelectiveSearch().to('mps')
    # iw = ImageWarp()
    # ds = RProposalDataset('.data/traffic-signs/train', ss, iw, save_regions_on_picture=False)
    # for file_id, images, labels in ds:
    #     for i, (image, class_tensor) in enumerate(zip(images, labels)):
    #         if class_tensor.sum() > 0:
    #             cv2.imwrite(f'.data/traffic-signs/train_positives/{file_id}_{i}.jpg', np.transpose(image.numpy(), (2, 1, 0)))
    

    # ------------------------------
    # PRINT IMAGES:
    # im_out = im.copy()
    # for i, rect in enumerate(rects):
    #     if i < 1000:
    #         x1, y1, w, h = rect
    #         cv2.rectangle(im_out, (x1, y1), (x1+w, y1+h), (0, 255, 0), 1, cv2.LINE_AA)
    #     else:
    #         break
    # cv2.imshow('im+bboxes', im_out)
    # k = cv2.waitKey(0) & 0xFF
    # cv2.destroyAllWindows()

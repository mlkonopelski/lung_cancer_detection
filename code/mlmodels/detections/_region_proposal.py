import cv2
import torch.nn as nn
import os
import torch
import numpy as np
from torch.utils.data import Dataset


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


class SelectiveSearch(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self) -> None:
        super().__init__()
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        
    def _select_candidate_regions(self, im, rects, n:int = 2000):
        
        W, H = im.shape[1], im.shape[0]
        min_rect_area, max_rect_area = 0.05 * W*H, 0.5* W*H
        return [rect for rect in rects if not (rect[2]*rect[3] >= min_rect_area and rect[2]*rect[3] <= max_rect_area)][:2000]
        
    def forward(self, im):
        start_t = cv2.getTickCount()
        
        # set input image on which we will run segmentation
        self.ss.setBaseImage(im)
        self.ss.switchToSelectiveSearchQuality()

        # run selective search segmentation on input image
        rects = self.ss.process()
        print('\ttotal Number of Region Proposals: {}'.format(len(rects)))
        
        # rect format: (x1, x2, w, h)
        rects = self._select_candidate_regions(im, rects)
        
        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000
        print('\trocessing time (ms): {}'.format(stop_t)) 

        return rects
    
class ImageWarp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dilation_p = 16
        self.cnn_input_shape = (3, 227, 227)
        
    def forward(self, im, rects):
        cnn_c = self.cnn_input_shape[0]
        cnn_w = self.cnn_input_shape[1]
        cnn_h = self.cnn_input_shape[2]
        im_tensor = torch.empty((len(rects), cnn_c, cnn_w, cnn_h), dtype=torch.float32)
        im_w = im.shape[1]
        im_h = im.shape[0]

        for i, rect in enumerate(rects):
            x_min = max(0, rect[0] - self.dilation_p)
            x_max = min(im_w, rect[0] + rect[2] + self.dilation_p)
            y_min = max(0, rect[1] - self.dilation_p)
            y_max = min(im_h, rect[1]+rect[3] + self.dilation_p)
            im_warped = im[y_min:y_max, x_min:x_max, :]
            im_warped = cv2.resize(im_warped, (cnn_w, cnn_h), interpolation=cv2.INTER_NEAREST)
            im_warped = np.transpose(im_warped, (2, 1, 0))
            im_tensor[i] = torch.from_numpy(im_warped)
        return im_tensor


class RProposalDataset(Dataset):
    label_format = '.txt'
    img_format = '.jpg'
    def __init__(self, path, region_proposal_method: nn.Module, image_warping_method: nn.Module, save_regions_on_picture: bool = False) -> None:
        super().__init__()

        self.path = path
        self.files = self._file_id_list()
        self.region_proposal_method = region_proposal_method
        self.warping_method = image_warping_method
        self.save_regions_on_picture = save_regions_on_picture

        self.class_tensor = self._find_classes()

    def _find_classes(self):
        classes = []
        for f_id in self.files:
            with open(os.path.join(self.path, f_id + f'{self.label_format}'), 'r') as f:
                for line in f.readlines():
                    classes.append(line.split(' ')[0])
        classes = list(set(classes))

        return torch.zeros(len(classes) + 1)

    def _file_id_list(self):
        labels_files = [file.replace(f'{self.label_format}', '') for file in os.listdir(self.path) if file.endswith(f'{self.label_format}')]
        img_files = [file.replace(f'{self.img_format}', '') for file in os.listdir(self.path) if file.endswith(f'{self.img_format}')]
        return sorted(list(set(labels_files).intersection(img_files)))

    def _assign_labels(self, im, rects, file_id):
        
        if self.save_regions_on_picture:
            im_out = im.copy()

        im_w, im_h = im.shape[1], im.shape[0]
        
        ground_truth_labels = []
        with open(os.path.join(self.path, file_id + f'{self.label_format}'), 'r') as f:
            for line in  f.readlines():
                class_, x_center_normalized, y_center_normalized, w_normalized, h_normalized = line.rstrip().split(' ')
                x_center, y_center = round(float(x_center_normalized)*im_w), round(float(y_center_normalized)*im_h)
                w, h = round(float(w_normalized)*im_w), round(float(h_normalized)*im_h)

                ground_truth_dict = {
                    'c': int(class_) + 1,   # 0 is reserved for category "no match"
                    'x1': int(x_center - w/2),
                    'y1': int(y_center - h/2),
                    'x2': int(x_center + w/2),
                    'y2': int(y_center + h/2)}
                ground_truth_labels.append(ground_truth_dict)

                if self.save_regions_on_picture:
                    cv2.rectangle(im_out, (ground_truth_dict['x1'], ground_truth_dict['y1']), (ground_truth_dict['x2'], ground_truth_dict['y2']), (0,255,0), 1, cv2.LINE_AA)
        
        labels = torch.zeros(len(rects), self.class_tensor.shape[0])
        print('\tregions vs ground truth iou:')
        for r, rect in enumerate(rects):
            rect_dict = {
                'x1': rect[0],
                'y1': rect[1],
                'x2': rect[0] + rect[2],
                'y2': rect[1] + rect[3]
            }
            labels_iou = self.class_tensor.clone()
            for grand_truth in ground_truth_labels:
                iou = get_iou(grand_truth, rect_dict)
                labels_iou[grand_truth['c']] = iou
                if iou >= 0.6:
                    print(f"\t\tr{r}: c:{grand_truth['c']}: {iou}")
                    if self.save_regions_on_picture:
                        cv2.rectangle(im_out, (rect_dict['x1'], rect_dict['y1']), (rect_dict['x2'], rect_dict['y2']), (255,0,0), 1, cv2.LINE_AA)
            labels[r] = labels_iou
        
        labels = torch.where(labels >= 0.6, 1, 0)

        if self.save_regions_on_picture:
            cv2.imwrite(self.path.replace('train', 'dev/') + f'{file_id}_v2.jpg', img=im_out)

        return labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file_id = self.files[idx]
        print(f'Processing: {file_id}')

        img_path = os.path.join(self.path, file_id + f'{self.img_format}')
        im = cv2.imread(img_path)
        
        rects = self.region_proposal_method(im)
        assert len(rects) == 2000
        
        labels = self._assign_labels(im, rects, file_id)
        
        warped_images = self.warping_method(im, rects)
        assert len(warped_images) == 2000

        return file_id, warped_images, labels

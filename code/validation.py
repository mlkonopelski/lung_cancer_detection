import argparse
import datetime
import os
import sys
import time
from timeit import timeit
from abc import ABC, abstractmethod
from typing import Dict, List
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

import numpy as np
import torch
import torch.nn as nn
from data import CandidateInfo, get_candidate_info_list, get_candidate_info_dict, Ct, CtBase, CtConversion, InferLuna2DSegmentationDataset, Luna3DClassificationDataset, BaseLuna2DSegmentationDataset
from mlmodels.classifiers import base_cnn
from mlmodels.segmentation import unet
from torch.utils.data import DataLoader
from utils.config import CONFIG
from utils.viz import IMG
from collections import namedtuple


class ConfussionMatrix:

    @staticmethod
    def calculate_scores(predictions, truth, threshold=0.5, threshold_mal=0.5):
        # Returns 3x4 confusion matrix for:
        # Rows: Truth: Non-Nodules, Benign, Malignant
        # Cols: Not Detected, Detected by Seg, Detected as Benign, Detected as Malignant
        # If one true nodule matches multiple detections, the "highest" detection is considered
        # If one detection matches several true nodule annotations, it counts for all of them
        true_nodules = [c for c in truth if c.is_nodule]
        true_diams = np.array([c.diameter_mm for c in true_nodules])
        true_xyz = np.array([c.center_xyz for c in true_nodules])

        predicted_xyz = np.array([n[2] for n in predictions])
        # detection classes will contain
        # 1 -> detected by seg but filtered by cls
        # 2 -> detected as benign nodule (or nodule if no malignancy model is used)
        # 3 -> detected as malignant nodule (if applicable)
        detected_classes = np.array([1 if d[0] < threshold
                                    else (2 if d[1] < threshold
                                        else 3) for d in predictions])

        confusion = np.zeros((3, 4), dtype=np.int)
        if len(predicted_xyz) == 0:
            for tn in true_nodules:
                confusion[2 if tn.is_mal else 1, 0] += 1
        elif len(true_xyz) == 0:
            for dc in detected_classes:
                confusion[0, dc] += 1
        else:
            normalized_dists = np.linalg.norm(true_xyz[:, None] - predicted_xyz[None], ord=2, axis=-1) / true_diams[:, None]
            matches = (normalized_dists < 0.7)
            unmatched_detections = np.ones(len(predictions), dtype=np.bool)
            matched_true_nodules = np.zeros(len(true_nodules), dtype=np.int)
            for i_tn, i_detection in zip(*matches.nonzero()):
                matched_true_nodules[i_tn] = max(matched_true_nodules[i_tn], detected_classes[i_detection])
                unmatched_detections[i_detection] = False

            for ud, dc in zip(unmatched_detections, detected_classes):
                if ud:
                    confusion[0, dc] += 1
            for tn, dc in zip(true_nodules, matched_true_nodules):
                confusion[2 if tn.is_mal else 1, dc] += 1
        return confusion

    @staticmethod
    def print_confusion(label, confusions, do_mal):
        row_labels = ['Non-Nodules', 'Benign', 'Malignant']

        if do_mal:
            col_labels = ['', 'Complete Miss', 'Filtered Out', 'Pred. Benign', 'Pred. Malignant']
        else:
            col_labels = ['', 'Complete Miss', 'Filtered Out', 'Pred. Nodule']
            confusions[:, -2] += confusions[:, -1]
            confusions = confusions[:, :-1]
        cell_width = 16
        f = '{:>' + str(cell_width) + '}'
        print(label)
        print(' | '.join([f.format(s) for s in col_labels]))
        for i, (l, r) in enumerate(zip(row_labels, confusions)):
            r = [l] + list(r)
            if i == 0:
                r[1] = ''
            print(' | '.join([f.format(i) for i in r]))


class ValTwoStepLUNAApp:
    """This is expanded version of luna.TwoStepLUNAApp where we apply Segmentation and Classification 
        models to validation data and verify performance of the whole process. 
    """
    def __init__(self, classification_model: nn.Module, segmentation_model: nn.Module) -> None:
        """Load selected models. #TODO: Write some more description of expected imput/outputs

        Args:
            classification_model (nn.Module): _description_
            segmentation_model (nn.Module): _description_
        """
        self.cls_model = classification_model.to('cpu')
        self.seg_model = segmentation_model.to(CONFIG.general.device)
        self.malignacy_model = None
        self.seg_cls_treshold = 0.5 # Increasing this value will increase true and false positives
    
    def _init_cls_dl(self, ct: CtBase, candidates: List[CandidateInfo]) -> DataLoader:
        
        ds = Luna3DClassificationDataset(ct, candidates)
        dl = DataLoader(ds, batch_size=CONFIG.cls_training.batch_size)
        
        return dl
    
    def _init_seg_dl(self, ct: CtBase, series_uid: str) -> DataLoader:
        
        ds = BaseLuna2DSegmentationDataset(ct, series_uid)
        dl = DataLoader(ds, batch_size=CONFIG.cls_training.batch_size) #FIXME: Later create a new config attribute: interferance
        return dl
    
    def group_segmentation_output(self, ct: Ct, series_uid: str, seg_mask: torch.Tensor) -> List[CandidateInfo]:
        """Grouping segmentation consists of:
            1. Labeling each "blob" with consectuive integer using ndimage.label algorithm
            2. Canculate center of each "blob" using ndimage.center_of_mass algoithm
        Because Classification DataLoader takes list of CandidateInfo tupples we convert center of "blob" 
        from IRC to XYZ with CtConversion.irc2xyz. 
        As we don't know grand truth here the resulting Tupple has False on elements:
            1. is_nodule = False
            2. has_annotation = False
            3. is_malignant = False
            4. diammeter_mm = 0.0

        Args:
            ct (Ct): CT scan object with attributes as .hu or .origin_xyz
            series_uid (str): CT identifier
            seg_mask (torch.Tensor): Raw result of segmentation model

        Returns:
            List[CandidateInfo]: List of possible candidates of nodules ready for classification model
        """
        seg_group_mask, candidate_count = ndimage.label(input=seg_mask)
        # To match the function’s (ndimage.center_of_mass) expectation that the mass is non-negative, 
        # we offset the (clipped) ct.hu_a by 1,001. Note that this leads to all flagged voxels carrying some weight, 
        # since we clamped the lowest air value to -1,000 HU in the native CT units.
        ct_input = ct.hu.clip(-1000, 1000) + 10001
        centers_irc = ndimage.center_of_mass(input=ct_input, 
                                            labels=seg_group_mask, 
                                            index=np.arange(1, candidate_count+1))
        
        candidates = []
        
        for i, center_irc in enumerate(centers_irc):
            center_xyz = CtConversion.irc2xyz(center_irc,
                                              ct.origin_xyz,
                                              ct.vx_size_xyz,
                                              ct.direction)

            assert np.all(np.isfinite(center_irc)), repr(['irc', center_irc, i, candidate_count])
            assert np.all(np.isfinite(center_xyz)), repr(['xyz', center_xyz])

            candidate = CandidateInfo(False, False, False, 0.0, series_uid, center_xyz)
            candidates.append(candidate)

        return candidates

    def segment_ct(self, ct: Ct, series_uid: str) -> torch.Tensor:
        """Apply segmentation model on each ct slice separatly to select potential nodule cells. 
        self.seg_model takes 2D arrays as imput and returns Bool 2D array of the same size therefore
        seg_mask has the same size as imput=ct.hu. 
        Additionally binary_erosion is performed to remove noise. 

        Args:
            ct (Ct): CT scan object which contains attribute .hu with raw img 
            series_uid (str): CT identifier

        Returns:
            torch.Tensor: Bool matrix with True values on pixles where candidate nodule is 
        """
        with torch.no_grad():
            output = np.zeros_like(ct.hu, dtype=np.float32)
            seg_dl = self._init_seg_dl(series_uid)
            for input, _, _, slices in seg_dl:
                input = input.to(CONFIG.general.device)
                predictions = self.seg_model(input)
                # Each input and slices is shape of batch_size. 
                for i, slice_ix in enumerate(slices):
                    output[slice_ix] = predictions[i].cpu().numpy()

            seg_mask = output > self.seg_cls_treshold
            seg_mask = ndimage.binary_erosion(seg_mask, iterations=1)
            
        return seg_mask
    
    def classify_candidates(self, ct: Ct, candidates: List[CandidateInfo]) -> List[CandidateInfo]:
        """Apply:
            1. Classification model
            2. Malignacy model
        to each nodule (cropped out 3D nodule from full ct.hu image) candidate separatly and return the result as list of 
            1. pred_nodule: <0, 1> prediction if "blob" is a nodule 
            2. pred_malignacy: <0, 1> prediction if nodule is malignant 
            3. center_xyz: center of nodule in XZY coordinates
            4. center_irc: center of nodule in IRC coordinates
        
        Args:
            ct (Ct): CT scan object with attributes as .hu or .origin_xyz
            candidates (List[CandidateInfo]): List of potential candidates for being a nodule

        Returns:
            List[tuple]: List of potential nodules with predictions and coordinates
        """
        cls_dl = self._init_cls_dl(candidates)
        
        cls_list = []
        
        for input, _, _, _, center_irc in cls_dl:
            
            input = input.to('cpu')
            with torch.no_grad():
                _, predictions_nodule = self.cls_model(input)
                if self.malignacy_model:
                    _, predictions_malignacy = self.malignacy_model(input)
                else:
                    predictions_malignacy = np.zeros_like(predictions_nodule, dtype=np.float32)
                    
            zip_iter = zip(center_irc,
                           predictions_nodule[:, 1].tolist(),
                           predictions_malignacy[:, 1].tolist())
            for c_irc, pred_nodule, pred_malignacy in zip_iter:
                c_xyz = CtConversion.irc2xyz(c_irc,
                                             ct.origin_xyz,
                                             ct.vx_size_xyz,
                                             ct.direction)
                cls_tupple = (pred_nodule, pred_malignacy, c_xyz, c_irc)
                cls_list.append(cls_tupple)

        return cls_list

    def run(self, series: str = None):
        
        val_ds = Luna3DClassificationDataset(val_stride=10, is_val_set=True)
        val_set = set([candidate.series_uid for candidate in val_ds.candidates_info])
        # pos_set = set([candidate.series_uid for candidate in val_ds.candidates_info if candidate.is_nodule]) # not used now
        
        if series:
            series_set = set(series.split(','))
        else:
            series_set = set(
                candidateInfo_tup.series_uid
                for candidateInfo_tup in get_candidate_info_list()
            )

        val_list = sorted(series_set & val_set) # In case of validating on list of series_uid instead of all
        nodules = get_candidate_info_dict()

        all_confusion = np.zeros((3, 4), dtype=np.int)
    
        for series_uid in val_list:
            ct = CtBase(series_uid)
            seg_mask = self.segment_ct(ct, series_uid)
            seg_candidates = self.group_segmentation_output(ct, series_uid, seg_mask)
            classifications = self.classify_candidates(ct, seg_candidates)
            one_confusion = ConfussionMatrix.calculate_scores(classifications, nodules)
            all_confusion += one_confusion
            ConfussionMatrix.print_confusion(series_uid, one_confusion, self.malignancy_model is not None)
        
        ConfussionMatrix.print_confusion("Total", all_confusion, self.malignancy_model is not None)


if __name__ == '__main__':
    
    EXAMPLE_UID = '1.3.6.1.4.1.14519.5.2.1.6279.6001.511347030803753100045216493273'
    
    # SEGMENTATION MODEL
    seg_dict = torch.load('.model/best/dev/UNETLuna_202310011601.best.state')
    seg_model = unet.UNETLuna()
    seg_model.load_state_dict(state_dict=seg_dict['model_state'])
    seg_model.eval()
    
    # CLASSIFICATION MODEL
    cls_dict = torch.load('.model/best/dev/CNN_202309301909.best.state')
    cls_model = base_cnn.CNN()
    cls_model.load_state_dict(state_dict=cls_dict['model_state'])
    cls_model.eval()
    
    luna_app = ValTwoStepLUNAApp(segmentation_model=seg_model, classification_model=cls_model)
    predicted_nodules = luna_app.run(EXAMPLE_UID)
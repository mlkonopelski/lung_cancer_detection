import argparse
import datetime
import os
import sys
import time
from timeit import timeit
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

import numpy as np
import torch
import torch.nn as nn
from data import CandidateInfo, Ct, CtBase, CtConversion, InferLuna2DSegmentationDataset, InferLuna3DClassificationDataset
from mlmodels.classifiers import base_cnn
from mlmodels.segmentation import unet
from torch.utils.data import DataLoader
from utils.config import CONFIG
from utils.viz import IMG
from collections import namedtuple


CandidatePred = namedtuple(typename='candidate_pred_tupple',
                           field_names='pred_nodule, pred_malignacy, center_xyz, center_irc')

class TwoStepLUNAApp:
    """Two Step Nodule identification consists of:
        1. using segmetation model to identify potential nodules out of many possible body structures
        2. using classification model to identify which of those potential nodules are actual nodues
        * 3. additional possible to distinguish Benign from Malignant nodule in future.
    The first step gives thousands of possible candidates from each CT scan while the second
    trim down this list to just few. Both models are optimized to score high on Recall since missing 
    potential nodule is worse than identyfying too many of them.
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
        # TODO: create some asserts if the model has the right architecture e.g. output of segementation should be 2D per slice
        # or check if shape of the imput equals expected by model.
        self.seg_cls_treshold = 0.5 # Increasing this value will increase true and false positives
    
    def _init_cls_dl(self, ct: CtBase, candidates: List[CandidateInfo]) -> DataLoader:
        
        ds = InferLuna3DClassificationDataset(ct, candidates)
        dl = DataLoader(ds, batch_size=CONFIG.cls_training.batch_size)
        
        return dl
    
    def _init_seg_dl(self, ct: CtBase) -> DataLoader:
        
        ds = InferLuna2DSegmentationDataset(ct)
        dl = DataLoader(ds, batch_size=CONFIG.cls_training.batch_size) #FIXME: Later create a new config attribute: interferance
        return dl
    
    def group_segmentation_output(self, ct: Ct, seg_mask: torch.Tensor) -> List[Tuple]:
        """Grouping segmentation consists of:
            1. Labeling each "blob" with consectuive integer using ndimage.label algorithm
            2. Canculate center of each "blob" using ndimage.center_of_mass algoithm
        Additionally we calculate XYZ Cooridnates of nodule for later analysis. 

        Args:
            ct (Ct): CT scan object with raw image in attribute .hu
            seg_mask (torch.Tensor): Raw result of segmentation model

        Returns:
            List[Tuple]: List of possible candidates of nodules ready for classification model
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

            candidate = (center_irc, center_xyz)
            candidates.append(candidate)

        return candidates

    def segment_ct(self, ct: Ct) -> torch.Tensor:
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
            seg_dl = self._init_seg_dl(ct)
            for input, slices in seg_dl:
                input = input.to(CONFIG.general.device)
                predictions = self.seg_model(input)
                # Each input and slices is shape of batch_size. 
                for i, slice_ix in enumerate(slices):
                    output[slice_ix] = predictions[i].cpu().numpy()

            seg_mask = output > self.seg_cls_treshold
            seg_mask = ndimage.binary_erosion(seg_mask, iterations=1)
            
        return seg_mask
    
    def classify_candidates(self, ct: Ct, candidates: List[CandidateInfo]) -> List[CandidatePred]:
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
        cls_dl = self._init_cls_dl(ct, candidates)
        
        cls_list = []
        
        for input, center_irc, center_xyz in cls_dl:
            
            input = input.to('cpu')
            with torch.no_grad():
                _, predictions_nodule = self.cls_model(input)
                if self.malignacy_model:
                    _, predictions_malignacy = self.malignacy_model(input)
                else:
                    predictions_malignacy = np.zeros_like(predictions_nodule, dtype=np.float32)
                    
            zip_iter = zip(center_irc.tolist(),
                           center_xyz.tolist(),
                           predictions_nodule[:, 1].tolist(),
                           predictions_malignacy[:, 1].tolist())
            for c_irc, c_xyz, pred_nodule, pred_malignacy in zip_iter:
                cls_tupple = CandidatePred(pred_nodule, pred_malignacy, c_xyz, c_irc)
                cls_list.append(cls_tupple)

        return cls_list

    def run(self, series_uid: str):
        
        ct = CtBase(series_uid)
        seg_mask = self.segment_ct(ct)
        candidates = self.group_segmentation_output(ct, seg_mask)
        classification = self.classify_candidates(ct, candidates)
        nodules = [c for c in classification if c.pred_nodule > 0.5]
        nodules = [(ct.hu[int(n.center_irc[0])], seg_mask[int(n.center_irc[0])], n.center_irc, n.center_xyz) for n in nodules]
        return nodules

        
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
    
    luna_app = TwoStepLUNAApp(segmentation_model=seg_model, classification_model=cls_model)
    predicted_nodules = luna_app.run(EXAMPLE_UID)

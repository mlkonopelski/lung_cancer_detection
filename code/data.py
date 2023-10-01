import copy
import csv
import glob
import math
import os
import random
from collections import namedtuple
from functools import lru_cache
from typing import Dict, List, NamedTuple, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from utils.disk import getCache
from utils.viz import IMG

raw_cache = getCache('ct_chunk_raw')

CandidateInfo = namedtuple(typename='candidate_info_tupple',
                           field_names='is_nodule, has_annotation, is_malignant, diameter_mm, series_uid, center_xyz')

@lru_cache(1)
def get_candidate_info_list(require_on_disk: bool = True, random_sample: bool = False) -> List[NamedTuple]:
    '''
    Parse files:
        annotations.csv
        candidates.csv
    in order to obtain a complete list of potential nodules. 
    
    Each entry in resulting list: candidates_info is consisted of 4 elements (named tupple):
        - is_nodule: bool - is this nodule malignment or begin
        - diameter_mm: float32 - size of nodule in mm. If no information then size: 0.0. 
            If significant difference in xyz coordiantes of nodule in both files then omit this file.
        - series_uid: str - unique identifier of each CT scan
        - center_xyz: tupple - coordinates of center of nodule
        
    List of candidates is sorted from largest based on diameter_mm. 
    
    Arguments:
        require_on_disk: str - Process only CT scans from candidates file which
            exists on disc. This is used mainly to speed up the process in case if
            we want to test the script on only few examples. 
    Returns:
        candidates_info: List[NamedTuple]
    '''
    mhd = glob.glob(pathname='.data/*/*.mhd')
    present_on_disk = {os.path.split(filepath)[-1][:-4] for filepath in mhd}
    
    if random_sample:
        present_on_disk_series = list(present_on_disk.keys())
        sample_keys = random.choices(present_on_disk_series, k=min(len(present_on_disk_series), 10))
        present_on_disk = {k:v for k, v in present_on_disk.items() if k in sample_keys} 
    
    annotation_info = {}
    with open('.data/annotations.csv', 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotation_center_xyz = tuple([float(x) for x in row[1:4]])
            annotation_diameter_mm = float(row[4])
            annotation_info.setdefault(series_uid, []).append(
                (annotation_center_xyz, annotation_diameter_mm))

    candidates_info = []
    
    # This file is prepared by authors and include removed duplication of certain nodule localisations
    with open('.data/annotations_with_malignancy.csv') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in present_on_disk and require_on_disk:
                continue
            annotation_center_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            is_malignant = {'False': False, 'True': True}[row[5]]
            candidates_info.append(
            CandidateInfo(
                True, # is_nodule,
                True, # has_annotation
                is_malignant,
                annotationDiameter_mm,
                series_uid,
                annotation_center_xyz,
            )
            )
    
    with open('.data/candidates.csv', 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in present_on_disk and require_on_disk:
                continue
            is_nodule = bool(int(row[4]))
            candidate_center_xyz = tuple([float(x) for x in row[1:4]])
            # As default diameter of nodule is set = 0 because it will be used only for 
            # balanced train/test split.
            candidate_diameter_mm = 0.0
            # if candidate exists in annotations set (we might use diameter from it)
            for annotation in annotation_info.get(series_uid, []): 
                annotation_center_xyz, annotation_diameter_mm = annotation
                for i in range(3):
                    delta_mm = abs(candidate_center_xyz[i] - annotation_center_xyz[i])
                    # if discrepency in coordinates between two files are bigger than 1/2 of radius of nodule 
                    # (in any of the dimensions) set diameter_mm as default = 0.0 even if exists in annotations data
                    if delta_mm > annotation_diameter_mm / 4:
                        break
                    else:
                        candidate_diameter_mm = annotation_diameter_mm
                        break
                    
            # We include only non-nodules because nodules are provided by different file
            if not is_nodule:
                candidates_info.append(
                    CandidateInfo(
                        False, # is_nodule,
                        False, # has_annotation
                        False, # is_malignant
                        candidate_diameter_mm,
                        series_uid,
                        candidate_center_xyz
                    )
                )
            
    candidates_info.sort(reverse=True)
            
    return candidates_info

@lru_cache(1)
def get_candidate_info_dict(requireOnDisk_bool=True):
    candidate_info_list = get_candidate_info_list(requireOnDisk_bool)
    candidate_info_dict = {}

    for candidateInfo_tup in candidate_info_list:
        candidate_info_dict.setdefault(candidateInfo_tup.series_uid,
                                      []).append(candidateInfo_tup)

    return candidate_info_dict

IRCTuple = namedtuple('IRCTuple', ['index', 'row', 'col'])
XYZTupple = namedtuple('XYZTupple', ['x', 'y', 'z'])

class CtConversion:
    '''
    Helper methods for converting resolutions from XYZ (millimetrs) to IRC (voxel) and back
    '''
    @staticmethod
    def irc2xyz(coord_irc, origin_xyz, vx_size_xyz, direction):
        cri = np.array(coord_irc)[::-1] # Flip the coordinates from IRC to CRI to align with XYZ
        origin = np.array(origin_xyz)
        vx_size_xyz = np.array(vx_size_xyz)
        # 3 steps plan:
        ## Scale the indices with the voxel sizes.
        ## Matrix-multiply with the directions matrix, using @ in Python.
        ## Add the offset for the origin.
        coord_xyz = (direction @ (cri * vx_size_xyz)) + origin 
        return XYZTupple(*coord_xyz)
    
    @staticmethod
    def xyz2irc(coord_xyz: ArrayLike, origin_xyz: NamedTuple, vx_size_xyz: NamedTuple, direction: ArrayLike) -> NamedTuple:
        origin_xyz = np.array(origin_xyz)
        vx_size_xyz = np.array(vx_size_xyz)
        coord_xyz = np.array(coord_xyz)
        # 3 steps plan:
        ## Substruct the offset for the origin.
        ## Matrix-multiply with the directions matrix, using @ in Python.
        ## Scale the indices with the voxel sizes.
        cri = ((coord_xyz - origin_xyz) @ np.linalg.inv(direction)) / vx_size_xyz
        cri = np.round(cri) # Add rounding before converting to integers
        return IRCTuple(int(cri[2]), int(cri[1]), int(cri[0])) # Flip the coordinates from cri to irc to align with XYZ
    
    @staticmethod
    def print_histogram(array: ArrayLike, title: str = 'Histogram of voxel values') -> None:
        """Helper function to print histogram (matplotlib.pyplot) of voxel values before and after conversion

        Args:
            array (ArrayLike): _description_
            title (str, optional): _description_. Defaults to 'Histogram of voxel values'.
        """
        
        counts, bins = np.histogram(array.reshape(-1))
        plt.stairs(counts, bins)
        plt.title(label=title)
        plt.show()


class Ct:
    """_summary_
    """
    def __init__(self, series_uid: str) -> None:
        mdh_path = glob.glob(f'.data/subset*/{series_uid}.mhd')[0]
        # Original files are in MetaIO format: "https://itk.org/Wiki/MetaIO/""
        # and therefore sitk library is used for converting .mdh + .raw files into array
        # Hovewer teh native format for CT scans in DICOM: "https://dicomstandard.org" but unfortunately it's not 
        # standarized enough.
        mdh_img = sitk.ReadImage(mdh_path)
        mdh_array = np.array(sitk.GetArrayFromImage(mdh_img), dtype=np.float32)
        # voxels need to be expressed in HU units: “https://en.wikipedia.org/wiki/Hounsfield_scale”
        # however our data shows negative values lower than -1000 (density of air) and higher than 1000 (density of bones)
        # in cases respectively of object outside a scanner and e.g. metal implants. However for our purpose we 
        # we dont need to create this division therefore we can limit the range <-1000, 10000>. Even if this will not 
        # be accurate biologically, it is a method to remove outliers.
        mdh_array.clip(-1000, 1000, mdh_array)
        self.hu = torch.from_numpy(mdh_array).to(torch.float32)
        self.series_uid = series_uid
        # Conversion from XYZ to IRC
        self.origin_xyz = XYZTupple(*mdh_img.GetOrigin()) 
        self.vx_size_xyz = XYZTupple(*mdh_img.GetSpacing())
        ## Flipping the axes (and potentially a rotation or other transforms) is encoded in a 3 × 3 matrix
        self.direction = np.array(mdh_img.GetDirection()).reshape(3, 3)
        
        # Prepare data for Segmentation
        ## List of actual nodules
        candidates = get_candidate_info_dict()[self.series_uid]
        positive_candidates = [c for c in candidates if c.diameter_mm != 0.0 and c.is_nodule == True]
        self.positive_mask = self.build_annotation_mask(nodule_list = positive_candidates)
        self.positive_indexes = self.positive_mask.sum(axis=(1,2)).nonzero().tolist()                         

        
    def get_raw_candidate(self, center_xyz, width_irc) -> Tuple[ArrayLike, ArrayLike]:
        """Use nodule center as b ase to cut of chunk od the original picture for classifications

        Args:
            center_xyz (_type_): cooridinates of center fo nodule
            width_irc (_type_): Default size of chunk that we want to cut from original picture

        Returns:
            Tupple[ArrayLike, ArrayLike]: (New chunk with nodule in the middle, center of nodule using IRC coordinates)
        """
        center_irc = CtConversion.xyz2irc(center_xyz, 
                                          self.origin_xyz, 
                                          self.vx_size_xyz, 
                                          self.direction)
        slice_list = []
        for axis, center_val in enumerate(center_irc):
            # Based on center_irc and width_irc calculate start and end of chunk ix.
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])
            
            assert center_val >= 0 and center_val < self.hu.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vx_size_xyz, center_irc, axis])

            # In case of nodule center close to left min column, 
            # chunk should start from 0 and have width of width_irc
            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            # In case of nodule center close to right max column,
            # chunk should start from maximum column - width_irc
            if end_ndx > self.hu.shape[axis]:
                end_ndx = self.hu.shape[axis]
                start_ndx = int(self.hu.shape[axis] - width_irc[axis])
            
            slice_list.append(slice(start_ndx, end_ndx))
        ct_chunk = self.hu[tuple(slice_list)] # Use slices in 3 dimensions to cut chunk out of whole Ct Scan
        pos_slices = self.positive_mask[tuple(slice_list)] # Use slices in 3 dimensions to choose positive mask of each nodule
        return ct_chunk, center_irc, pos_slices
    
    def build_annotation_mask(self, nodule_list, threshold_hu: int = -700):
        
        # Bounding box templat
        annotated_mask = torch.zeros(self.hu.shape, dtype=torch.bool)
        
        for c_ix, candidate in enumerate(nodule_list):
            center_irc = CtConversion.xyz2irc(candidate.center_xyz, self.origin_xyz, self.vx_size_xyz, self.direction)
            ci, cr, cc = center_irc.index, center_irc.row, center_irc.col
            
            index_radius = 2
            try:
                while self.hu[ci - index_radius, cr, cc] > threshold_hu and self.hu[ci + index_radius, cr, cc] > threshold_hu:
                    index_radius+=1
            except IndexError:
                index_radius -= 1
                
            row_radius = 2
            try:
                while self.hu[ci, cr - row_radius, cc] > threshold_hu and self.hu[ci, cr + row_radius, cc] > threshold_hu:
                    row_radius+=1
            except IndexError:
                row_radius -= 1
                
            col_radius = 2
            try:
                while self.hu[ci, cr, cc - col_radius] > threshold_hu and self.hu[ci, cr, cc + col_radius] > threshold_hu:
                    col_radius+=1
            except IndexError:
                col_radius -= 1
                
            annotated_mask[ci-index_radius:ci+index_radius+1, 
                           cr-row_radius:cr+row_radius+1, 
                           cc-col_radius:cc+col_radius+1] = True
            
        annotated_mask = annotated_mask & (self.hu > threshold_hu)
            
        return annotated_mask
    

@lru_cache(1, typed=True)
def get_ct(series_uid: str):
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def get_ct_raw_candidate(series_uid: str, center_xyz: NamedTuple, width_irc: Tuple) -> Tuple:
    ct = get_ct(series_uid)
    ct_chunk, center_irc, pos_slices = ct.get_raw_candidate(center_xyz, width_irc)
    return ct_chunk, pos_slices, center_irc


@raw_cache.memoize(typed=True)
def get_ct_sample_size(series_uid: str):
    ct = Ct(series_uid)
    return int(ct.hu.shape[0]), ct.positive_indexes


def get_ct_augmented_candidate(
    augmentation: Dict,
    series_uid: str,
    center_xyz: NamedTuple,
    width_irc: Tuple,
    use_cache: bool = True
):
    if use_cache:
        ct_chunk, _, center_irc = get_ct_raw_candidate(series_uid, center_xyz, width_irc)
    else:
        ct = get_ct(series_uid)
        ct_chunk, _, center_irc = ct.get_raw_candidate(center_xyz, width_irc)
    ct_chunk = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)
    
    transform_t = torch.eye(n=4)
    
    for i in range(3):
        if 'flip' in augmentation:
            if random.random() > 0.5:
                transform_t[i, i] *= -1
        if 'offset' in augmentation:
            offset_float = augmentation['offset']
            random_float = (random.random() * 2 - 1)
            transform_t[i, 3] = offset_float * random_float            
        if 'scale' in augmentation:
            scale_float = augmentation['scale']
            random_float = (random.random() * 2 -1)
            transform_t[i, i] = 1.0 + scale_float * random_float
        if 'rotate' in augmentation:
            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)
            rotation_t = torch.tensor([
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            transform_t @= rotation_t
        if 'noise' in augmentation:
            noise_t = torch.randn_like(ct_chunk)
            noise_t *= augmentation['noise']
            ct_chunk += noise_t
    
    affine_t = F.affine_grid(
        transform_t[:3].unsqueeze(0).to(torch.float32),
        ct_chunk.size(),
        align_corners=False,
    )
    augmented_chunk = F.grid_sample(
        input=ct_chunk,
        grid=affine_t,
        padding_mode='border',
        align_corners=False,
    ).to('cpu')

    return augmented_chunk[0], center_irc
    
def validate_augmentations(augmentations: Dict) -> Dict:
    
    user_augmentations = augmentations.keys()
    expected_augmentations = set(['flip', 'offset', 'scale', 'rotation', 'noise'])
    unsupported_augmentations = set(user_augmentations).difference(expected_augmentations)
    if unsupported_augmentations:
        for unsupported_augmentation in list(unsupported_augmentations):
            augmentations.pop(unsupported_augmentation)
            raise Warning(f'Augmentation: {unsupported_augmentation} is not supported and will be removed from list')
    if 'scale' in user_augmentations:
        if augmentations['scale'] <0 or augmentations['scale'] > 1:
            augmentations.pop('scale')
            raise Warning(f'Scaling ratio out of range <0, 1> and will be removed from list')
    if 'offset' in user_augmentations:
        if augmentations['offset'] <0 or augmentations['offset'] > 1:
            augmentations.pop('offset')
            raise Warning(f'Offseting ratio out of range <0, 1> and will be removed from list')
    if 'noise' in user_augmentations:
        if augmentations['noise'] <= 0 or augmentations['noise'] >= 100:
            augmentations.pop('noise')
            raise Warning(f'Noise ratio out of range <0, 100> and will be removed from list')

    return augmentations
    

class Luna3DClassificationDataset(Dataset):
    def __init__(self, 
                 val_stride: int = 0,
                 is_val_set: bool = None,
                 ratio: int = 0,
                 augmentations: Dict = {},
                 series_uid=None) -> None:
        super().__init__()
        # Copy the return value of get_candidate_info_list func so the 
        # cached copy won’t be impacted by altering self.candidates_info
        self.candidates_info = copy.copy(get_candidate_info_list())
        # Handle to process only 1 example (for dev)
        if series_uid:
            self.candidates_info = [x for x in self.candidates_info if x.series_uid == series_uid]
        if is_val_set:
            assert val_stride > 0, val_stride
            self.candidates_info[::val_stride]
        elif val_stride > 0:
            del self.candidates_info[::val_stride]
            assert self.candidates_info
        # help with over
        self.augmentations = validate_augmentations(augmentations)
        # variables useful for balancing the dataset
        self.ratio = ratio
        self.negative_list = [nt for nt in self.candidates_info if not nt.is_nodule]
        self.positive_list = [nt for nt in self.candidates_info if nt.is_nodule]
    
    def __len__(self):
        return len(self.candidates_info)

    def shuffle_samples(self):
        if self.ratio:
            random.shuffle(self.negative_list)
            random.shuffle(self.positive_list)
    
    def __getitem__(self, ix: int) -> Tuple:
        """
        If order to balance the dataset we implement artificially chaning the order of dataset ix 
        which is pulled from Dataset. E.g. in case we want to have 2:1 ratio of negative to positive,
        we would take samples in following order:
        DS Ix   0 1 2 3 4 5 6 7 8 9 ...
        Label   + - - + - - + - - + 
        Pos Ix  0     1     2     3
        Neg Ix    0 1   2 3   4 5 
        So the method is literally jumping over ix in order to receive this ratio. In order for not sampling
        the same few first examples of majority class. Original lists are shuffled with each opoch. 
        
        Args:
            ix (int): Iterator goes from ix=0 to len(dataset)

        Returns:
            Tuple: with 4 elements chategorizing each sample: (X, y, uid, center_irc)
        """
        if self.ratio:
            pos_ix = ix // (self.ratio + 1)
            if ix % (self.ratio + 1):
                neg_ix = ix - 1 - pos_ix
                neg_ix %= len(self.negative_list)
                candidate_info = self.negative_list[neg_ix]
            else:
                pos_ix %= len(self.positive_list)
                candidate_info = self.positive_list[pos_ix]
        else:
            candidate_info = self.candidates_info[ix]

        width_irc = (32, 48, 48)

        ct_chunk, center_irc = get_ct_augmented_candidate(augmentation=self.augmentations,
                                                          series_uid=candidate_info.series_uid,
                                                          center_xyz=candidate_info.center_xyz,
                                                          width_irc=width_irc
                                                          )
        
        label = torch.tensor([
            not candidate_info.is_nodule,
            candidate_info.is_nodule
        ],
            dtype=torch.long
        )
        
        return (
            ct_chunk,
            label,
            candidate_info.series_uid,
            torch.tensor(center_irc)
            
        ) 


class BaseLuna2DSegmentationDataset(Dataset):
    def __init__(self,
                 is_val_set: bool, 
                 val_stride: int,
                 series_uid: str = None, 
                 process_full_ct: bool = False,
                 context_slice_count: int = 3) -> None:
        
        # variables
        self.process_full_ct = process_full_ct
        self.context_slice_count = context_slice_count
        # generate list of series to include it Dataset
        if not series_uid:
            self.series = sorted(get_candidate_info_dict().keys())
        else:
            self.series = [series_uid]
            
        print(f'len(series) before subset = {len(self.series)}')
        
        # Select Training or Validation version of this Dataset
        if is_val_set:
            assert val_stride > 0, val_stride
            self.series = self.series[::val_stride] # Starting with a series list containing all our series, we keep only every val_stride-th element, starting with 0
            assert self.series
        elif val_stride >0:
            # TODO: In case of Development of only 1 series_uid it fails here because tehre are no Val examples to remove from this list.
            if not self.series:
                del self.series[::val_stride]
                assert self.series
        
        #print(f'len(series) after subset = {len(self.series)}')
        
        # Two modes for training 
        self.sample_list = []
        self.index_count_summary = []
        for series_uid in self.series:
            index_count, positive_indexes = get_ct_sample_size(series_uid)
            self.index_count_summary.append(index_count)
            ## If we want to process pictures with original all channels [147, :, :]
            ## This will be useful when we’re evaluating end-to-end performance, since we need to pretend that we’re starting off with no prior information about the CT.
            if self.process_full_ct:
                self.sample_list += [(series_uid, idx) for idx in range(index_count)]
            ## If we want to cut only channels which includes positive nodule
            ## We will use for validation during training, which is when we’re limiting ourselves to only the CT slices that have a positive mask present.
            else:
                self.sample_list += [(series_uid, idx) for idx in positive_indexes]
                
        #print(f'len(sample_list) after subset = {len(set([s[0] for s in self.sample_list]))}')
        
        # prepare list of candidates
        self.all_candidates = get_candidate_info_list()
        #print(f'len(self.all_candidates) = {len(set([c.series_uid for c in self.all_candidates]))}')
        self.candidates = [candidate for candidate in self.all_candidates if candidate.series_uid in set(self.series)]
        #print(f'len(self.candidates) = {len(set([c.series_uid for c in self.candidates]))}')
        self.positive_candidates = [candidate for candidate in self.candidates if candidate.is_nodule]
        #print(f'len(self.positive_candidates) = {len(set([c.series_uid for c in self.positive_candidates]))}')
                
    def get_full_slice(self, series_uid: str, slice_ndx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        """Based on localisation index of nodule [slice_ndx] choose 3 slices before and after as 
        input for the model. In case the slice_ndx is close to first or last slice make sure it's 
        not out of range. 
        
        So ct_t will be input to model as 2D slice of MRI picture with 7 channels of surrounding slices. 
        The output is 2D matrix with segmentation [True/False] of particular slice_ndx slice. 

        Args:
            series_uid (str): Series identifier
            slice_ndx (int): SLice identifier

        Returns:
            Tuple: Tuple[ct_t, pos_t, series_uid, slice_ndx]
        """
        ct = get_ct(series_uid)
        ct_t = torch.zeros((self.context_slice_count * 2 + 1 , 512, 512))
        
        start_ndx = slice_ndx - self.context_slice_count
        end_ndx = slice_ndx + self.context_slice_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0) # make sure it's not outside picture on left
            context_ndx = min(context_ndx, ct.hu.shape[0] - 1) # make sure it's not outside picture on right
            ct_t[i] = ct.hu[context_ndx].to(torch.float32)
        
        ct_t.clamp_(-1000, 1000)

        pos_t = ct.positive_mask[slice_ndx].unsqueeze(0)

        return ct_t, pos_t, ct.series_uid, slice_ndx
                    
    def __len__(self):
        return len(self.sample_list)
        
    def __getitem__(self, ndx: int) -> Tuple:
        """
        Args:
            ndx (int): 

        Returns:
            Tuple: 
        """
        
        series_uid, slice_ndx  = self.sample_list[ndx % len(self.sample_list)]
        return self.get_full_slice(series_uid, slice_ndx[0])


class TrainLuna2DSegmentationDataset(BaseLuna2DSegmentationDataset):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

            self.ratio_int = 2
            self.target_size = (7, 96, 96)
            
        def get_cropped_slice(self, candidate: CandidateInfo) -> Tuple[torch.Tensor, torch.Tensor, str, int]:

            ct_chunk, pos_slices, center_irc = get_ct_raw_candidate(candidate.series_uid, 
                                                                    candidate.center_xyz, 
                                                                    self.target_size)
            
            pos_slice = pos_slices[3:4] # [1, 96, 96]
            row_offset = random.randrange(0, 32)
            col_offset = random.randrange(0, 32)
            
            ct_chunk = ct_chunk[:, row_offset:row_offset+64, col_offset:col_offset+64]
            pos_slice = pos_slice[:, row_offset:row_offset+64, col_offset:col_offset+64]

            return ct_chunk, pos_slice, candidate.series_uid, center_irc.index
            
            
        def __getitem__(self, ndx: int) -> Tuple:
            candidate  = self.positive_candidates[ndx % len(self.positive_candidates)]
            return self.get_cropped_slice(candidate)


class SegmentationAugmentation(nn.Module):
    def __init__(self, flip: bool = None, 
                 offset: float = None, 
                 scale: float = None, 
                 rotate: bool = None, 
                 noise: float = None) -> None:
        """Augment input data before feeding to NN. 

        Args:
            flip (bool, optional): Flip the image. Defaults to None.
            offset (float, optional): Move image by ratio in any direction. Defaults to None.
            scale (float, optional): Scale - make it bigger or smaller by ratio. Defaults to None.
            rotate (bool, optional): Rotate around by random angle. Defaults to None.
            noise (float, optional): Add random noise to the image. The higher ratio - more noise. Defaults to None.
        """

        super(SegmentationAugmentation, self).__init__()
        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise
        
    def _build2Dtransformation(self):
        
        transformation = torch.eye(3)
        
        for i in range(2):
            
            if self.flip:
                if random.random() > 0.5:
                    transformation[i,i] *= -1
            if self.offset:
                offset_float = self.offset
                random_float = (random.random() * 2 - 1)
                transformation[2,i] = offset_float * random_float
            if self.scale:
                scale_float = self.scale
                random_float = (random.random() * 2 - 1)
                transformation[i,i] *= 1.0 + scale_float * random_float
            if self.rotate:
                angle_rad = random.random() * math.pi * 2
                s = math.sin(angle_rad)
                c = math.cos(angle_rad)
                rotation = torch.Tensor([
                    [c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1]
                ])
                transformation @= rotation    
        return transformation

    def forward(self, input, label):
        
        transformation = self._build2Dtransformation()
        transformation = transformation.expand(input.shape[0], -1, -1)
        transformation = transformation.to(input.device, dtype=torch.float32)
        
        affine_grid = F.affine_grid(theta=transformation[:,:2], size=input.size(), align_corners=False)
        
        augmented_input = F.grid_sample(input=input, grid=affine_grid, padding_mode='border', align_corners=False)
        augmented_label = F.grid_sample(input=label.to(torch.float32), grid=affine_grid, padding_mode='border', align_corners=False)
        
        if self.noise:
            noise_transformation = torch.rand_like(augmented_input)
            noise_transformation *= self.noise
            augmented_input += noise_transformation
            
        return augmented_input, augmented_label > 0.5 # Just before returning, we convert the mask back to Booleans by comparing to 0.5. The interpolation that grid_sample results in fractional values.

    
if __name__ == '__main__':
    
    # CONFIG
    show_images = False
    
    # -----------------------------------------------------------------------------------
    # EXPLORE RAW DATA
    # -----------------------------------------------------------------------------------
    
    print('-' * 20, '\n', 'EXPLORE RAW DATA', '\n','-' * 20)
    candidates = get_candidate_info_list()
    print(f'{len(candidates)} candidates nodules for {len(list(set([c.series_uid for c in candidates])))} files.')
    
    EXAMPLE_UID = '1.3.6.1.4.1.14519.5.2.1.6279.6001.511347030803753100045216493273'
    #EXAMPLE_UID = '1.3.6.1.4.1.14519.5.2.1.6279.6001.277445975068759205899107114231'
    
    # Convert RAW image to Array
    img = Ct(series_uid=EXAMPLE_UID)
    if show_images:
        CtConversion.print_histogram(array=img.hu)
    
    # Create chunk around nodule coordinates
    example_candidate = [c for c in candidates if c.series_uid == EXAMPLE_UID and c.diameter_mm != 0.0][0]
    width_irc = (32, 48, 48)
    ct_chunk, center_irc, pos_slices  = img.get_raw_candidate(center_xyz=example_candidate.center_xyz, 
                                                 width_irc=width_irc)
    print(f'Chunk size: {ct_chunk.shape} and center: {center_irc}')
    
    if show_images:
        CtConversion.print_histogram(array=ct_chunk)
    
    # Print full pictire and chunk in one plot for comparison:
    if show_images:
        IMG.img_by_chunk_sidebyside(img.hu, 
                                    ct_chunk, 
                                    center_irc, 
                                    fig_title=EXAMPLE_UID, 
                                    add_rectangular=[center_irc, round(example_candidate.diameter_mm)])
        
    # Verify sizes of images
    candidates_series = list(set([c.series_uid for c in candidates]))
    candidate_indexes = [Ct(series_uid).hu.shape for series_uid in candidates_series]
    
    # -----------------------------------------------------------------------------------
    # LUNA DATASET FOR CLASSIFICATION
    # -----------------------------------------------------------------------------------
    
    # Test Luna Dataset
    print('-' * 20, '\n', 'LUNA DATASET FOR CLASSIFICATION', '\n','-' * 20)
    from utils.config import CONFIG

    train_ds = Luna3DClassificationDataset(val_stride=10,
                            is_val_set=False,
                            augmentations=CONFIG.training.classification_augmentation)
    
    print(f'LEN of LUNA DATASET: {len(train_ds)}')
    
    # Example of pictures and different augmentations.
    ct_chunk_original, _, _ = img.get_raw_candidate(center_xyz=example_candidate.center_xyz, 
                                                 width_irc=width_irc)
    ct_chunk_augmented, _ = get_ct_augmented_candidate(augmentation={'flip': True}, 
                                                       series_uid=EXAMPLE_UID,
                                                       center_xyz=example_candidate.center_xyz,
                                                       width_irc=width_irc,
                                                       use_cache=False)
    # IMG.single_image(ct_chunk_original[16])
    # IMG.single_image(ct_chunk_augmented[0][16])
    
    from utils.config import CONFIG
    if show_images:
        IMG.visualize_augmentations(func=get_ct_augmented_candidate,
                                    series_uid=EXAMPLE_UID,
                                    center_xyz=example_candidate.center_xyz,
                                    width_irc=width_irc,
                                    augmentations=CONFIG.training.classification_augmentation)
    
    
    # Loading DAtasets to DataLoader
    train_dl = DataLoader(train_ds, 
                        batch_size=8,
                        shuffle=True,
                        #num_workers=1,
                        pin_memory=False,
                        )

    for batch_idx, batch in enumerate(train_dl):
        X, y, series_uid, center_list = batch
        print(X.shape, y.shape)
        print(len(series_uid), len(center_list))
        break
    
    # -----------------------------------------------------------------------------------
    # LUNA DATASET FOR SEGMENTATION
    # -----------------------------------------------------------------------------------
    
    # Calculate and Visualize segmentation mask
    print('-' * 20, '\n', 'LUNA DATASET FOR SEGMENTATION', '\n','-' * 20)
    segmentation_mask = img.build_annotation_mask([example_candidate])
    if show_images:
        IMG.visualize_mask(img.hu[center_irc.index,:, :], mask_array=segmentation_mask[center_irc.index, :, :])
    
    # Build Dataset for model
    # Training Dataset
    segmentation_train_ds = TrainLuna2DSegmentationDataset(is_val_set=False, val_stride=10)
    input_t, label_t, series_uid, slice_ndx = segmentation_train_ds[0]
    ## Valdiation Dataset
    segmentation_val_ds = BaseLuna2DSegmentationDataset(is_val_set=True, val_stride=10)
    input_t_val, label_t_val, series_uid_val, slice_ndx_val = segmentation_val_ds[0]
    
    # Test and visualize augmentations
    ## prepare data to visualize
    show_images = True
    simple_input = img.hu[50]
    simple_label = torch.ones(simple_input.shape[0], simple_input.shape[1]).triu()
    cols = int(simple_label.shape[1])
    for i in range(int(cols / 2), cols):
        simple_label[:, i] = 0.0
    for i in range(0, int(cols / 4)):
        simple_label[i, :] = 0.0
    if show_images:
        plt.title(label='Original')
        plt.imshow(simple_input, cmap='gray', interpolation='nearest')
        plt.imshow(simple_label, cmap='bwr', alpha=0.5)
        plt.show()
    simple_input, simple_label = simple_input.unsqueeze(0).unsqueeze(0), simple_label.unsqueeze(0).unsqueeze(0)
        
    flip_augmentations = SegmentationAugmentation(flip=True)
    offset_augmentations = SegmentationAugmentation(offset=0.1)
    scale_augmentations = SegmentationAugmentation(scale=0.2)
    rotate_augmentations = SegmentationAugmentation(rotate=True)
    noise_augmentations = SegmentationAugmentation(noise=5.0)
    
    flip_augmented = flip_augmentations(simple_input, simple_label) # Expected: (1, I, R, C), (1, 1, R, C),
    if show_images:
        plt.title(label='Flipped')
        plt.imshow(flip_augmented[0][0][0], cmap='gray', interpolation='nearest')
        plt.imshow(flip_augmented[1][0][0], cmap='bwr', alpha=0.5)
        plt.show()
    offset_augmented = offset_augmentations(simple_input, simple_label)
    if show_images:
        plt.title(label='Offset')
        plt.imshow(offset_augmented[0][0][0], cmap='gray', interpolation='nearest')
        plt.imshow(offset_augmented[1][0][0], cmap='bwr', alpha=0.5)
        plt.show()
    scale_augmented = scale_augmentations(simple_input, simple_label)
    if show_images:
        plt.title(label='Scaled')
        plt.imshow(scale_augmented[0][0][0], cmap='gray', interpolation='nearest')
        plt.imshow(scale_augmented[1][0][0], cmap='bwr', alpha=0.5)
        plt.show()
    rotate_augmented = rotate_augmentations(simple_input, simple_label)
    if show_images:
        plt.title(label='Rotated')
        plt.imshow(rotate_augmented[0][0][0], cmap='gray', interpolation='nearest')
        plt.imshow(rotate_augmented[1][0][0], cmap='bwr', alpha=0.5)
        plt.show()
    noise_augmented = noise_augmentations(simple_input, simple_label)
    if show_images:
        plt.title(label='Noised')
        plt.imshow(noise_augmented[0][0][0], cmap='gray', interpolation='nearest')
        plt.imshow(noise_augmented[1][0][0], cmap='bwr', alpha=0.5)
        plt.show()

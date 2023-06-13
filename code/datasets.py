from collections import namedtuple
from functools import lru_cache
import glob
import os
import csv
import SimpleITK as sitk
from typing import List, NamedTuple, Tuple
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import copy
from utils.disk import getCache
from utils.viz import IMG


raw_cache = getCache('ct_chunk_raw')

CandidateInfo = namedtuple(typename='candidate_info_tupple',
                           field_names='is_nodule, diameter_mm, series_uid, center_xyz')

@lru_cache(1)
def get_candidate_info_list(require_on_disk: bool = True) -> List[NamedTuple]:
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
    
    annotation_info = {}
    with open('.data/annotations.csv', 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotation_center_xyz = tuple([float(x) for x in row[1:4]])
            annotation_diameter_mm = float(row[4])
            annotation_info.setdefault(series_uid, []).append(
                (annotation_center_xyz, annotation_diameter_mm))

    candidates_info = []
    
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
                    
            candidates_info.append(
                CandidateInfo(
                    is_nodule,
                    candidate_diameter_mm,
                    series_uid,
                    candidate_center_xyz
                )
            )
            
    candidates_info.sort(reverse=True)
            
    return candidates_info


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
        self.hu = mdh_array
        self.series_uid = series_uid
        # Conversion from XYZ to IRC
        self.origin_xyz = XYZTupple(*mdh_img.GetOrigin()) 
        self.vx_size_xyz = XYZTupple(*mdh_img.GetSpacing())
        ## Flipping the axes (and potentially a rotation or other transforms) is encoded in a 3 × 3 matrix
        self.direction = np.array(mdh_img.GetDirection()).reshape(3, 3)

        
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
        return ct_chunk, center_irc
    

@lru_cache(1, typed=True)
def get_ct(series_uid: str):
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def get_ct_raw_candidate(series_uid: str, center_xyz: NamedTuple, width_irc: Tuple) -> Tuple:
    ct = get_ct(series_uid)
    ct_chunk, center_irc = ct.get_raw_candidate(center_xyz, width_irc)
    return ct_chunk, center_irc


class LunaDataset(Dataset):
    def __init__(self, 
                 val_stride: int = 0,
                 is_val_set: bool = None,
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
    
    def __len__(self):
        return len(self.candidates_info)
    
    def __getitem__(self, ix):
        candidate_info = self.candidates_info[ix]
        width_irc = (32, 48, 48)
        
        ct_chunk, center_irc = get_ct_raw_candidate(candidate_info.series_uid,
                                                    candidate_info.center_xyz,
                                                    width_irc)
        
        ct_chunk = torch.from_numpy(ct_chunk)
        ct_chunk = ct_chunk.to(dtype=torch.float32)
        ct_chunk = ct_chunk.unsqueeze(0)
        
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


if __name__ == '__main__':
    
    candidates = get_candidate_info_list()
    print(f'{len(candidates)} candidates nodules for {len(list(set([c.series_uid for c in candidates])))} files.')
    
    EXAMPLE_UID = '1.3.6.1.4.1.14519.5.2.1.6279.6001.511347030803753100045216493273'
    #EXAMPLE_UID = '1.3.6.1.4.1.14519.5.2.1.6279.6001.277445975068759205899107114231'
    
    # Convert RAW image to Array
    img = Ct(series_uid=EXAMPLE_UID)
    CtConversion.print_histogram(array=img.hu)
    
    # Create chunk around nodule coordinates
    example_candidate = [c for c in candidates if c.series_uid == EXAMPLE_UID and c.diameter_mm != 0.0][0]
    width_irc = (32, 48, 48)
    ct_chunk, center_irc = img.get_raw_candidate(center_xyz=example_candidate.center_xyz, 
                                                 width_irc=width_irc)
    print(f'Chunk size: {ct_chunk.shape} and center: {center_irc}')
    CtConversion.print_histogram(array=ct_chunk)
    
    # Print full pictire and chunk in one plot for comparison:
    IMG.img_by_chunk_sidebyside(img.hu, 
                                ct_chunk, 
                                center_irc, 
                                fig_title=EXAMPLE_UID, 
                                add_rectangular=[center_irc, round(example_candidate.diameter_mm)])
    
    # Test Luna Dataset
    luna_ds = LunaDataset()
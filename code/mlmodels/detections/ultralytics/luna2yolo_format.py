import pandas as pd
import glob 
import os
from typing import Dict
import numpy as np
from collections import namedtuple
from numpy.typing import ArrayLike
from typing import Dict, List, NamedTuple, Tuple, Union
from tqdm import tqdm 
from PIL import Image
import random 

import sys
sys.path.append('code')
from data import CtBase

PATH = ''
import os.path

def calculate_YOLOv8_format(ct: CtBase, ann: Dict, dim: int = 2):
    
    lower_coord = np.array([ann['bboxLowX'], ann['bboxLowY'], ann['bboxLowZ']])
    higher_coord = np.array([ann['bboxHighX'], ann['bboxHighY'], ann['bboxHighZ']])
    coord_xyz = [ann['coordX'], ann['coordY'], ann['coordZ']]
    origin_xyz = np.array(ct.origin_xyz)
    vx_size_xyz = np.array(ct.vx_size_xyz)
    direction = np.array(ct.direction)

    cri = ((coord_xyz - origin_xyz) @ np.linalg.inv(direction)) / vx_size_xyz
    
    x1y1z1 = ((lower_coord - origin_xyz) @ np.linalg.inv(direction)) / vx_size_xyz
    x2y2z2 =  ((higher_coord - origin_xyz) @ np.linalg.inv(direction)) / vx_size_xyz
    
    xyz_yolo = cri / [ct.hu.size()[2], ct.hu.size()[1], ct.hu.size()[0]]
    
    x1y1z1_yolo = x1y1z1 / [ct.hu.size()[2], ct.hu.size()[1], ct.hu.size()[0]]
    x2y2z2_yolo = x2y2z2 / [ct.hu.size()[2], ct.hu.size()[1], ct.hu.size()[0]]

    x = xyz_yolo[0]
    y = xyz_yolo[1]
    z = xyz_yolo[2]
    w = x2y2z2_yolo[0] - x1y1z1_yolo[0]
    h = x2y2z2_yolo[1] - x1y1z1_yolo[1]
    z = x2y2z2_yolo[2] - x1y1z1_yolo[2]

    return int(round(cri[2])), (x, y, w, h)


def write_empty_annotation(file_path, file_name):
    path = os.path.join(file_path, file_name)
    with open(path, 'w') as f:
            f.write('')


def write_annotation(file_path, file_name, ann):
    row_str = f"{1 if ann['cls'] else 0} {ann['xywh'][0]} {ann['xywh'][1]} {ann['xywh'][2]} {ann['xywh'][3]}\n"
    path = os.path.join(file_path, file_name)
    print(os.getcwd())
    
    if os.path.isfile(path):
        with open(path, 'a+') as f:
                f.write(row_str)
    else:
        with open(path, 'w') as f:
            f.write(row_str)

        
def write_image(file_path, file_name, img: ArrayLike):
    im = Image.fromarray(img).convert("L") # for grayscale
    im.save(os.path.join(file_path, file_name))


if __name__ == '__main__':
    
    DEV = False
    CLEAR_SPACE = True
    
    mhd = glob.glob(pathname= PATH + '.data/*/*.mhd')
    present_in_disk = [os.path.split(filepath)[-1][:-4] for filepath in mhd]
    df = pd.read_csv(PATH + '.data/annotations_with_malignancy.csv')

    if DEV:
        EXAMPLE_UID = '1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492'
        EXAMPLE_UID = '1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059'
        df = df[df['seriesuid'] == EXAMPLE_UID]

    
    for i, series_uid in tqdm(enumerate(df['seriesuid'].unique())):
        # choose in which folder it falls
        if i % 10 == 0:
            subset = 'val'
        else:
            subset = 'train'  
    
        if series_uid in present_in_disk:
            ct_df = df[df['seriesuid'] == series_uid]
            ct = CtBase(series_uid)
            
            # save slices with malignacy
            mal_slices = []
            annotations = ct_df[['seriesuid', 'mal_bool', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'bboxLowX','bboxLowY','bboxLowZ','bboxHighX','bboxHighY','bboxHighZ']].to_dict('records')
            for ann in tqdm(annotations, leave=False):
                slice_ix, x_y_w_h = calculate_YOLOv8_format(ct, ann)
                mal_slices.append(slice_ix)
                ann['cls'] = ann['mal_bool']
                ann['xywh'] = x_y_w_h  
                write_annotation(f'c.data/yolo_detection/labels/{subset}', f'{series_uid}_{slice_ix}.txt', ann)
                write_image(f'.data/yolo_detection/images/{subset}', f'{series_uid}_{slice_ix}.png', img=ct.hu[slice_ix].numpy())

            # save slices without malignacy (10 random slices)
            remove_ranges = []
            max_c = ct.hu.shape[2]
            for ms in mal_slices:
                remove_ranges.extend([list(range((max(0, ms-10), min(max_c, ms+10))))])
            possible_slices = [slice for slice in list(range(ct.hu.shape[2])) if slice not in remove_ranges]
            slices_to_export = random.choice(possible_slices, 9)
            for slice_ix in slices_to_export:
                write_image(f'.data/yolo_detection/images/{subset}', f'{series_uid}_{slice_ix}.png', img=ct.hu[slice_ix].numpy())
                write_empty_annotation(f'c.data/yolo_detection/labels/{subset}', f'{series_uid}_{slice_ix}.txt')

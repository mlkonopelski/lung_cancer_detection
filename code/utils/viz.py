from typing import Dict, List, NamedTuple, Optional, Union

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
import torch

class IMG:
    """
    Helper class to visualize different images
    """
    @staticmethod
    def single_image(img_array: ArrayLike, img_title: str = None):
        fig = plt.figure(figsize=(8, 8))
        sub = fig.add_subplot(1, 1, 1)
        if img_title:
            sub.set_title(img_title)
        sub.imshow(img_array, interpolation='nearest')
        plt.show()
    
    @staticmethod
    def visualize_augmentations(func, series_uid, center_xyz, width_irc, augmentations: Dict):
        viz_count = len(augmentations.keys())
        if viz_count < 4:
            shape = (1, viz_count + 1)
        elif viz_count >= 4 and viz_count < 8:
            shape = (2, int((viz_count + 1)/ 2))
        elif viz_count >= 8 and viz_count < 12:
            shape (3, viz_count + 1)
        else:
            raise Exception('Not developed for augmentations > 11')
        
        fig = plt.figure(figsize=(10, 8))
        
        # first image 
        img_chunk, _ = func(augmentation={}, 
                            series_uid=series_uid,
                            center_xyz=center_xyz,
                            width_irc=width_irc,
                            use_cache=False)
        sub = fig.add_subplot(shape[0], shape[1], 1)
        sub.set_title(label='Original')
        sub.imshow(X=img_chunk[0][16], interpolation='nearest')
        
        # All augmentations
        for i, (key, value) in enumerate(augmentations.items()):
             img_chunk, _ = func(augmentation={key: value}, 
                            series_uid=series_uid,
                            center_xyz=center_xyz,
                            width_irc=width_irc,
                            use_cache=False)
             sub = fig.add_subplot(shape[0], shape[1], i+2)
             sub.set_title(label=f'{key}: {value}')
             sub.imshow(X=img_chunk[0][16], interpolation='nearest')            
        
    def img_by_chunk_sidebyside(img: ArrayLike,
                                chunk: ArrayLike, 
                                center_irc: NamedTuple,
                                fig_title: str,
                                add_rectangular: List[NamedTuple, ]
                                ):
        
        TITLE_LIST = ['Index', 'Row', 'Column']
        
        # Recatangular x, y
        diameter = add_rectangular[1]
        x = add_rectangular[0][2] - diameter
        y = add_rectangular[0][1] - diameter
        z = add_rectangular[0][0] - diameter
        
        fig = plt.figure(figsize=(10, 8))
        if fig_title:
            fig.suptitle(fig_title, fontsize=15)
        
        plot_position = 1
        for i, arr in enumerate([img, chunk]):
            for d in range(3):
                sub = fig.add_subplot(2, 3, plot_position)
                sub.set_title(f'{TITLE_LIST[d]} = {center_irc[d]}')
                
                sub.axis('off')
                
                # TODO: This if[if] is not the most readible. 
                # I should refactor it maybe to do each slice separetly instead of loop.
                if d == 0:
                    x1, y1 = x, y
                    if i == 0:
                        slice = arr[center_irc[d]]
                    else:
                        slice = arr[16]
                elif d == 1:
                    x1, y1 = x, z
                    if i == 0:
                        slice = arr[:, center_irc[d]]
                    else:
                        slice = arr[:, 24]
                elif d == 2:
                    x1, y1 = y, z
                    if i == 0:
                       slice = arr[:, :, center_irc[d]]
                    else:
                        slice = arr[:, :, 24]
                
                # TODO: This solution needs better approach because pictures are skewed and 
                # it's hard to say where should be new bounding box exactly
                # if slice.shape != (512, 512):
                #     slice = cv2.resize(slice, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

                sub.imshow(slice, interpolation='nearest')
                if i == 0:        
                    rect = patches.Rectangle((x1, y1), diameter * 2, diameter * 2, linewidth=1, edgecolor='r', facecolor='none')
                    sub.add_patch(rect)
                    
                plot_position += 1
                
        plt.show()
    
    def img_by_augmentation(img, augmentations: Union[None, Dict] = None):
        # TODO: Create visualizations of different augmentations
        # idea: original picture in left upper corner and depending on the amount of augmentations 
        # create a rectangular grid. Also include options to default
        ...
    
    @staticmethod
    def visualize_mask(img_array, mask_array):
        fig = plt.figure(figsize=(8, 8))
        sub = fig.add_subplot(1, 1, 1)
        sub.imshow(img_array, cmap='gray', interpolation='nearest')
        plt.imshow(mask_array.to(dtype=torch.float), cmap='bwr', alpha=0.5)
        plt.show()
        
        
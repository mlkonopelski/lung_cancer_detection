from typing import Dict, List, NamedTuple, Optional, Union

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike


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
import numpy as np 
import os 
import re 
import sys 



def pad(image: np.array, factor: int = 32, border: int = cv2.BORDER_REFLEXT_101) -> tuple:
    """
    Pad image so it is divided by factor
    """
    height, width = image.shape[:2]
    if height % factor == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = factor - height % factor 
        y_min_pad = y_pad // 2
        y_max_pad = y_pad - y_min_pad 
    
    if width % factor == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = factor - width % factor 
        x_min_pad = x_pad // 2
        x_max_pad = x_pad - x_min_pad

    padded_image = cv2.copyMakeBorder(image, y_min_pad, y_max_pad, x_min_pad, x_max_pad, border)

    return padded_image, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)


def unpad(image: np.array, pads: list) -> np.array:
    """
    Crop padded image based on pads list
    """
    x_min_pad, y_min_pad, x_max_pad, y_max_pad = pads

    height, width = image.shape[:2]

    return image[y_min_pad : height -  y_max_pad, x_min_pad : width - x_max_pad]
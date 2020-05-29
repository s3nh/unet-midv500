import numpy as np 
import os 
import sys 
import re 
from pathlib import Path
from typing import Tuple, List
import cv2
import json

def create_mask(size: Tuple[int, int], label: dict) -> np.ndarray:
    mask = np.zeros(size)
    poly = np.array(label["quad"])

    return cv2.fillPoly(mask, [poly], (255, ))

def main():
    DATA_PATH = 'data'
    images = sorted(Path(DATA_PATH).rglob('*.tif'))
    print("Number of images {}".format(len(images)))
    labels = sorted(Path(DATA_PATH).rglob("*.json"))
    print("Number of labels {}".format(len(labels)))

    # Images to ids 
    images_id2path = {x.stem : x for x in images}
    labels_id2path = {x.stem : x for x in labels}

    assert len(images_id2path) == len(labels_id2path)

    path_to_create = ['data_processed', 'data_processed/labels', 'data_processed/images']
    for path in path_to_create:
        os.makedirs(path, exist_ok=True)

    image_output_path = 'data_processed/images'
    label_output_path = 'data_processed/labels'

    for _id in images_id2path.keys():
        image_path = images_id2path[_id]
        label_path = labels_id2path[_id]
        image = cv2.imread(str(image_path))
        # Get image shape
        height, width = image.shape[:2]

        with open(label_path) as l:
            label = json.load(l)
        print(label)
        if "quad" not in label.keys():
            continue 

        mask = create_mask((height, width), label)
        cv2.imwrite(os.path.join(image_output_path , f"{_id}.jpg"), image)
        cv2.imwrite(os.path.join(label_output_path , f"{_id}.png"), mask)

if __name__ == "__main__":
    main()
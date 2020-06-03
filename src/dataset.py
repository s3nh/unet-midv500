import time 
import torch
import numpy as np 
from pathlib import Path
#import time.time
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models 
from typing import List, Tuple, Dict, Any
import albumentations  
from src.utils import pad, unpad
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image

class MidvDataset(Dataset):

    def __init__(self, samples: List[Tuple[Path, Path]], transform = albumentations.Compose) -> None:
        self.samples = samples 
        self.transform = transform 


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path, mask_path = self.samples[idx]
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        sample = self.transform(image = image, mask = mask)
        image, mask = sample["image"], sample["mask"]

        mask = (mask > 0).astype(np.uint8)

        mask = torch.from_numpy(mask)

        return {
            "image_id" : Path(image_path).stem, 
            "features" : tensor_from_rgb_image(image), 
            "masks" : torch.unsqueeze(mask, 0).float(),
        }



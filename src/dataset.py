import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models 
from typing import List, Tuple, Dict, Any
import albumentations  


class MidvDataset(Dataset):

    def __init__(self, samples: List[Tuple[Path, Path]], transform = None) -> None:
        
        self.samples = samples 
        self.transform = transform 


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path, mask_path = self.samples[idx]

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        height, width = image.shape[:2]


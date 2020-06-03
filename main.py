import torch.nn as nn 
import albumentations
import torch
import torch.nn.functional as F
import typing
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.unet import UNet
from src.dataset import MidvDataset
from src.train import train_model
from pathlib import Path 
from models.loss import dice_loss
from torchsummary import summary 
from pytorch_toolbelt.losses import JaccardLoss, BinaryFocalLoss
import os 


os.makedirs('trained_model', exist_ok = True)


def main():
    torch.cuda.empty_cache()

    device = 'cuda'
    model = UNet(n_class = 1)
    model = model.cuda()
   
    list_images = list(Path('data_processed/images').rglob('*.jpg'))
    list_masks = list(Path('data_processed/labels').rglob('*.png'))
    list_masks = [str(el) for el in list_masks]
    list_images = [str(el) for el in list_images]
    samples = list(zip(list_images, list_masks))
    samples = [tuple(el) for el in samples]
    optimizer = torch.optim.RMSprop(model.parameters(), lr = 1e-4, momentum = 0.9)
    scheduler = StepLR(optimizer, step_size = 30, gamma = 0.1) 
    train_model(model = model, optimizer = optimizer, scheduler = None, num_epochs = 25, samples = samples)

if __name__ == "__main__":
    main()

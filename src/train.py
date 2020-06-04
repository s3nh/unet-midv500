from models.unet import UNet
from src.dataset import MidvDataset
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
from pathlib import Path 
from models.loss import dice_loss
from torchsummary import summary 
from pytorch_toolbelt.losses import JaccardLoss, BinaryFocalLoss
import os 
import time


def train_model(model , optimizer, scheduler , num_epochs, samples) -> None:
    dataset = MidvDataset(samples = samples, transform = albumentations.Compose( [albumentations.LongestMaxSize(max_size=512 , p=1)], p=1  ))
    train_dt, test_dt = torch.utils.data.random_split(dataset,[ int(0.8* len(dataset)), int(0.2* len(dataset))])
    train_loader = DataLoader(train_dt,  batch_size = 4, shuffle = True, num_workers = 0)
    test_loader = DataLoader(test_dt, shuffle = True, batch_size = 4)
    model = model.cuda()
    criterion = JaccardLoss(mode="binary", from_logits=True) 
    
    model.train()
    for epoch in range(num_epochs):
        for i,  res in enumerate(train_loader, 0):
            inputs = res['features'].float().cuda()
            labels = res['masks'].long().cuda()
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f' loss values in epoch {epoch} iter {i} : {loss}')
        print(f" Loss functions after {epoch} : {loss}")
        if epoch % 5 == 0:
            save_path = os.path.join('trained_model', f'unet_midv_adam_{epoch}.pt') 
            torch.save(model.state_dict(), save_path)
    return model 


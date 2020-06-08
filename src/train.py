from models.resnet18_unet import ResNetUNet
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
import matplotlib.pyplot as plt
from models.unet import UNet
from src.dataset import MidvDataset
from pathlib import Path 
from models.loss import dice_loss
from torchsummary import summary 
from pytorch_toolbelt.losses import JaccardLoss, BinaryFocalLoss
import os 
import time
import numpy as np 


def train_model(model , optimizer, scheduler , num_epochs, samples) -> None:
    best_loss = 1.0
    dataset = MidvDataset(samples = samples, transform = albumentations.Compose( [albumentations.LongestMaxSize(max_size=768 , p=1)], p=1  ))
    train_dt, test_dt = torch.utils.data.random_split(dataset,[ int(0.8* len(dataset)), int(0.2* len(dataset))])
    train_loader = DataLoader(train_dt,  batch_size = 1, shuffle = True, num_workers = 0)
    test_loader = DataLoader(test_dt, shuffle = True, batch_size = 1)
    model = model.cuda()
    _act = nn.Sigmoid()
    #criterion =  JaccardLoss(mode ='binary', from_logits = True) 
    #criterion = torch.nn.BCELoss().cuda()
    criterion = nn.BCEWithLogitsLoss()
    val_loss = []
    for epoch in range(num_epochs):
        model.train()
        for i, res in enumerate(train_loader, 0):
            inputs = res['features'].to(device = 'cuda', dtype = torch.float32)
            labels = res['masks'].to(device = 'cuda', dtype = torch.float32)
            output = model(inputs)
            loss = criterion(output, labels) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f'loss {loss}')

        model.eval()

        for t_i, t_res in enumerate(test_loader, 0):
            t_inputs = t_res['features'].to(device = 'cuda', dtype = torch.float32)
            t_labels = t_res['masks'].to(device = 'cuda', dtype = torch.float32)
            with torch.no_grad():
                val_pred = model(t_inputs)
            v_loss = criterion(val_pred, t_labels)
            val_loss.append(v_loss)
        v_loss_mean = np.sum(val_loss)/len(val_loss)
        scheduler.step(v_loss_mean)
           

        print(f' Epoch {epoch} Validation loss {v_loss_mean}')

        if v_loss_mean < best_loss:
            save_path = os.path.join('trained_model', f'unet_best_{epoch}.pt') 
            torch.save({
                    'epoch' : epoch, 
                    'model_state_dict' : model.state_dict(), 
                    'optimizer_state_dict' : optimizer.state_dict(), 
                    'scheduler_state_dict' : scheduler.state_dict(), 
                    'loss' : loss, 
                    'val_loss' : v_loss_mean},
                    save_path 
                )
            best_loss = v_loss_mean
    return model 

import torch.nn as nn 
import albumentations
import torch
import torch.nn.functional as F
import typing
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.unet import UNet
from src.dataset import MidvDataset
from pathlib import Path 
from models.loss import dice_loss
from torchsummary import summary 
from pytorch_toolbelt.losses import JaccardLoss, BinaryFocalLoss
import os 




os.makedirs('trained_model', exist_ok = True)

device = 'cuda'

def train_model(model , optimizer, scheduler , num_epochs, samples) -> None:
    dataset = MidvDataset(samples = samples, transform = albumentations.Compose( [albumentations.LongestMaxSize(max_size=512 , p=1)], p=1  ))
    dataloader = DataLoader(dataset,  batch_size = 4, shuffle = True, num_workers = 0)
    model = model.cuda()
    criterion = JaccardLoss(mode="binary", from_logits=True) 

    for epoch in range(num_epochs):
        model.train()

        for i,  res in enumerate(dataloader, 0):
            inputs = res['features'].float().cuda()
            labels = res['masks'].cuda()
            optimizer.zero_grad()

            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f' loss values in epoch {epoch} iter {i} : {loss}')
        print(f" Loss functions after {epoch} : {loss}")
        save_path = os.path.join('trained_model', f'unet_midv_{epoch}.pt')
        torch.save(model.state_dict(), save_path)
        

        # Saving general checkpoint 
    #model.load_state_dict(best_model_wts)
    return model 


def main():
    torch.cuda.empty_cache()

    device = 'cuda'
    model = UNet(n_class = 1)
    model = model.cuda()
   
    #summary(model, input_size = (3, 764, 764))

    list_images = list(Path('data_processed/images').rglob('*.jpg'))
    list_masks = list(Path('data_processed/labels').rglob('*.png'))
    list_masks = [str(el) for el in list_masks]
    list_images = [str(el) for el in list_images]
    samples = list(zip(list_masks, list_images))
    samples = [tuple(el) for el in samples]
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.99)
    train_model(model = model, optimizer = optimizer, scheduler = None, num_epochs = 10, samples = samples)

if __name__ == "__main__":
    main()

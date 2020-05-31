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



device = 'cuda'

def train_model(model , optimizer, scheduler , num_epochs, samples) -> None:
    dataset = MidvDataset(samples = samples, transform = albumentations.Compose( [albumentations.LongestMaxSize(max_size=768 , p=1)], p=1  ))
    dataloader = DataLoader(dataset,  batch_size = 1, shuffle = True, num_workers = 0)
    model = model.cuda()
    for epoch in range(num_epochs):
        #since = time.time()
        model.train()

        #metrics = defaultdict(float)
        for res in dataloader:
            inputs = res['features'].float().cuda()
            
            labels = res['masks'].float().cuda()
            optimizer.zero_grad()

            output = model(inputs)
            loss = dice_loss(output, labels, smooth = 0.01)
            loss.backward()
            optimizer.step()
        print(f" Loss functions after {epoch} : {loss}")
        torch.save(model.state_dict(), f'model_state_dict_{epoch}.pt')
        
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
    optimizer = Adam(model.parameters(), lr = 0.001)
    train_model(model = model, optimizer = optimizer, scheduler = None, num_epochs = 100, samples = samples)

if __name__ == "__main__":
    main()

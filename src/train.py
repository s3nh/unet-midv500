from models.unet import UNet
from src.dataset import MidvDataset

import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
from models.loss import dice_loss


## TODO: Add helpers and metrics 
## TODO: compress to pytorch lightning

def train_model(model , optimizer, scheduler , num_epochs, samples) -> None:
    dataset = MidvDataset(samples = samples, transform = albumentations.Compose( [albumentations.LongestMaxSize(max_size=128, p=1)], p=1  ))

    dataloader = DataLoader(dataset,  batch_size = 1, shuffle = True, num_workers = 0)

    model = model.cuda()
    for epoch in range(num_epochs):
        #since = time.time()
        model.train()

        #metrics = defaultdict(float)
        for res in dataloader:
            inputs = res['features'].float().cuda()
            print(inputs.shape)
            labels = res['masks'].float().cuda()
            print(labels.shape)
            optimizer.zero_grad()

            output = model(inputs)
            loss = dice_loss(output, labels)
            loss.backward()
            optimizer.step()


    #model.load_state_dict(best_model_wts)
    return model 
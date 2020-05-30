from models.unet import UNet
from src.dataset import MidvDataset


import torch 
import torch.nn as nn
import torch.nn.functional as F
## TODO: Add helpers and metrics 
## TODO: compress to pytorch lightning

def train_model(model, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        since = time.time()

        metrics = defaultdict(float)
        epoch_samples = 0

        for inputs, labels in dataloader[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = calc_loss(ouput, labels, metrics)
            loss.backward()
            optimizer.step()


    #model.load_state_dict(best_model_wts)
    return model 

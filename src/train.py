from models.unet import UNet
import torch 
import torch.nn as nn
import torch.nn.functional as F
## TODO: Add helpers and metrics 

def train_model(model, optimizer, scheduler, num_epochs):
    for epoch in rane(num_epochs):
        since = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()
            else:
                model.eval()

        metrics = defaultdict(float)
        epoch_samples = 0

        for inputs, labels in dataloader[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = calc_loss(ouput, labels, metrics)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            epoch_samples += inputs.size(0)

        epoch_loss = metrics['loss'] / epoch_samples
        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss 
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model 

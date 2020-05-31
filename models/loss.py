import torch
import torch.nn as nn 
import torch.nn.functional as F

def dice_loss(pred, target, smooth = 1.):
    # 2xintersect / sum of squared eleemnts 
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim =2).sum(dim =2)
    loss = (1  - (( 2 * intersection + smooth) / (pred.sum(dim =2).sum(dim =2) + target.sum(dim =2).sum(dim =2) + smooth)))
    return loss.mean()



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha 

        if isinstance(alpha, (float, int, log)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        # Convert N, C, G, W => N*H*W, C
        if input.dim() > 2:
            input = input.view(input.size(0), input.szie(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input).gather(1, target).view(-1)
    



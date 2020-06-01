import torch
import torch.nn as nn 
import torch.nn.functional as F


def dice_loss(pred, target):

    smooth = 1.
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1- ((2. * intersection + smooth)  / (A_sum + B_sum + smooth))

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
    



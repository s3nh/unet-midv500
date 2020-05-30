import torch 
import torch.nn as nn 
import torchvision.models
from models.unet import dconv
from models.unet import swish
import typing

# conv to swish  block 
def cnrelu(in_channels : int = 3, out_channels : int = 64, kernel : int = 3, padding : int = 1) -> None:
    return nn.Sequential(
        nn.Conv2d(in_channels =  in_channels, out_channels = out_channels), 
        swish(),
    )


class ResNetUnet(nn.Module):
    def __init__(self, n_class: int):
        super().__init__()

        self.base_model = torchvision.models.resnet18(pretrained = True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # N, 64, x.H/2, x.W/2
        self.layer0_n = cnrelu(64, 64, 1, 0)

        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1_n = cnrelu(64, 64, 1, 0)

        self.layer2 = *self.base_layers[5]
        self.layer2_n = cnrelu(128, 128, 1, 0)

        self.layer3 = *self.base_layers[6]
        self.layer3_n = cnrelu(256, 256, 1, 0)

        self.layer4 =  self.base_layers[7]
        self.layer4_n = cnrelu(512, 512, 1, 0)

        self.upsample =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 

        self.conv_up3 =  cnrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = cnrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = cnrelu(64 + 256, 256, 3, 1)
        self.conv_up0 =  cnrelu(64 + 256, 128, 3, 1)




